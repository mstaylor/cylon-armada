# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""InputPlacement and per-rank shard bounds (E-SP1 v2).

PreDistributed exists for a workload whose upstream stage already ran per rank
— AstroMAE inference over a slice of the survey — so each rank holds its own
input and Preprocess's Scatter would be a redundant round trip through root.
What is proved here: the Scatter's *data movement* is the only thing that
changes, every operator still runs, and the shard split covers an indivisible
population exactly once.

Run: pytest tests/armada/test_input_placement.py -v
"""

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import pyarrow as pa
import pytest

from armada.executor import ArmadaExecutor, InputPlacement
from armada.operator import ArmadaOperator
from armada.run_cosmic_local import shard_bounds
from cylon_armada.dag_compiler import CollectivePattern

TEXT = pa.schema([pa.field("raw_text", pa.large_utf8())])


class SpyBridge:
    def __init__(self, world_size):
        self.world_size = world_size
        self.available = world_size > 1
        self.calls = []

    def scatter(self, tables, root=0):
        self.calls.append("scatter")
        return tables[0] if isinstance(tables, list) else tables

    def gather(self, table, root=0):
        self.calls.append("gather")
        return [table]

    def broadcast(self, table, root=0):
        self.calls.append("broadcast")
        return table


def _sequence(recorder):
    def preprocess(table):
        recorder.append("preprocess")
        return pa.table({"raw_text": [t.upper() for t in table.column(0).to_pylist()]},
                        schema=TEXT)

    def reason(table):
        recorder.append("reason")
        return table

    return (ArmadaOperator("Preprocess", CollectivePattern.Scatter, TEXT, TEXT, fn=preprocess)
            | ArmadaOperator("Reason", CollectivePattern.PointToPoint, TEXT, TEXT, fn=reason))


# ---------------------------------------------------------------------------
# Shard bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 5, 8, 16])
def test_shards_cover_every_item_exactly_once(world_size):
    n = 1253
    covered = []
    for rank in range(world_size):
        start, stop = shard_bounds(n, world_size, rank)
        covered.extend(range(start, stop))
    assert covered == list(range(n))


@pytest.mark.parametrize("world_size", [3, 5, 7])
def test_indivisible_population_spreads_the_remainder(world_size):
    """An arbitrary N must not drop the tail — sizes differ by at most one."""
    n = 1253
    sizes = [stop - start
             for start, stop in (shard_bounds(n, world_size, r) for r in range(world_size))]
    assert sum(sizes) == n
    assert max(sizes) - min(sizes) <= 1


def test_more_ranks_than_items_gives_some_ranks_nothing():
    sizes = [stop - start for start, stop in (shard_bounds(3, 8, r) for r in range(8))]
    assert sum(sizes) == 3
    assert sizes.count(0) == 5


# ---------------------------------------------------------------------------
# Placement dispatch
# ---------------------------------------------------------------------------

def test_pre_distributed_skips_the_scatter_but_still_runs_every_operator():
    ran = []
    bridge = SpyBridge(world_size=4)
    table = pa.table({"raw_text": ["a", "b"]}, schema=TEXT)

    result = ArmadaExecutor(bridge).run(_sequence(ran), input_tables=table, ctx=None,
                                        placement=InputPlacement.PreDistributed)

    assert "scatter" not in bridge.calls
    assert ran == ["preprocess", "reason"]
    assert result.column("raw_text").to_pylist() == ["A", "B"]


def test_centralized_still_scatters():
    ran = []
    bridge = SpyBridge(world_size=4)
    tables = [pa.table({"raw_text": ["a"]}, schema=TEXT) for _ in range(4)]

    ArmadaExecutor(bridge).run(_sequence(ran), input_tables=tables, ctx=None,
                               placement=InputPlacement.Centralized)

    assert bridge.calls == ["scatter"]
    assert ran == ["preprocess", "reason"]


def test_centralized_is_the_default():
    bridge = SpyBridge(world_size=4)
    tables = [pa.table({"raw_text": ["a"]}, schema=TEXT) for _ in range(4)]
    ArmadaExecutor(bridge).run(_sequence([]), input_tables=tables, ctx=None)
    assert bridge.calls == ["scatter"]


@pytest.mark.parametrize("placement", list(InputPlacement))
def test_single_rank_result_is_identical_under_either_placement(placement):
    """At world_size 1 there is nothing to distribute, so placement cannot
    change the answer — the swap-equivalence control must survive the flag."""
    ran = []
    bridge = SpyBridge(world_size=1)
    table = pa.table({"raw_text": ["a", "b"]}, schema=TEXT)

    result = ArmadaExecutor(bridge).run(_sequence(ran), input_tables=table, ctx=None,
                                        placement=placement)

    assert bridge.calls == []
    assert result.column("raw_text").to_pylist() == ["A", "B"]