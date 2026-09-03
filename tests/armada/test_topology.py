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

"""Unit tests for plan-derived connection topology (armada/topology.py).

Run: pytest tests/armada/test_topology.py -v
"""

import math
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import pytest

from armada.topology import (
    binomial_tree_peers,
    format_peer_list,
    format_peer_map,
    linear_gather_peers,
    peer_map,
    recursive_doubling_peers,
    required_peers,
)
from cylon_armada.dag_compiler import CollectivePattern

ALL_TREE = {CollectivePattern.Scatter, CollectivePattern.Broadcast}
WITH_GATHER = {CollectivePattern.Scatter, CollectivePattern.ScatterGather,
               CollectivePattern.Reduce, CollectivePattern.Broadcast}


def _edges(world_size, patterns, roots=(0,)):
    edges = set()
    for r in range(world_size):
        for p in required_peers(world_size, r, patterns, roots):
            edges.add(tuple(sorted((r, p))))
    return edges


def test_peer_sets_are_symmetric():
    """If a needs b, b must need a — an asymmetric set means one side waits forever."""
    for world_size in (2, 3, 4, 5, 7, 8, 16, 33):
        for r in range(world_size):
            for p in required_peers(world_size, r, WITH_GATHER):
                assert r in required_peers(world_size, p, WITH_GATHER), (
                    f"world_size={world_size}: rank {r} wants {p}, but not vice versa"
                )


def test_never_includes_self():
    for world_size in (2, 5, 8, 17):
        for r in range(world_size):
            assert r not in required_peers(world_size, r, WITH_GATHER)


def test_world_size_one_needs_nobody():
    assert required_peers(1, 0, WITH_GATHER) == set()


def test_tree_degree_is_logarithmic_not_linear():
    """The whole point: non-root ranks stay at ~log2(N), not N-1."""
    world_size = 64
    for r in range(1, world_size):
        peers = required_peers(world_size, r, ALL_TREE)
        assert len(peers) <= 2 * math.ceil(math.log2(world_size))


def test_gather_makes_root_a_hub_but_not_the_others():
    """Linear gatherv fans into root, so root keeps degree N-1 while others don't."""
    world_size = 32
    assert len(required_peers(world_size, 0, WITH_GATHER)) == world_size - 1
    for r in range(1, world_size):
        assert len(required_peers(world_size, r, WITH_GATHER)) < world_size - 1


def test_restricted_edge_count_beats_full_mesh():
    world_size = 64
    full_mesh = world_size * (world_size - 1) // 2
    assert len(_edges(world_size, WITH_GATHER)) < full_mesh / 5


def test_non_zero_root_shifts_the_tree():
    world_size = 8
    assert binomial_tree_peers(world_size, 0, root=0) != binomial_tree_peers(world_size, 0, root=3)


def test_multiple_roots_union_covers_each_root():
    world_size = 8
    both = required_peers(world_size, 5, ALL_TREE, roots=(0, 3))
    assert binomial_tree_peers(world_size, 5, root=0) <= both
    assert binomial_tree_peers(world_size, 5, root=3) <= both


def test_recursive_doubling_always_included_for_barriers():
    """A barrier is an allreduce and can fire regardless of operator patterns."""
    world_size = 8
    peers = required_peers(world_size, 3, {CollectivePattern.PointToPoint})
    assert recursive_doubling_peers(world_size, 3) <= peers


def test_linear_gather_peers_shape():
    assert linear_gather_peers(4, 0) == {1, 2, 3}
    assert linear_gather_peers(4, 2) == {0}


def test_format_peer_list_is_sorted_and_comma_separated():
    assert format_peer_list({3, 1, 2}) == "1,2,3"
    assert format_peer_list(set()) == ""


def test_peer_map_covers_every_rank():
    world_size = 6
    mapping = peer_map(world_size, WITH_GATHER)
    assert sorted(mapping) == list(range(world_size))


def test_format_peer_map_round_trips_every_row():
    """Each rank gets the whole map because its real rank is only assigned by the
    Redis INCR counter after this value has to be set."""
    mapping = peer_map(4, WITH_GATHER)
    rendered = format_peer_map(mapping)

    parsed = {}
    for row in rendered.split(";"):
        rank_text, _, peers_text = row.partition(":")
        parsed[int(rank_text)] = {int(p) for p in peers_text.split(",") if p}

    assert parsed == mapping


def test_format_peer_map_handles_a_rank_with_no_peers():
    assert format_peer_map({0: set(), 1: {0}}) == "0:;1:0"


# ---------------------------------------------------------------------------
# Derivation straight off a compiled ExecutionPlan
# ---------------------------------------------------------------------------

def _seq():
    import pyarrow as pa
    from armada.operator import ArmadaOperator
    s = pa.schema([pa.field("v", pa.int64())])
    return (
        ArmadaOperator("Preprocess", CollectivePattern.Scatter, s, s, fn=lambda x: x)
        | ArmadaOperator("Retrieve", CollectivePattern.Reduce, s, s, fn=lambda x: x)
    )


def test_required_peer_map_matches_the_plans_own_patterns():
    from armada.executor import lower, required_peer_map

    world_size = 8
    seq = _seq()
    rendered = required_peer_map(seq, world_size)

    expected = format_peer_map(peer_map(
        world_size, set(lower(seq).assignments.values()), (0,)))
    assert rendered == expected


def test_required_peer_map_is_empty_for_a_single_rank():
    from armada.executor import required_peer_map
    assert required_peer_map(_seq(), 1) == ""


def test_required_peer_map_covers_every_rank():
    from armada.executor import required_peer_map
    rendered = required_peer_map(_seq(), 5)
    assert sorted(int(row.split(":")[0]) for row in rendered.split(";")) == [0, 1, 2, 3, 4]