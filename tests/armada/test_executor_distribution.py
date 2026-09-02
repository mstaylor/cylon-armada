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

"""Unit tests for ArmadaExecutor.run() (SP1 Task 4).

A FMIBridge spy records every collective call so we can assert the executor
genuinely dispatches through the bridge's collectives — the #1 correctness
gate per the plan — rather than silently falling back to in-process invoke.

Run: pytest tests/armada/test_executor_distribution.py -x
"""

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import pytest

from armada.executor import ArmadaExecutor
from armada.operator import ArmadaOperator
from cylon_armada.dag_compiler import CollectivePattern


def _schema():
    import pyarrow as pa
    return pa.schema([pa.field("v", pa.int64())])


class SpyBridge:
    """Records every collective call; never falls back to in-process behavior."""

    def __init__(self, scatter_result=None, reduce_result=None, world_size=2):
        self.calls = []
        self.available = True
        self.world_size = world_size
        self._scatter_result = scatter_result
        self._reduce_result = reduce_result
        self._gather_result = None

    def scatter(self, tables, root=0):
        self.calls.append(("scatter", tables, root))
        return self._scatter_result

    def gather(self, table, root=0):
        self.calls.append(("gather", table, root))
        return self._gather_result

    def allgather(self, table):
        self.calls.append(("allgather", table))
        return None

    def broadcast(self, table, root=0):
        self.calls.append(("broadcast", table, root))
        return table

    def reduce_table(self, column, op, root=0):
        self.calls.append(("reduce_table", column, op, root))
        return self._reduce_result


def test_run_dispatches_scatter_then_reduce_through_the_bridge():
    s = _schema()
    scattered_value = "scattered-shard"
    reduced_value = "reduced-total"
    bridge = SpyBridge(scatter_result=scattered_value, reduce_result=reduced_value)

    scatter_op = ArmadaOperator("Preprocess", CollectivePattern.Scatter, s, s,
                                fn=lambda x: f"prepared({x})")
    reduce_op = ArmadaOperator("Retrieve", CollectivePattern.Reduce, s, s,
                               fn=lambda x: f"column({x})")
    seq = scatter_op | reduce_op

    executor = ArmadaExecutor(bridge)
    result = executor.run(seq, input_tables="raw-input", ctx=None)

    assert [c[0] for c in bridge.calls] == ["scatter", "reduce_table"]

    # Scatter moves data BEFORE fn runs: bridge.scatter sees the raw input, not
    # fn's output — each rank only gets its own shard to compute on afterward.
    scatter_call = bridge.calls[0]
    assert scatter_call[1] == "raw-input"
    assert scatter_call[2] == 0

    reduce_call = bridge.calls[1]
    assert reduce_call[1] == f"column(prepared({scattered_value}))"
    assert reduce_call[3] == 0

    assert result == reduced_value


def test_run_raises_when_bridge_unavailable():
    s = _schema()
    bridge = SpyBridge()
    bridge.available = False

    seq = ArmadaOperator("Preprocess", CollectivePattern.Scatter, s, s, fn=lambda x: x)
    executor = ArmadaExecutor(bridge)

    with pytest.raises(RuntimeError):
        executor.run(seq, input_tables="raw-input", ctx=None)


def test_run_point_to_point_stays_local_no_collective_call():
    s = _schema()
    bridge = SpyBridge()

    seq = ArmadaOperator("Reason", CollectivePattern.PointToPoint, s, s,
                         fn=lambda x: f"reasoned({x})")
    executor = ArmadaExecutor(bridge)
    result = executor.run(seq, input_tables="local-input", ctx=None)

    assert bridge.calls == []
    assert result == "reasoned(local-input)"


def test_run_broadcast_dispatches_through_the_bridge():
    s = _schema()
    bridge = SpyBridge()

    seq = ArmadaOperator("MemoryUpsert", CollectivePattern.Broadcast, s, s,
                         fn=lambda x: f"ack({x})")
    executor = ArmadaExecutor(bridge)
    result = executor.run(seq, input_tables="findings", ctx=None)

    assert [c[0] for c in bridge.calls] == ["broadcast"]
    assert bridge.calls[0][1] == "ack(findings)"
    assert result == "ack(findings)"


def test_run_scattergather_computes_locally_then_gathers_no_scatter_call():
    s = _schema()
    gathered_value = ["shard-0-embedding", "shard-1-embedding"]
    bridge = SpyBridge()
    bridge._gather_result = gathered_value

    seq = ArmadaOperator("Embed", CollectivePattern.ScatterGather, s, s,
                         fn=lambda x: f"embedded({x})")
    executor = ArmadaExecutor(bridge)
    result = executor.run(seq, input_tables="my-shard", ctx=None)

    # ScatterGather computes locally first (each rank already has its own
    # shard from the prior Scatter step) then only gathers — no redundant
    # scatter call at this operator.
    assert [c[0] for c in bridge.calls] == ["gather"]
    assert bridge.calls[0][1] == "embedded(my-shard)"
    assert result == gathered_value


def test_world_size_one_equals_plain_sequence_invoke():
    """The swap-equivalence seed for E-SP2: at world_size==1, executor.run()
    must produce the exact same result as calling seq.invoke() directly, with
    no bridge involved at all — proving single_rank execution really is plain
    local invocation, not a distribution-shape-preserving approximation of it.
    """
    s = _schema()
    bridge = SpyBridge(world_size=1)

    seq = (
        ArmadaOperator("Preprocess", CollectivePattern.Scatter, s, s, fn=lambda x: x + 1)
        | ArmadaOperator("Embed", CollectivePattern.ScatterGather, s, s, fn=lambda x: x * 2)
        | ArmadaOperator("Retrieve", CollectivePattern.Reduce, s, s, fn=lambda x: x - 3)
        | ArmadaOperator("Reason", CollectivePattern.PointToPoint, s, s, fn=lambda x: x + 100)
        | ArmadaOperator("MemoryUpsert", CollectivePattern.Broadcast, s, s, fn=lambda x: x)
    )

    executor = ArmadaExecutor(bridge)
    executor_result = executor.run(seq, input_tables=5, ctx=None)
    invoke_result = seq.invoke(5)

    assert bridge.calls == []
    assert executor_result == invoke_result == (((5 + 1) * 2) - 3) + 100