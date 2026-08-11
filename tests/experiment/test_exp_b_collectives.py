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

"""Unit tests for the Experiment B collective benchmark timing core.

These use a mock communicator whose collectives sleep a known duration and an
injected payload builder, so the timing logic is validated without the pycylon /
FMI native stack. Run: pytest tests/experiment/test_exp_b_collectives.py -x
"""

import os
import sys
import time

# Put the shared scripts root on the path (the modules import each other as
# `experiment.*`, so target/shared/scripts must be importable).
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from experiment import exp_b_collectives as ebc  # noqa: E402


SLEEP_S = 0.01  # 10 ms per collective call


class MockComm:
    """A communicator whose every collective sleeps a fixed time and echoes back."""

    def __init__(self, sleep_s=SLEEP_S):
        self.sleep_s = sleep_s
        self.calls = []

    def broadcast(self, table, root, ctx):
        self.calls.append("broadcast")
        time.sleep(self.sleep_s)
        return table

    def scatter(self, tables, root, ctx):
        self.calls.append("scatter")
        time.sleep(self.sleep_s)
        return tables[0] if tables else None

    def gather(self, table, root):
        self.calls.append("gather")
        time.sleep(self.sleep_s)
        return [table]

    def allgather(self, table):
        self.calls.append("allgather")
        time.sleep(self.sleep_s)
        return [table]

    def reduce(self, value, op, root):
        self.calls.append("reduce")
        time.sleep(self.sleep_s)
        return value

    def allreduce(self, value, op):
        self.calls.append("allreduce")
        time.sleep(self.sleep_s)
        return value


class MockCtx:
    def __init__(self):
        self.barriers = 0

    def barrier(self):
        self.barriers += 1


def _dummy_builder(name, msg_size, ctx, world_size, rank, root):
    """Payload builder that never touches pycylon: dummy payloads, real byte counts."""
    if name == "barrier":
        return (None, 0)
    if name in ("scatter", "scatterv"):
        return ([object() for _ in range(world_size)] if rank == root else [], msg_size)
    return (object(), msg_size)


def test_measure_collective_broadcast_timing_and_throughput():
    comm, ctx = MockComm(), MockCtx()
    m = ebc.measure_collective(
        comm, ctx, "broadcast", msg_size=4096, warmup=1, reps=5,
        payload_builder=_dummy_builder, world_size=4, rank=0,
    )
    # ~10 ms per call, generous window for scheduler jitter.
    assert 6.0 <= m["latency_p50_ms"] <= 40.0, m
    assert m["latency_p99_ms"] >= m["latency_p50_ms"]
    assert m["throughput_MBps"] > 0.0
    assert m["unsupported"] is False
    assert m["payload_bytes"] == 4096
    # 1 warmup + 5 timed broadcast calls.
    assert comm.calls.count("broadcast") == 6


def test_measure_collective_barrier_has_no_throughput():
    comm, ctx = MockComm(), MockCtx()
    m = ebc.measure_collective(
        comm, ctx, "barrier", msg_size=0, warmup=1, reps=3,
        payload_builder=_dummy_builder,
    )
    assert m["throughput_MBps"] == 0.0
    assert m["barrier_latency_ms"] == m["latency_p50_ms"]
    # barrier does not add a pre-sync barrier, so only the timed invokes call it.
    assert ctx.barriers == 1 + 3  # 1 warmup + 3 timed invokes


def test_measure_collective_reduce_uses_reduce_op():
    comm, ctx = MockComm(), MockCtx()
    sentinel = object()
    m = ebc.measure_collective(
        comm, ctx, "reduce", msg_size=512, warmup=0, reps=3,
        reduce_op=sentinel, payload_builder=_dummy_builder, world_size=2, rank=0,
    )
    assert m["throughput_MBps"] > 0.0
    assert comm.calls.count("reduce") == 3


def test_unsupported_collective_records_na_not_zero_timing():
    comm, ctx = MockComm(), MockCtx()

    def _none_builder(*a, **k):
        return None  # simulate an unsupported (channel, collective) stub

    m = ebc.measure_collective(
        comm, ctx, "gather", msg_size=512, warmup=1, reps=3,
        payload_builder=_none_builder,
    )
    assert m["unsupported"] is True
    assert m["latency_p50_ms"] == 0.0
    assert m["throughput_MBps"] == 0.0
    # No collective was actually invoked.
    assert comm.calls == []


def test_run_sweeps_collectives_and_sizes_with_metadata():
    comm, ctx = MockComm(), MockCtx()
    config = {
        "comm": comm,
        "ctx": ctx,
        "collectives": ["broadcast", "barrier"],
        "msg_sizes": [8, 4096],
        "warmup": 0,
        "reps": 2,
        "channel": "mock",
        "rank": 0,
        "world_size": 4,
        "run_id": 2,
        "payload_builder": _dummy_builder,
    }
    rows = ebc.run(config)
    # broadcast over 2 sizes + barrier once (barrier ignores msg_sizes).
    assert len(rows) == 3
    for r in rows:
        assert r["channel"] == "mock"
        assert r["world_size"] == 4 and r["N"] == 4
        assert r["run_id"] == 2
        assert r["rank"] == 0
    bcast_rows = [r for r in rows if r["collective"] == "broadcast"]
    assert sorted(r["msg_size"] for r in bcast_rows) == [8, 4096]
    barrier_rows = [r for r in rows if r["collective"] == "barrier"]
    assert len(barrier_rows) == 1 and barrier_rows[0]["msg_size"] == 0