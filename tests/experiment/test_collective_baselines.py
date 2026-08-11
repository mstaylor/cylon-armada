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

"""Correctness + latency-row tests for the Experiment B HTTP and redis baselines.

The HTTP test runs the coordinator + N worker threads in-process. The redis test
uses the Parallels-host redis (10.211.55.2:6379) and is skipped when unreachable.
Both check that allreduce/broadcast return correct values and that a full sweep
produces measure_collective-shaped rows. Run:
  pytest tests/experiment/test_collective_baselines.py -x
"""

import os
import sys
import threading

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from baselines import http_collective_baseline as http_base  # noqa: E402

WS = 4
REDIS_HOST = os.environ.get("REDIS_HOST", "10.211.55.2")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))


def _run_workers(worker_fn, world_size):
    """Run `world_size` worker threads and collect each rank's return value."""
    results = {}
    errors = {}

    def _wrap(rank):
        try:
            results[rank] = worker_fn(rank)
        except Exception as e:  # surface worker exceptions to the test
            errors[rank] = e

    threads = [threading.Thread(target=_wrap, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, f"worker errors: {errors}"
    return results


# --------------------------------------------------------------------------- HTTP

def test_http_allreduce_broadcast_correct():
    server = http_base.make_server(WS)
    host, port = server.server_address
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    base = f"http://{host}:{port}"

    def worker(rank):
        client = http_base.HTTPCollective(base, rank, WS)
        ar = client.allreduce([float(rank + 1)])
        bc = client.broadcast([42.0, 42.0, 42.0] if rank == 0 else [0.0, 0.0, 0.0], root=0)
        return ar, bc

    try:
        results = _run_workers(worker, WS)
    finally:
        server.shutdown()

    expected_sum = float(sum(r + 1 for r in range(WS)))  # 1+2+3+4 = 10
    for rank in range(WS):
        ar, bc = results[rank]
        assert ar == [expected_sum], f"rank {rank} allreduce {ar}"
        assert bc == [42.0, 42.0, 42.0], f"rank {rank} broadcast {bc}"


def test_http_run_sweep_produces_rows():
    server = http_base.make_server(WS)
    host, port = server.server_address
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://{host}:{port}"

    collectives = ["allreduce", "broadcast", "barrier"]
    msg_sizes = [64, 512]

    def worker(rank):
        client = http_base.HTTPCollective(base, rank, WS)
        return http_base.run({
            "client": client, "collectives": collectives, "msg_sizes": msg_sizes,
            "warmup": 1, "reps": 3, "rank": rank, "world_size": WS,
        })

    try:
        results = _run_workers(worker, WS)
    finally:
        server.shutdown()

    rows = results[0]
    # allreduce (scalar, 1 size) + broadcast (2 sizes) + barrier (1) = 4 rows
    assert len(rows) == 4
    for row in rows:
        assert row["baseline"] == "http"
        assert row["latency_p50_ms"] > 0.0
        assert row["reps"] == 3
    bcast = [r for r in rows if r["collective"] == "broadcast"]
    assert sorted(r["msg_size"] for r in bcast) == [64, 512]
    assert all(r["throughput_MBps"] > 0.0 for r in bcast)


# -------------------------------------------------------------------------- redis

def _redis_available():
    try:
        import redis
        redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=2).ping()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _redis_available(), reason="host redis not reachable")
def test_redis_allreduce_broadcast_correct():
    import redis
    from baselines import redis_coord_baseline as redis_base

    ns = f"expb_test_{os.getpid()}"

    def worker(rank):
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        client = redis_base.RedisCollective(r, rank, WS, namespace=ns)
        ar = client.allreduce([float(rank + 1)])
        bc = client.broadcast([7.0, 7.0] if rank == 0 else [0.0, 0.0], root=0)
        return ar, bc

    results = _run_workers(worker, WS)

    expected_sum = float(sum(r + 1 for r in range(WS)))
    for rank in range(WS):
        ar, bc = results[rank]
        assert ar == [expected_sum], f"rank {rank} allreduce {ar}"
        assert bc == [7.0, 7.0], f"rank {rank} broadcast {bc}"


@pytest.mark.skipif(not _redis_available(), reason="host redis not reachable")
def test_redis_run_sweep_produces_rows():
    import redis
    from baselines import redis_coord_baseline as redis_base

    ns = f"expb_test_sweep_{os.getpid()}"
    collectives = ["allreduce", "broadcast", "barrier"]
    msg_sizes = [64, 512]

    def worker(rank):
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        client = redis_base.RedisCollective(r, rank, WS, namespace=ns)
        return redis_base.run({
            "client": client, "collectives": collectives, "msg_sizes": msg_sizes,
            "warmup": 1, "reps": 3, "rank": rank, "world_size": WS,
        })

    results = _run_workers(worker, WS)
    rows = results[0]
    assert len(rows) == 4
    for row in rows:
        assert row["baseline"] == "redis"
        assert row["latency_p50_ms"] > 0.0