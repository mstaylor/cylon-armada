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

"""ArmadaAsyncExecutor over a real FMIBridge, one OS process per rank.

Every rank stores every rank's rows exactly once, and fast ranks see each
other's contributions without waiting for a slow rank, as ray-native's do.
"""

import json
import os
import socket
import subprocess
import sys
import textwrap
import uuid

import pytest

REDIS_HOST = os.environ.get("REDIS_ADDR", "10.211.55.2:6379").split(":")[0]
REDIS_PORT = int(os.environ.get("REDIS_ADDR", "10.211.55.2:6379").split(":")[1])
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")


def _redis_available():
    try:
        import redis
        redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=2).ping()
        return True
    except Exception:
        return False


def _async_exchange_available():
    try:
        import pycylon.net.async_table_exchange  # noqa: F401
        return True
    except Exception:
        return False


WORKER_SCRIPT = textwrap.dedent("""
    import json
    import os
    import sys
    import time

    sys.path.insert(0, {scripts_dir!r})

    import pyarrow as pa

    from armada.async_executor import ArmadaAsyncExecutor
    from armada.operator import ArmadaOperator
    from communicator.fmi_bridge import FMIBridge
    from cylon_armada.dag_compiler import CollectivePattern

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    epochs = json.loads(os.environ["EPOCHS"])[rank]
    reason_s = json.loads(os.environ["REASON_S"])[rank]

    schema = pa.schema([pa.field("value", pa.int64())])
    upserts = []
    t0 = [0.0]

    def reason(table):
        time.sleep(reason_s)
        return table

    def upsert(table):
        upserts.append((sorted(table.column("value").to_pylist()), time.perf_counter() - t0[0]))
        return table

    seq = (ArmadaOperator("Retrieve", CollectivePattern.PointToPoint, schema, schema,
                          fn=lambda table: table)
           | ArmadaOperator("Reason", CollectivePattern.PointToPoint, schema, schema, fn=reason)
           | ArmadaOperator("MemoryUpsert", CollectivePattern.AllGather, schema, schema,
                            fn=upsert))

    bridge = FMIBridge(
        world_size=world_size, rank=rank, channel_type="direct-redis",
        listen_port=int(os.environ["FMI_LISTEN_PORT"]),
        redis_host=os.environ["REDIS_HOST"], redis_port=int(os.environ["REDIS_PORT"]),
        comm_name=os.environ["COMM_NAME"], maxtimeout=30000,
        advertise_host="127.0.0.1", required_peers=None,
    )
    bridge.barrier()
    t0[0] = time.perf_counter()
    executor = ArmadaAsyncExecutor(bridge.async_exchange, ("Reason",), poll_s=0.005,
                                   settle_s=1.0)
    for values in epochs:
        executor.run(seq, pa.table({{"value": values}}, schema=schema))
    executor.finish(seq, timeout_s=60.0)
    bridge.barrier()
    bridge.finalize()

    with open(os.environ["RESULT_PATH"], "w") as f:
        json.dump({{"rank": bridge.rank, "requested_rank": rank, "upserts": upserts}}, f)
""")


def _free_ports(count):
    sockets = [socket.socket() for _ in range(count)]
    for s in sockets:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in sockets]
    for s in sockets:
        s.close()
    return ports


def _run_world(tmp_path, epochs, reason_s):
    world_size = len(epochs)
    comm_name = f"armada_async_{uuid.uuid4().hex[:8]}"
    ports = _free_ports(world_size)
    script_path = os.path.join(str(tmp_path), "worker.py")
    with open(script_path, "w") as f:
        f.write(WORKER_SCRIPT.format(scripts_dir=_SCRIPTS))

    procs = []
    for rank in range(world_size):
        env = dict(os.environ)
        env.update({
            "RANK": str(rank), "WORLD_SIZE": str(world_size),
            "REDIS_HOST": REDIS_HOST, "REDIS_PORT": str(REDIS_PORT),
            "COMM_NAME": comm_name, "FMI_LISTEN_PORT": str(ports[rank]),
            "EPOCHS": json.dumps(epochs), "REASON_S": json.dumps(reason_s),
            "RESULT_PATH": os.path.join(str(tmp_path), f"result_{rank}.json"),
        })
        log_path = os.path.join(str(tmp_path), f"log_{rank}.txt")
        log_file = open(log_path, "w")
        procs.append((rank, subprocess.Popen([sys.executable, script_path], env=env,
                                             stdout=log_file, stderr=subprocess.STDOUT),
                      log_file, log_path))
    try:
        results = {}
        for rank, proc, log_file, log_path in procs:
            rc = proc.wait(timeout=120)
            log_file.close()
            result_path = os.path.join(str(tmp_path), f"result_{rank}.json")
            if rc != 0 or not os.path.exists(result_path):
                pytest.fail(f"rank {rank} failed (exit {rc}):\n{open(log_path).read()}")
            data = json.load(open(result_path))
            results[data["requested_rank"]] = data
        return results
    finally:
        for _, proc, log_file, _ in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if not log_file.closed:
                log_file.close()


def _rows(rank, n_epochs, per_epoch=2):
    return [[rank * 100 + epoch * 10 + i for i in range(per_epoch)] for epoch in range(n_epochs)]


needs_fmi = [
    pytest.mark.skipif(not _redis_available(), reason="host redis not reachable"),
    pytest.mark.skipif(not _async_exchange_available(),
                       reason="pycylon AsyncTableExchange not importable"),
]


@needs_fmi[0]
@needs_fmi[1]
@pytest.mark.parametrize("world_size", [2, 3])
def test_every_rank_stores_every_row_exactly_once(tmp_path, world_size):
    epochs = [_rows(rank, 3) for rank in range(world_size)]
    every_row = sorted(v for rank_epochs in epochs for epoch in rank_epochs for v in epoch)

    results = _run_world(tmp_path, epochs, [0.05] * world_size)

    for rank, result in results.items():
        stored = sorted(v for values, _ in result["upserts"] for v in values)
        assert stored == every_row, rank


@needs_fmi[0]
@needs_fmi[1]
def test_fast_ranks_see_each_others_epoch_without_waiting_for_the_slow_rank(tmp_path):
    epochs = [_rows(rank, 3) for rank in range(3)]

    results = _run_world(tmp_path, epochs, [0.1, 0.1, 1.5])

    for fast, other in ((0, 1), (1, 0)):
        first_two = results[fast]["upserts"][:2]
        seen = {v for values, _ in first_two for v in values}
        assert set(epochs[other][0]) <= seen, (fast, first_two)
        assert first_two[-1][1] < 1.5, (fast, first_two)