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

"""OverlappedEpochs over a real FMIBridge, one OS process per rank.

Every rank stores every rank's rows exactly once, whatever the epoch sizes,
and the FMI AllGather on the calling thread must release the GIL so the
worker's model calls keep running while it waits for a slow peer.
"""

import json
import os
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


def _pycylon_fmi_available():
    try:
        import pycylon.net.fmi_config  # noqa: F401
        return True
    except Exception:
        return False


WORKER_SCRIPT = textwrap.dedent("""
    import json
    import os
    import sys
    import threading
    import time

    sys.path.insert(0, {scripts_dir!r})

    import pyarrow as pa

    from armada.exchange_schedule import OverlappedEpochs
    from armada.executor import ArmadaExecutor, required_peer_map
    from armada.operator import ArmadaOperator
    from communicator.fmi_bridge import FMIBridge
    from cylon_armada.dag_compiler import CollectivePattern

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    epochs = json.loads(os.environ["EPOCHS"])[rank]
    reason_s = float(os.environ["REASON_S"])
    lead_s = json.loads(os.environ["LEAD_S"])[rank]

    schema = pa.schema([pa.field("value", pa.int64())])
    stored = []
    reason_done_at = []
    caller = threading.get_ident()
    cylon_threads = set()

    def retrieve(table):
        cylon_threads.add(threading.get_ident())
        time.sleep(lead_s)
        return table

    def reason(table):
        time.sleep(reason_s)
        reason_done_at.append(time.perf_counter())
        return table

    def upsert(table):
        cylon_threads.add(threading.get_ident())
        stored.extend(table.column("value").to_pylist())
        return table

    seq = (ArmadaOperator("Retrieve", CollectivePattern.PointToPoint, schema, schema, fn=retrieve)
           | ArmadaOperator("Reason", CollectivePattern.PointToPoint, schema, schema, fn=reason)
           | ArmadaOperator("MemoryUpsert", CollectivePattern.AllGather, schema, schema,
                            fn=upsert))

    bridge = FMIBridge(
        world_size=world_size, rank=rank, channel_type="direct-redis",
        listen_port=int(os.environ["FMI_LISTEN_PORT"]),
        redis_host=os.environ["REDIS_HOST"], redis_port=int(os.environ["REDIS_PORT"]),
        comm_name=os.environ["COMM_NAME"], maxtimeout=30000,
        advertise_host="127.0.0.1",
        required_peers=required_peer_map(seq, world_size),
    )
    executor = ArmadaExecutor(bridge)
    exchange_returned_at = []

    def exchange(contribution):
        cylon_threads.add(threading.get_ident())
        peers = executor.exchange_peers(contribution, bridge._ctx)
        exchange_returned_at.append(time.perf_counter())
        return peers

    bridge.barrier()
    schedule = OverlappedEpochs(exchange, ("Reason",))
    for values in epochs:
        schedule.run(seq, pa.table({{"value": values}}, schema=schema))
    schedule.drain()
    bridge.barrier()
    bridge.finalize()

    with open(os.environ["RESULT_PATH"], "w") as f:
        json.dump({{"rank": bridge.rank, "requested_rank": rank, "stored": sorted(stored),
                    "reason_done_at": reason_done_at,
                    "exchange_returned_at": exchange_returned_at,
                    "cylon_on_caller_only": cylon_threads == {{caller}}}}, f)
""")


def _run_world(tmp_path, epochs, reason_s=0.0, lead_s=None):
    world_size = len(epochs)
    lead_s = lead_s or [0.0] * world_size
    comm_name = f"armada_overlap_{uuid.uuid4().hex[:8]}"
    port_base = 21000 + (os.getpid() % 20000)
    script_path = os.path.join(str(tmp_path), "worker.py")
    with open(script_path, "w") as f:
        f.write(WORKER_SCRIPT.format(scripts_dir=_SCRIPTS))

    procs = []
    for rank in range(world_size):
        env = dict(os.environ)
        env.update({
            "RANK": str(rank), "WORLD_SIZE": str(world_size),
            "REDIS_HOST": REDIS_HOST, "REDIS_PORT": str(REDIS_PORT),
            "COMM_NAME": comm_name, "FMI_LISTEN_PORT": str(port_base + rank),
            "EPOCHS": json.dumps(epochs), "REASON_S": str(reason_s),
            "LEAD_S": json.dumps(lead_s),
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
            results[data["rank"]] = data
        return results
    finally:
        for _, proc, log_file, _ in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if not log_file.closed:
                log_file.close()


needs_fmi = [
    pytest.mark.skipif(not _redis_available(), reason="host redis not reachable"),
    pytest.mark.skipif(not _pycylon_fmi_available(), reason="pycylon FMI native stack not importable"),
]


@needs_fmi[0]
@needs_fmi[1]
@pytest.mark.parametrize("epochs", [
    [[[1], [2]], [[3], [4]]],
    [[[1, 2], [3]], [[4]], []],
    [[[1]], [[2]], [[3]], [[4]]],
    [[[1], [2]], [], [[3, 4]], [[5], [6], [7]], [[8]], [[9]], [[10]], []],
], ids=["n2", "n3-uneven", "n4", "n8-uneven"])
def test_every_rank_stores_every_row_once_with_cylon_on_one_thread(tmp_path, epochs):
    n_epochs = max(len(rank_epochs) for rank_epochs in epochs)
    padded = [rank_epochs + [[]] * (n_epochs - len(rank_epochs)) for rank_epochs in epochs]
    every_row = sorted(v for rank_epochs in padded for epoch in rank_epochs for v in epoch)

    results = _run_world(tmp_path, padded)

    for rank, result in results.items():
        assert result["stored"] == every_row, rank
        assert result["cylon_on_caller_only"], rank


@needs_fmi[0]
@needs_fmi[1]
def test_model_calls_finish_while_the_allgather_waits_for_a_slow_peer(tmp_path):
    """Rank 1 reaches each exchange 1.5 s after rank 0, whose model calls take
    0.3 s. If the AllGather held the GIL, rank 0's worker could not record its
    finish until the AllGather returned."""
    results = _run_world(tmp_path, [[[1], [2], [3]], [[4], [5], [6]]], reason_s=0.3,
                         lead_s=[0.0, 1.5])

    fast = next(r for r in results.values() if r["requested_rank"] == 0)
    reason_done, exchange_returned = fast["reason_done_at"], fast["exchange_returned_at"]
    for epoch in (1, 2):
        assert reason_done[epoch] < exchange_returned[epoch - 1] - 0.5, (
            f"epoch {epoch}: model calls finished {reason_done[epoch]:.3f}, the exchange of "
            f"epoch {epoch - 1} returned {exchange_returned[epoch - 1]:.3f}")