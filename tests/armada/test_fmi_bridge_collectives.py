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

"""Multi-rank correctness tests for FMIBridge's table collectives (SP1 Task 2).

Launches world_size real OS processes over the direct-redis channel against the
Parallels-host redis; skipped when unreachable or the pycylon FMI native stack
is not importable. Run: pytest tests/armada/test_fmi_bridge_collectives.py -x
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

    sys.path.insert(0, {scripts_dir!r})

    import numpy as np
    import pyarrow as pa
    from pycylon import Table
    from pycylon.data.column import Column

    from communicator.fmi_bridge import FMIBridge

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    bridge = FMIBridge(
        world_size=world_size, rank=rank, channel_type="direct-redis",
        listen_port=int(os.environ["FMI_LISTEN_PORT"]),
        redis_host=os.environ["REDIS_HOST"], redis_port=int(os.environ["REDIS_PORT"]),
        comm_name=os.environ["COMM_NAME"], maxtimeout=20000,
        advertise_host="127.0.0.1",
    )
    ctx = bridge._ctx
    r = bridge.rank
    N = bridge.world_size

    def tbl(v, n=3):
        return Table.from_arrow(ctx, pa.table({{"v": pa.array(np.full(n, v, dtype=np.float32))}}))

    result = {{"rank": r}}
    root = 0

    shards = [tbl(float(i + 1)) for i in range(N)] if r == root else []
    out = bridge.scatter(shards, root)
    result["scatter"] = out.to_pandas()["v"].to_numpy().tolist()

    ctx.barrier()

    gathered = bridge.gather(tbl(float(r + 1)), root)
    if r == root:
        result["gather"] = sorted(float(t.to_pandas()["v"].to_numpy()[0]) for t in gathered)

    ctx.barrier()

    ag = bridge.allgather(tbl(float(r + 1)))
    result["allgather"] = sorted(float(t.to_pandas()["v"].to_numpy()[0]) for t in ag)

    ctx.barrier()

    bout = bridge.broadcast(tbl(777.0 if r == root else -1.0), root)
    result["broadcast"] = bout.to_pandas()["v"].to_numpy().tolist()

    ctx.barrier()

    col = Column(np.arange(1, 4, dtype=np.float32) * (r + 1))
    rres = bridge.reduce_table(col, "sum", root)
    if r == root:
        result["reduce_table"] = rres.data.to_numpy().tolist()

    ctx.barrier()
    bridge.finalize()

    with open(os.environ["RESULT_PATH"], "w") as f:
        json.dump(result, f)
""")


def _run_rank(rank, world_size, comm_name, result_dir, port_base):
    script_path = os.path.join(result_dir, f"worker_{rank}.py")
    with open(script_path, "w") as f:
        f.write(WORKER_SCRIPT.format(scripts_dir=_SCRIPTS))

    env = dict(os.environ)
    env.update({
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "REDIS_HOST": REDIS_HOST,
        "REDIS_PORT": str(REDIS_PORT),
        "COMM_NAME": comm_name,
        "FMI_LISTEN_PORT": str(port_base + rank),
        "ADVERTISE_HOST": "127.0.0.1",
        "RESULT_PATH": os.path.join(result_dir, f"result_{rank}.json"),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "LD_PRELOAD": os.environ.get("LD_PRELOAD", ""),
    })
    log_path = os.path.join(result_dir, f"log_{rank}.txt")
    log_file = open(log_path, "w")
    proc = subprocess.Popen([sys.executable, script_path], env=env,
                            stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file, log_path


def _run_world(world_size, tmp_path):
    comm_name = f"armada_test_fmi_bridge_{uuid.uuid4().hex[:8]}"
    result_dir = str(tmp_path)
    port_base = 20000 + (os.getpid() % 20000)
    procs = []
    for rank in range(world_size):
        proc, log_file, log_path = _run_rank(rank, world_size, comm_name, result_dir, port_base)
        procs.append((rank, proc, log_file, log_path))

    try:
        results = {}
        for rank, proc, log_file, log_path in procs:
            rc = proc.wait(timeout=60)
            log_file.close()
            result_path = os.path.join(result_dir, f"result_{rank}.json")
            if rc != 0 or not os.path.exists(result_path):
                with open(log_path) as lf:
                    log_contents = lf.read()
                pytest.fail(f"launch-slot {rank} failed (exit {rc}):\n{log_contents}")
            with open(result_path) as f:
                data = json.load(f)
            results[data["rank"]] = data
        assert sorted(results.keys()) == list(range(world_size)), (
            f"expected true FMI ranks 0..{world_size - 1}, got {sorted(results.keys())}"
        )
        return results
    finally:
        for rank, proc, log_file, log_path in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if not log_file.closed:
                log_file.close()


@pytest.mark.skipif(not _redis_available(), reason="host redis not reachable")
@pytest.mark.skipif(not _pycylon_fmi_available(), reason="pycylon FMI native stack not importable")
def test_table_collectives_correct_at_world_size_4(tmp_path):
    N = 4
    results = _run_world(N, tmp_path)

    for rank in range(N):
        assert results[rank]["scatter"] == [float(rank + 1)] * 3
        assert results[rank]["allgather"] == [float(i + 1) for i in range(N)]
        assert results[rank]["broadcast"] == [777.0] * 3

    assert results[0]["gather"] == [float(i + 1) for i in range(N)]
    expected_reduce = [float(sum(k * (r + 1) for r in range(N))) for k in (1, 2, 3)]
    assert results[0]["reduce_table"] == expected_reduce