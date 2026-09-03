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

"""Multi-rank correctness test for ArmadaExecutor.run() (SP1 Task 4, Step 4).

Runs a real Scatter -> Reduce ArmadaSequence through a real FMIBridge over
genuine OS processes (direct-redis channel), proving data provably moves
through the B collectives end-to-end via the executor — not just that the
executor calls the right bridge methods (that's test_executor_distribution.py's
spy-based proof).

world_size == 1 degenerates to a local run with no peers; its result must
match a single-process invoke of the same ArmadaSequence.

Run: pytest tests/armada/test_executor_distribution_multirank.py -x
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
    from pycylon import CylonContext, Table
    from pycylon.data.column import Column

    from armada.executor import ArmadaExecutor, required_peer_map
    from armada.operator import ArmadaOperator
    from communicator.fmi_bridge import FMIBridge
    from cylon_armada.dag_compiler import CollectivePattern

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    s = pa.schema([pa.field("v", pa.float32())])

    def preprocess_fn(tables):
        return tables

    def retrieve_fn(table):
        return Column(table.to_pandas()["v"].to_numpy().astype(np.float32))

    # Built before the bridge: the channel establishes its connections while the
    # communicator is constructed, so the peer map has to be derived first.
    seq = (
        ArmadaOperator("Preprocess", CollectivePattern.Scatter, s, s, fn=preprocess_fn)
        | ArmadaOperator("Retrieve", CollectivePattern.Reduce, s, s, fn=retrieve_fn)
    )

    peers = None
    if os.environ.get("USE_REQUIRED_PEERS") == "1":
        peers = required_peer_map(seq, world_size)

    bridge = FMIBridge(
        world_size=world_size, rank=rank, channel_type="direct-redis",
        listen_port=int(os.environ["FMI_LISTEN_PORT"]),
        redis_host=os.environ["REDIS_HOST"], redis_port=int(os.environ["REDIS_PORT"]),
        comm_name=os.environ["COMM_NAME"], maxtimeout=20000,
        advertise_host="127.0.0.1",
        required_peers=peers,
    )
    r = bridge.rank
    N = bridge.world_size
    root = 0

    local_ctx = bridge._ctx if bridge._ctx is not None else CylonContext()

    def tbl(v, n=3):
        return Table.from_arrow(local_ctx, pa.table({{"v": pa.array(np.full(n, v, dtype=np.float32))}}))

    executor = ArmadaExecutor(bridge)

    # world_size==1 has no scatter at all (single_rank just calls fn straight
    # through, matching plain Runnable.invoke() chaining) — its input is the
    # one shard directly, not a 1-element list of shards.
    if N == 1:
        input_tables = tbl(1.0)
    else:
        input_tables = [tbl(float(i + 1)) for i in range(N)] if r == root else []
    result = executor.run(seq, input_tables=input_tables, ctx=local_ctx, root=root)

    out = {{"rank": r}}
    if r == root:
        out["reduce_result"] = result.data.to_numpy().tolist()

    if bridge._ctx is not None:
        bridge._ctx.barrier()
    bridge.finalize()

    with open(os.environ["RESULT_PATH"], "w") as f:
        json.dump(out, f)
""")


def _run_rank(rank, world_size, comm_name, result_dir, port_base, restrict_peers=False):
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
        "USE_REQUIRED_PEERS": "1" if restrict_peers else "0",
    })
    log_path = os.path.join(result_dir, f"log_{rank}.txt")
    log_file = open(log_path, "w")
    proc = subprocess.Popen([sys.executable, script_path], env=env,
                            stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file, log_path


def _run_world(world_size, tmp_path, restrict_peers=False):
    comm_name = f"armada_test_executor_{uuid.uuid4().hex[:8]}"
    result_dir = str(tmp_path)
    port_base = 20000 + (os.getpid() % 20000)
    procs = []
    for rank in range(world_size):
        proc, log_file, log_path = _run_rank(rank, world_size, comm_name, result_dir,
                                             port_base, restrict_peers)
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
        assert sorted(results.keys()) == list(range(world_size))
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
@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 5, 8])
def test_executor_scatter_reduce_correct_over_real_collectives(world_size, tmp_path):
    results = _run_world(world_size, tmp_path)
    expected_sum = [float(sum(range(1, world_size + 1)))] * 3
    assert results[0]["reduce_result"] == expected_sum


@pytest.mark.skipif(not _redis_available(), reason="host redis not reachable")
@pytest.mark.skipif(not _pycylon_fmi_available(), reason="pycylon FMI native stack not importable")
@pytest.mark.parametrize("world_size", [2, 3, 4, 5, 8])
def test_same_result_when_only_plan_required_peers_are_connected(world_size, tmp_path):
    """Restricting the connection set to what the plan needs must not change the answer.

    FMI's direct channel otherwise connects every rank to every other rank
    (N(N-1)/2 pairings) because it can't know the schedule. armada does know it,
    so it passes the derived peer set down — a wrong or asymmetric set shows up
    here as a hang or a wrong reduce, not as a silent degradation.
    """
    results = _run_world(world_size, tmp_path, restrict_peers=True)
    expected_sum = [float(sum(range(1, world_size + 1)))] * 3
    assert results[0]["reduce_result"] == expected_sum