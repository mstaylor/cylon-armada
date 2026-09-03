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

"""Real TCPunch rendezvous test at odd world sizes (not Lambda — local processes,
real rendezvous server over the network).

Launches world_size real OS processes, each independently connecting through
the live TCPunch rendezvous server (channel_type="direct") to run the
binomial-tree allreduce — the same connection-establishment and collective
code path Lambda would use, just without Lambda's own ephemeral/NAT-specific
runtime conditions. Verifies the pairing math (ceil(log2(N)) rounds, at most
one peer per round, min(2^i, N-src) clamp at non-power-of-2 boundaries — see
PeerToPeer.cpp) is genuinely N-agnostic, not just in theory.

Requires RENDEZVOUS_ADDR reachable (default: the live cylon-rendezvous
server) and the pycylon FMI native stack; skipped otherwise.

Run: pytest tests/armada/test_rendezvous_direct_odd_world_size.py -v -s
"""

import json
import os
import subprocess
import sys
import textwrap
import uuid

import pytest

RENDEZVOUS_HOST = os.environ.get("RENDEZVOUS_ADDR", "cylon-rendezvous.aws-cylondata.com:10000").split(":")[0]
RENDEZVOUS_PORT = int(os.environ.get("RENDEZVOUS_ADDR", "cylon-rendezvous.aws-cylondata.com:10000").split(":")[1])
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")


def _rendezvous_reachable():
    import socket
    try:
        with socket.create_connection((RENDEZVOUS_HOST, RENDEZVOUS_PORT), timeout=5):
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

    from communicator.fmi_bridge import FMIBridge

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    bridge = FMIBridge(
        world_size=world_size, rank=rank, channel_type="direct",
        rendezvous_host=os.environ["RENDEZVOUS_HOST"],
        rendezvous_port=int(os.environ["RENDEZVOUS_PORT"]),
        comm_name=os.environ["COMM_NAME"], maxtimeout=30000,
    )

    if not bridge.available:
        result = {{"rank": rank, "available": False}}
    else:
        r = bridge.rank
        bridge.barrier()
        total = bridge.reduce_float(1.0, op="sum")
        result = {{"rank": r, "available": True, "allreduce_result": total}}
        bridge.barrier()

    bridge.finalize()

    with open(os.environ["RESULT_PATH"], "w") as f:
        json.dump(result, f)
""")


def _run_rank(rank, world_size, comm_name, result_dir):
    script_path = os.path.join(result_dir, f"worker_{rank}.py")
    with open(script_path, "w") as f:
        f.write(WORKER_SCRIPT.format(scripts_dir=_SCRIPTS))

    env = dict(os.environ)
    env.update({
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "RENDEZVOUS_HOST": RENDEZVOUS_HOST,
        "RENDEZVOUS_PORT": str(RENDEZVOUS_PORT),
        "COMM_NAME": comm_name,
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
    comm_name = f"armada_test_rendezvous_{uuid.uuid4().hex[:8]}"
    result_dir = str(tmp_path)
    procs = []
    for rank in range(world_size):
        proc, log_file, log_path = _run_rank(rank, world_size, comm_name, result_dir)
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


@pytest.mark.skipif(
    os.environ.get("ARMADA_TCPUNCH_MULTIHOST") != "1",
    reason="TCPunch hole punching needs ranks on distinct hosts/NATs — all ranks on one "
           "host share a public IP, and do_hole_punch() has no same-IP case, so it would "
           "require NAT hairpinning. Set ARMADA_TCPUNCH_MULTIHOST=1 only when ranks really "
           "are on separate hosts (e.g. real Lambda).",
)
@pytest.mark.skipif(not _rendezvous_reachable(), reason="rendezvous server not reachable")
@pytest.mark.skipif(not _pycylon_fmi_available(), reason="pycylon FMI native stack not importable")
@pytest.mark.parametrize("world_size", [3, 5, 7])
def test_direct_channel_allreduce_correct_at_odd_world_size(world_size, tmp_path):
    results = _run_world(world_size, tmp_path)
    for rank in range(world_size):
        assert results[rank]["available"], f"rank {rank}: bridge unavailable"
        assert results[rank]["allreduce_result"] == float(world_size), (
            f"rank {rank}: allreduce returned {results[rank]['allreduce_result']}, "
            f"expected {world_size}"
        )