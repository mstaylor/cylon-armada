"""Arm B's broadcast and gather helpers over real pycylon FMI collectives.

test_arm_b.py fakes FMIBridge; this runs world_size real OS processes through the real
FMIBridge on the direct-redis channel against the Parallels-host redis, so the
empty-schema broadcast on non-root ranks and the root-only gather list are exercised
by the same collectives the Lambda's direct channel uses. Skipped when redis or the
pycylon FMI native stack is unavailable.
Run: pytest tests/cosmic_ai_lambda/test_arm_b_real_fmi.py -v
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
_LAMBDA_SRC = os.path.join(_REPO, "target", "aws", "scripts", "terraform-cosmic-ai", "lambda_src")


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
        import torch  # noqa: F401
        return True
    except Exception:
        return False


WORKER_SCRIPT = textwrap.dedent("""
    import json
    import os
    import sys
    import types

    sys.path.insert(0, {scripts_dir!r})
    sys.path.insert(0, {lambda_src!r})
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    from communicator.fmi_bridge import FMIBridge
    import inference_FMI

    world_size = int(os.environ["WORLD_SIZE"])
    bridge = FMIBridge(
        world_size=world_size, rank=int(os.environ["RANK"]), channel_type="direct-redis",
        listen_port=int(os.environ["FMI_LISTEN_PORT"]),
        redis_host=os.environ["REDIS_HOST"], redis_port=int(os.environ["REDIS_PORT"]),
        comm_name=os.environ["COMM_NAME"], maxtimeout=20000, advertise_host="127.0.0.1",
        required_peers=inference_FMI.required_peer_map(world_size),
    )
    rank = bridge.rank
    data_map = json.loads(os.environ["DATA_MAP"])

    class PayloadS3:
        def get_object(self, Bucket, Key):
            body = json.dumps({{"data_map": data_map}}).encode()
            return {{"Body": types.SimpleNamespace(read=lambda: body)}}

    inference_FMI.s3_client = PayloadS3()
    args = types.SimpleNamespace(payload_bucket="bkt", payload_key="payload.json", data_prefix="10MB")

    inference_FMI.timed_barrier(bridge)
    partition_map = inference_FMI.broadcast_partition_map(bridge, rank, world_size, args)
    record = {{field: float(rank) for field in inference_FMI.GATHERED_FIELDS}}
    record["code_fetch_s"] = None
    gathered = inference_FMI.gather_table(
        bridge, inference_FMI.row_table(rank, inference_FMI.GATHERED_FIELDS, record), 0,
    )

    result = {{
        "rank": rank,
        "paths": inference_FMI.decode_rank_paths(partition_map, rank, "10MB"),
        "gathered": len(gathered),
    }}
    if rank == 0:
        rows = inference_FMI.rows_by_rank(gathered, inference_FMI.GATHERED_FIELDS)
        result["rows"] = {{str(r): rows[r] for r in rows}}
    bridge.finalize()

    with open(os.environ["RESULT_PATH"], "w") as f:
        json.dump(result, f)
""")


def _run_world(world_size, data_map, tmp_path):
    script_path = os.path.join(str(tmp_path), "worker.py")
    with open(script_path, "w") as f:
        f.write(WORKER_SCRIPT.format(scripts_dir=_SCRIPTS, lambda_src=_LAMBDA_SRC))

    comm_name = f"armb_real_fmi_{uuid.uuid4().hex[:8]}"
    port_base = 30000 + (os.getpid() % 20000)
    procs = []
    for slot in range(world_size):
        env = dict(os.environ)
        env.update({
            "RANK": str(slot),
            "WORLD_SIZE": str(world_size),
            "REDIS_HOST": REDIS_HOST,
            "REDIS_PORT": str(REDIS_PORT),
            "COMM_NAME": comm_name,
            "FMI_LISTEN_PORT": str(port_base + slot),
            "DATA_MAP": json.dumps(data_map),
            "RESULT_PATH": os.path.join(str(tmp_path), f"result_{slot}.json"),
        })
        log_path = os.path.join(str(tmp_path), f"log_{slot}.txt")
        log_file = open(log_path, "w")
        proc = subprocess.Popen([sys.executable, script_path], env=env,
                                stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((slot, proc, log_file, log_path))

    try:
        results = {}
        for slot, proc, log_file, log_path in procs:
            rc = proc.wait(timeout=90)
            log_file.close()
            result_path = os.path.join(str(tmp_path), f"result_{slot}.json")
            if rc != 0 or not os.path.exists(result_path):
                with open(log_path) as lf:
                    pytest.fail(f"launch slot {slot} failed (exit {rc}):\n{lf.read()}")
            with open(result_path) as f:
                data = json.load(f)
            results[data["rank"]] = data
        return results
    finally:
        for _, proc, log_file, _ in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if not log_file.closed:
                log_file.close()


@pytest.mark.skipif(not _redis_available(), reason="host redis not reachable")
@pytest.mark.skipif(not _pycylon_fmi_available(), reason="pycylon FMI native stack not importable")
@pytest.mark.parametrize("world_size", [2, 3, 5, 11])
def test_partition_map_broadcast_and_stage_gather_over_real_fmi(world_size, tmp_path):
    data_map = {str(r): f"10MB/{r + 1}.pt" for r in range(world_size)}
    data_map["0"] = ["10MB/1.pt", "10MB/10.pt"]

    results = _run_world(world_size, data_map, tmp_path)

    assert sorted(results) == list(range(world_size))
    for rank in range(world_size):
        assert results[rank]["paths"] == data_map[str(rank)]
        assert results[rank]["gathered"] == (world_size if rank == 0 else 0)
    rows = results[0]["rows"]
    assert sorted(rows, key=int) == [str(r) for r in range(world_size)]
    for rank in range(world_size):
        assert rows[str(rank)]["init_s"] == float(rank)
        assert rows[str(rank)]["code_fetch_s"] is None
