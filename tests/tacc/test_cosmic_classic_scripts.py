"""Vista SLURM scripts for classic Cosmic AI: syntax, and the sweep's dry-run plan.

Run: pytest tests/tacc/test_cosmic_classic_scripts.py -v
"""

import json
import os
import re
import subprocess

import pytest

_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "target", "tacc", "scripts"))
SWEEP = os.path.join(_SCRIPTS, "sweep_cosmic_classic.sh")
SHELL_SCRIPTS = ["sweep_cosmic_classic.sh", "cosmic_classic.slurm", "pull_image.slurm", "stage.sh"]


def _dry_run(**env):
    base = {"PATH": os.environ["PATH"], "ACCOUNT": "TG-TEST", "COSMIC_ROOT": "/scratch/u/cosmic",
            "SIF_IMAGE": "/scratch/u/cosmic/cosmic-ai-lambda_arm64.sif"}
    result = subprocess.run(["bash", SWEEP, "--dry-run"], env={**base, **env},
                            capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line.startswith("sbatch")]


def _option(line, name):
    match = re.search(rf"--{name}=(\S+)", line)
    return match.group(1) if match else None


def _export(line, key):
    match = re.search(rf"[,=]{key}=([^,\s]+)", line)
    return match.group(1) if match else None


@pytest.mark.parametrize("script", SHELL_SCRIPTS)
def test_script_passes_bash_syntax_check(script):
    subprocess.run(["bash", "-n", os.path.join(_SCRIPTS, script)], check=True)


def test_sweep_dry_run_includes_one_and_two_and_runs_warmup_plus_measured_per_size():
    lines = _dry_run()

    sizes = [int(_export(line, "WORLD_SIZE")) for line in lines]
    assert sorted(set(sizes)) == [1, 2, 4, 8, 16, 41]
    assert all(sizes.count(n) == 4 for n in set(sizes))
    labels = [os.path.basename(_export(line, "RUN_DIR")) for line in lines if _export(line, "WORLD_SIZE") == "2"]
    assert labels == ["warmup1", "run1", "run2", "run3"]


def test_sweep_sizes_nodes_from_cores_per_worker():
    lines = _dry_run()

    nodes = {int(_export(line, "WORLD_SIZE")): int(_option(line, "nodes")) for line in lines}
    assert nodes[1] == nodes[16] == 1
    assert nodes[41] == 2
    first = lines[0]
    assert _option(first, "ntasks") == _export(first, "WORLD_SIZE")
    assert _option(first, "cpus-per-task") == "6"
    assert _option(first, "account") == "TG-TEST" and _option(first, "partition") == "gg"


def test_sweep_honours_overridden_sizes_and_cores():
    lines = _dry_run(WORLD_SIZES="1 2 48", CPUS_PER_WORKER="12", RUNS="1", WARMUP_RUNS="0")

    assert [(_export(line, "WORLD_SIZE"), _option(line, "nodes")) for line in lines] == [
        ("1", "1"), ("2", "1"), ("48", "4")]


def test_sweep_dry_run_chains_every_submission_after_the_previous():
    lines = _dry_run()

    assert _option(lines[0], "dependency") is None
    for index, line in enumerate(lines[1:]):
        assert _option(line, "dependency") == f"afterany:dryrun{index}"
    assert len({_export(line, "RUN_DIR") for line in lines}) == len(lines)


def test_sweep_chains_on_the_job_id_even_when_sbatch_prints_a_banner(tmp_path):
    calls = tmp_path / "calls.txt"
    fake = tmp_path / "sbatch"
    fake.write_text(
        "#!/bin/bash\n"
        f"echo \"$@\" >> {calls}\n"
        "n=$(wc -l < " + str(calls) + ")\n"
        "echo '-----------------------------------------------------------------'\n"
        "echo '          Welcome to the Vista Supercomputer'\n"
        "echo '--> Checking available allocation (TG-TEST)...OK'\n"
        "echo $((1000 + n))\n")
    fake.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "ACCOUNT": "TG-TEST", "COSMIC_ROOT": str(tmp_path),
           "SIF_IMAGE": "/x.sif", "WORLD_SIZES": "1 2", "RUNS": "1", "WARMUP_RUNS": "0",
           "RESULTS_ROOT": str(tmp_path / "results")}

    subprocess.run(["bash", SWEEP], env=env, capture_output=True, text=True, check=True)

    lines = calls.read_text().splitlines()
    assert len(lines) == 2
    assert _option(lines[0], "dependency") is None
    assert _option(lines[1], "dependency") == "afterany:1001"


def test_sweep_without_data_sizes_uses_the_file_limit_as_the_partition_count():
    lines = _dry_run()

    assert {_export(line, "PARTITIONS") for line in lines} == {"41"}


def test_data_size_sweep_converts_gigabytes_to_partitions_at_a_fixed_world_size():
    lines = _dry_run(WORLD_SIZES="768", DATA_SIZES_GB="12.6 1000", PARTITION_MB="100", RUNS="1",
                     WARMUP_RUNS="0", DATA_DIR="/scratch/u/cosmic/data/100MB")

    assert [(_export(line, "PARTITIONS"), _option(line, "nodes")) for line in lines] == [
        ("130", "32"), ("10240", "32")]
    assert all(_export(line, "DATA_DIR") == "/scratch/u/cosmic/data/100MB" for line in lines)
    assert [_export(line, "DATA_GB") for line in lines] == ["12.6", "1000"]
    assert len({_export(line, "RUN_DIR") for line in lines}) == 2
    assert "gb1000" in _export(lines[1], "RUN_DIR")


def test_sweep_passes_the_inference_unit_and_defaults_to_rank():
    default = _dry_run(WORLD_SIZES="1", RUNS="1", WARMUP_RUNS="0")
    per_partition = _dry_run(WORLD_SIZES="1", RUNS="1", WARMUP_RUNS="0", INFERENCE_UNIT="partition")

    assert _export(default[0], "INFERENCE_UNIT") == "rank"
    assert _export(per_partition[0], "INFERENCE_UNIT") == "partition"


def test_data_size_sweep_requires_the_partition_size():
    result = subprocess.run(["bash", SWEEP, "--dry-run"],
                            env={"PATH": os.environ["PATH"], "ACCOUNT": "TG-TEST", "COSMIC_ROOT": "/x",
                                 "SIF_IMAGE": "/x.sif", "DATA_SIZES_GB": "100"},
                            capture_output=True, text=True)

    assert result.returncode != 0
    assert "PARTITION_MB" in result.stderr


def test_sweep_refuses_to_run_without_an_account():
    result = subprocess.run(["bash", SWEEP, "--dry-run"],
                            env={"PATH": os.environ["PATH"], "COSMIC_ROOT": "/x", "SIF_IMAGE": "/x.sif"},
                            capture_output=True, text=True)

    assert result.returncode != 0
    assert "ACCOUNT" in result.stderr

def test_sweep_defaults_to_cpu_and_keeps_gpu_results_apart_from_cpu_results():
    cpu = _dry_run(WORLD_SIZES="1", RUNS="1", WARMUP_RUNS="0")
    gpu = _dry_run(WORLD_SIZES="1", RUNS="1", WARMUP_RUNS="0", DEVICE="cuda", PARTITION="gh", CORES_PER_NODE="72")

    assert _export(cpu[0], "DEVICE") == "cpu"
    assert _export(gpu[0], "DEVICE") == "cuda"
    assert "/results/classic/" in _export(cpu[0], "RUN_DIR")
    assert "/results/gpu/" in _export(gpu[0], "RUN_DIR")
    assert _option(gpu[0], "partition") == "gh"
    assert "cuda" in _option(gpu[0], "job-name")


def test_sweep_refuses_an_unknown_device():
    result = subprocess.run(["bash", SWEEP, "--dry-run"],
                            env={"PATH": os.environ["PATH"], "ACCOUNT": "TG-TEST", "COSMIC_ROOT": "/x",
                                 "SIF_IMAGE": "/x.sif", "DEVICE": "tpu"},
                            capture_output=True, text=True)

    assert result.returncode != 0
    assert "DEVICE" in result.stderr


def _run_job_with_stubs(tmp_path, **env):
    calls = tmp_path / "calls.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("module", "srun", "apptainer"):
        stub = bin_dir / name
        stub.write_text(f"#!/bin/bash\necho \"{name} $*\" >> {calls}\n")
        stub.chmod(0o755)
    run_dir = tmp_path / "runs" / "run1"
    base = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "WORLD_SIZE": "2", "FILE_LIMIT": "130",
            "BATCH_SIZE": "512", "CPUS_PER_WORKER": "6", "RUN_DIR": str(run_dir), "SIF_IMAGE": "/x.sif",
            "COSMIC_ROOT": str(tmp_path)}
    subprocess.run(["bash", os.path.join(_SCRIPTS, "cosmic_classic.slurm")], env={**base, **env},
                   capture_output=True, text=True, check=True)
    with open(run_dir / "timing.json") as f:
        timing = json.load(f)
    return calls.read_text().splitlines(), timing


@pytest.mark.parametrize("device, gpu_flag", [("cpu", False), ("cuda", True)])
def test_job_gives_only_the_workers_the_gpu_and_records_the_device(tmp_path, device, gpu_flag):
    calls, timing = _run_job_with_stubs(tmp_path, DEVICE=device)

    workers = [line for line in calls if line.startswith("srun")]
    host_side = [line for line in calls if line.startswith("apptainer")]
    assert len(workers) == 1 and len(host_side) == 2
    assert (" --nv " in workers[0]) is gpu_flag
    assert f"--device {device}" in workers[0]
    assert all(" --nv " not in line for line in host_side)
    assert timing["device"] == device


def test_job_defaults_to_cpu(tmp_path):
    calls, timing = _run_job_with_stubs(tmp_path)

    assert timing["device"] == "cpu"
    assert "--device cpu" in [line for line in calls if line.startswith("srun")][0]
