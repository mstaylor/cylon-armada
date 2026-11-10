import argparse
import importlib.util
import os
import subprocess
import sys
import types

import pytest

_DRIVER = os.path.join(os.path.dirname(__file__), "..", "target", "aws", "scripts",
                       "experiment", "fargate_expb_sweep.py")

LEGACY_ENV_NAMES = ["RANK", "WORLD_SIZE", "REDIS_ADDR", "EXPB_RUNS", "RESULTS_BUCKET",
                    "S3_RESULTS_PREFIX", "CYLON_SESSION_ID"]


def _driver():
    spec = importlib.util.spec_from_file_location("expb_sweep", _DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env(overrides):
    return {e["name"]: e["value"] for e in overrides["containerOverrides"][0]["environment"]}


def _args(**overrides):
    values = dict(world_sizes=[1, 2], runs=4, dry_run=True, channel="fmi-direct-redis",
                  payload="float32", task_definition="cylon-armada-python",
                  results_prefix="cylon-armada/results/exp_b_collectives_fargate",
                  scripts_prefix=None, max_start_retries=3, poll_s=0.0, wait_timeout_s=60.0,
                  allgather_algorithm=None)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_default_flags_reproduce_the_existing_direct_redis_launch():
    sweep = _driver()

    env = _env(sweep.build_overrides(0, 2, 4, "p/ws2/", "sess", "fmi-direct-redis", "float32",
                                     None))

    assert list(env) == LEGACY_ENV_NAMES + ["EXPB_CHANNEL", "EXPB_PAYLOAD"]
    assert env["EXPB_CHANNEL"] == "fmi-direct-redis"
    assert env["EXPB_PAYLOAD"] == "float32"


def test_a_chosen_allgather_algorithm_reaches_every_cylon_rank():
    sweep = _driver()

    env = _env(sweep.build_overrides(3, 4, 4, "p/", "sess", "fmi-direct-redis", "float32", None,
                                     allgather_algorithm="recursive_doubling"))

    assert env["FMI_ALLGATHER_ALGORITHM"] == "recursive_doubling"


def test_launch_passes_the_allgather_algorithm_to_each_rank():
    ecs = _FakeEcs()

    _launch(ecs, 2, allgather_algorithm="recursive_doubling")

    assert all(_env(launch["overrides"])["FMI_ALLGATHER_ALGORITHM"] == "recursive_doubling"
               for launch in ecs.launches)


def test_ray_channels_carry_the_rendezvous_name_and_redis_host():
    sweep = _driver()

    env = _env(sweep.build_overrides(1, 2, 4, "p/", "sess", "ray-gloo", "kv_pairs", "scripts-x/"))

    assert env["COMM_NAME"] == "sess"
    assert (env["REDIS_HOST"], env["REDIS_PORT"]) == tuple(sweep.REDIS_ADDR.split(":"))
    assert env["S3_SCRIPTS_PREFIX"] == "scripts-x/"


def test_ray_channels_get_the_ray_security_group_and_cylon_does_not():
    sweep = _driver()

    assert sweep.security_groups_for_channel("ray-gloo", "sg-1") == ["sg-1"]
    assert sweep.security_groups_for_channel("ray-objectstore", "sg-1") == ["sg-1"]
    assert sweep.security_groups_for_channel("fmi-direct-redis", "sg-1") is None


def test_results_land_under_channel_payload_and_world_size():
    sweep = _driver()

    assert sweep.results_prefix_for("r/expb", "ray-gloo", "kv_pairs", 8) == "r/expb/ray-gloo/kv_pairs/ws8/"


class _NoAws:
    def __getattr__(self, name):
        raise AssertionError(f"AWS call attempted: {name}")


def test_dry_run_makes_no_aws_calls(monkeypatch):
    sweep = _driver()
    monkeypatch.setattr(sweep.boto3, "client", lambda *a, **k: _NoAws())
    monkeypatch.setattr(sys, "argv", ["sweep", "--channel", "ray-gloo", "--payload", "kv_pairs",
                                      "--task-definition", "cylon-armada-cosmic",
                                      "--world-sizes", "1", "2", "--dry-run"])

    sweep.main()


def _rank_of(launch):
    return int(_env(launch["overrides"])["RANK"])


class _FakeEcs:
    """Each launch's fate comes from outcome(rank, attempt): an exit code, or
    'no-start' for a task ECS stops before its container runs, or 'hang'."""

    def __init__(self, outcome=lambda rank, attempt: 0):
        self.launches = []
        self.stopped = []
        self._outcome = outcome
        self._fate = {}

    def run_task(self, **kwargs):
        self.launches.append(kwargs)
        rank = _rank_of(kwargs)
        attempt = sum(1 for launch in self.launches if _rank_of(launch) == rank)
        arn = f"arn:{len(self.launches)}"
        self._fate[arn] = self._outcome(rank, attempt)
        return {"tasks": [{"taskArn": arn}], "failures": []}

    def describe_tasks(self, cluster, tasks):
        described = []
        for arn in tasks:
            fate = self._fate[arn]
            if fate == "hang" and arn not in self.stopped:
                described.append({"taskArn": arn, "lastStatus": "RUNNING", "containers": [{}]})
            elif fate == "no-start":
                described.append({"taskArn": arn, "lastStatus": "STOPPED",
                                  "stopCode": "TaskFailedToStart",
                                  "stoppedReason": "CannotPullContainerError: i/o timeout",
                                  "containers": [{}]})
            else:
                code = None if fate == "hang" else fate
                described.append({"taskArn": arn, "lastStatus": "STOPPED",
                                  "stopCode": "EssentialContainerExited",
                                  "containers": [{"exitCode": code}]})
        return {"tasks": described}

    def stop_task(self, cluster, task, reason):
        self.stopped.append(task)
        return {"task": {"taskArn": task}}


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 1.0)


def _launch(ecs, world_size, clock=None, **arg_overrides):
    sweep = _driver()
    clock = clock or _Clock()
    return sweep.launch_world_size(ecs, world_size, _args(dry_run=False, **arg_overrides), None,
                                   sleep=clock.sleep, clock=clock)


def test_a_rank_that_fails_to_start_is_relaunched_with_the_same_overrides():
    ecs = _FakeEcs(lambda rank, attempt: "no-start" if (rank, attempt) == (3, 1) else 0)

    codes = _launch(ecs, 4)

    rank3 = [launch for launch in ecs.launches if _rank_of(launch) == 3]
    assert len(rank3) == 2
    assert rank3[0]["overrides"] == rank3[1]["overrides"]
    assert len(codes) == 4 and set(codes.values()) == {0}


def test_a_rank_that_never_starts_is_reported_failed_after_the_retry_limit():
    ecs = _FakeEcs(lambda rank, attempt: "no-start" if rank == 1 else 0)

    codes = _launch(ecs, 2, max_start_retries=2)

    assert sum(1 for launch in ecs.launches if _rank_of(launch) == 1) == 3
    assert sorted(codes.values(), key=str) == [0, None]


def test_a_container_that_exits_nonzero_is_not_relaunched():
    ecs = _FakeEcs(lambda rank, attempt: 1 if rank == 0 else 0)

    codes = _launch(ecs, 2)

    assert len(ecs.launches) == 2
    assert sorted(codes.values()) == [0, 1]


def test_tasks_still_running_at_the_wait_timeout_are_stopped_and_reported_failed():
    ecs = _FakeEcs(lambda rank, attempt: "hang" if rank == 1 else 0)

    codes = _launch(ecs, 2, wait_timeout_s=30.0)

    assert len(ecs.stopped) == 1
    assert sorted(codes.values(), key=str) == [0, None]


def test_launch_passes_the_task_definition_and_security_groups():
    sweep = _driver()
    ecs = _FakeEcs()

    codes = sweep.launch_world_size(ecs, 2, _args(channel="ray-gloo", dry_run=False,
                                                  task_definition="cylon-armada-cosmic"), ["sg-1"])

    assert set(codes.values()) == {0}
    assert {launch["taskDefinition"] for launch in ecs.launches} == {"cylon-armada-cosmic"}
    assert all(launch["networkConfiguration"]["awsvpcConfiguration"]["securityGroups"] == ["sg-1"]
               for launch in ecs.launches)


def test_a_cylon_launch_keeps_the_default_security_group():
    sweep = _driver()
    ecs = _FakeEcs()

    sweep.launch_world_size(ecs, 1, _args(dry_run=False), None)

    assert "securityGroups" not in ecs.launches[0]["networkConfiguration"]["awsvpcConfiguration"]


def _run_runner(monkeypatch, tmp_path, channel):
    sweep = _driver()
    commands = []
    fake_s3 = types.SimpleNamespace(download_file=lambda *a: None, upload_file=lambda *a: None)
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda name: fake_s3))
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: commands.append((cmd, kw)) or
                        types.SimpleNamespace(returncode=0))
    for name, value in {"WORLD_SIZE": "2", "EXPB_RUNS": "4", "REDIS_ADDR": "h:6379",
                        "S3_RESULTS_PREFIX": "p/", "EXPB_CHANNEL": channel,
                        "EXPB_PAYLOAD": "kv_pairs", "EXPB_OUTPUT_DIR": str(tmp_path)}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("S3_SCRIPTS_PREFIX", raising=False)
    with pytest.raises(SystemExit) as exited:
        exec(compile(sweep.RUNNER_SCRIPT, "runner", "exec"), {"__name__": "__main__"})
    assert exited.value.code == 0
    return commands[0]


def test_runner_runs_exp_b_ray_as_a_module_for_ray_channels(monkeypatch, tmp_path):
    cmd, kwargs = _run_runner(monkeypatch, tmp_path, "ray-objectstore")

    assert cmd[1:3] == ["-m", "experiment.exp_b_ray"]
    assert cmd[cmd.index("--channel") + 1] == "ray-objectstore"
    assert cmd[cmd.index("--payload") + 1] == "kv_pairs"
    assert "--redis-addr" not in cmd
    assert kwargs["cwd"] == "/cylon-armada/scripts"


def test_runner_runs_the_cylon_benchmark_for_the_direct_redis_channel(monkeypatch, tmp_path):
    cmd, _ = _run_runner(monkeypatch, tmp_path, "fmi-direct-redis")

    assert cmd[1:3] == ["-m", "experiment.exp_b_collectives"]
    assert cmd[cmd.index("--redis-addr") + 1] == "h:6379"