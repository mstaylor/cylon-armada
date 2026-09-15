"""Sweep driver for the Experiment E Fargate runs.

Order alternation is the cheapest protection the headline number has: Bedrock
latency drifts over a session, so running every Arm A before every Arm B would
convert that drift into an apparent effect with a convincing error bar. And
the two arms of a run must never overlap in time — concurrent arms would share
Redis and Bedrock quota and throttle each other.

Run: pytest tests/test_fargate_cosmic_sweep.py -v
"""

import argparse
import importlib.util
import os

import pytest

_DRIVER = os.path.join(os.path.dirname(__file__), "..", "target", "aws", "scripts",
                       "experiment", "fargate_cosmic_poc.py")


def _driver():
    spec = importlib.util.spec_from_file_location("sweep", _DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    base = dict(listen_port=10000, timeout_ms=120000, device="cpu", batch_size=32,
                epoch_batch_size=4, live=True, galaxies=152)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_weak_scaling_holds_per_rank_load_constant():
    m = _driver()
    assert m.galaxies_for("weak", 1, per_rank=19, total=1253) == 19
    assert m.galaxies_for("weak", 8, per_rank=19, total=1253) == 152
    assert m.galaxies_for("weak", 64, per_rank=19, total=1253) == 1216


def test_weak_scaling_never_exceeds_the_real_population():
    """Padding by repeating galaxies would inflate h by construction."""
    m = _driver()
    assert m.galaxies_for("weak", 64, per_rank=64, total=1253) <= 1253


def test_strong_scaling_holds_the_total_constant():
    m = _driver()
    for world_size in (1, 2, 8, 64):
        assert m.galaxies_for("strong", world_size, per_rank=19, total=1253) == 1253


def test_arm_order_alternates_between_runs():
    m = _driver()
    assert m.arm_order(0) == ("armada", "langchain")
    assert m.arm_order(1) == ("langchain", "armada")
    assert m.arm_order(2) == ("armada", "langchain")


def test_paired_launches_alternate_and_never_split_a_run():
    m = _driver()
    assert m.plan_launches("both", 3) == [
        (0, "armada"), (0, "langchain"),
        (1, "langchain"), (1, "armada"),
        (2, "armada"), (2, "langchain"),
    ]


def test_a_single_arm_keeps_one_launch_per_run():
    m = _driver()
    assert m.plan_launches("langchain", 3) == [(0, "langchain"), (1, "langchain"), (2, "langchain")]
    assert m.plan_launches("armada", 2) == [(0, "armada"), (1, "armada")]


def test_overrides_carry_the_backend_to_the_task():
    m = _driver()
    overrides = m.build_overrides(1, 8, "comm-x", "p/", _args(), backend="langchain")
    env = {e["name"]: e["value"] for e in overrides["containerOverrides"][0]["environment"]}

    assert env["EXECUTION_BACKEND"] == "langchain"
    assert env["CONTEXT_BACKEND"] == "redis"
    assert env["WORLD_SIZE"] == "8"
    assert env["EPOCH_BATCH_SIZE"] == "4"
    assert env["GALAXIES"] == "152"


def test_armada_backend_selects_the_cylon_context_store():
    m = _driver()
    overrides = m.build_overrides(0, 8, "comm-x", "p/", _args(), backend="armada")
    env = {e["name"]: e["value"] for e in overrides["containerOverrides"][0]["environment"]}

    assert env["CONTEXT_BACKEND"] == "cylon"


def test_epoch_and_inference_batch_sizes_are_distinct_variables():
    """The launcher has two batch sizes that mean different things."""
    m = _driver()
    overrides = m.build_overrides(0, 4, "c", "p/", _args(batch_size=32, epoch_batch_size=4),
                                  backend="armada")
    env = {e["name"]: e["value"] for e in overrides["containerOverrides"][0]["environment"]}

    assert env["INFERENCE_BATCH_SIZE"] == "32"
    assert env["EPOCH_BATCH_SIZE"] == "4"


def test_default_world_sizes_include_one_and_two():
    m = _driver()
    defaults = m.build_parser().parse_args([]).world_sizes
    assert defaults[:2] == [1, 2]
    assert defaults == [1, 2, 4, 8, 16, 32, 64]


def test_defaults_match_the_spec():
    m = _driver()
    args = m.build_parser().parse_args([])
    assert args.scaling == "weak"
    assert args.backend == "both"
    assert args.runs == 5
    assert args.per_rank == 19
    assert args.epoch_batch_size == 4


def test_dry_run_launches_nothing_and_touches_no_client():
    """A dry run must describe every launch without calling ECS at all."""
    m = _driver()

    class ExplodingEcs:
        def __getattr__(self, name):
            raise AssertionError(f"ECS client used during dry run: {name}")

    args = m.build_parser().parse_args(["--dry-run", "--runs", "2"])
    assert m.launch_world_size(ExplodingEcs(), 4, args) == []

def test_both_arms_disable_the_context_table_snapshot():
    """The snapshot serializes the whole ContextTable to Redis on every store.
    It has to be off, and off on BOTH arms — giving one arm a serialization
    step the other does not pay is exactly the asymmetry this experiment
    exists to rule out."""
    m = _driver()
    env_for = lambda arm: {
        e["name"]: e["value"]
        for e in m.build_overrides(0, 4, "c", "p/", _args(),
                                   backend=arm)["containerOverrides"][0]["environment"]
    }

    assert env_for("armada")["CONTEXT_TABLE_SNAPSHOT"] == "0"
    assert env_for("langchain")["CONTEXT_TABLE_SNAPSHOT"] == "0"
