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
                epoch_batch_size=4, live=True, galaxies=152,
                reuse_tolerance=0.0091, outlier_threshold=0.027677)
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


def test_arm_order_cycles_through_every_permutation():
    """Every arm must lead equally often, and — the part rotation misses —
    every arm must FOLLOW every other equally often."""
    m = _driver()
    orders = [m.arm_order(i) for i in range(6)]

    assert len(set(orders)) == 6
    assert m.arm_order(6) == m.arm_order(0)


def test_no_arm_systematically_follows_the_same_arm():
    """Rotation alone leaves a fixed cyclic sequence: isolated would always
    follow langchain and never armada. Both touch Redis and ContextTable state,
    and isolated supplies the headline number, so leftover-state bias would land
    on it in one constant undetected direction."""
    m = _driver()
    predecessors = {arm: set() for arm in m.ARMS}
    for i in range(6):
        order = m.arm_order(i)
        for earlier, later in zip(order, order[1:]):
            predecessors[later].add(earlier)

    for arm, preds in predecessors.items():
        assert preds == set(m.ARMS) - {arm}, f"{arm} never follows {set(m.ARMS) - {arm} - preds}"


def test_grouped_launches_never_split_a_run():
    m = _driver()
    launches = m.plan_launches("all", 3)

    assert len(launches) == 9
    for run_index in range(3):
        in_run = [arm for idx, arm in launches if idx == run_index]
        assert tuple(in_run) == m.arm_order(run_index)
    assert [idx for idx, _ in launches] == sorted(idx for idx, _ in launches)


def test_both_still_means_the_two_sharing_arms():
    """It meant exactly {armada, langchain} before the control existed. Folding
    it into "all" would silently add a third arm — 50% more Fargate and Bedrock
    — to every caller already passing it."""
    m = _driver()
    arms = {arm for _, arm in m.plan_launches("both", 6)}

    assert arms == {"armada", "langchain"}
    assert len(m.plan_launches("both", 3)) == 6


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
    assert args.scaling == "strong"
    assert args.backend == "all"
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

def test_every_arm_disables_the_context_table_snapshot():
    """The snapshot serializes the whole ContextTable to Redis on every store.
    It has to be off, and off on EVERY arm — giving one arm a serialization
    step the others do not pay is exactly the asymmetry this experiment
    exists to rule out."""
    m = _driver()
    env_for = lambda arm: {
        e["name"]: e["value"]
        for e in m.build_overrides(0, 4, "c", "p/", _args(),
                                   backend=arm)["containerOverrides"][0]["environment"]
    }

    for arm in ("armada", "langchain", "isolated"):
        assert env_for(arm)["CONTEXT_TABLE_SNAPSHOT"] == "0"


def test_all_three_arms_launch_per_run():
    m = _driver()
    arms = {arm for _, arm in m.plan_launches("all", 1)}
    assert arms == {"armada", "langchain", "isolated"}


def test_no_arm_is_systematically_first():
    """Arm order varies across runs so session drift cannot be mistaken for an
    effect. The cycle is one full set of permutations, so each arm leads
    exactly the same number of times over it."""
    from collections import Counter

    m = _driver()
    leaders = Counter(m.arm_order(i)[0] for i in range(6))

    assert set(leaders) == {"armada", "langchain", "isolated"}
    assert len(set(leaders.values())) == 1


def test_every_arm_receives_the_same_reuse_tolerance_and_outlier_threshold():
    """Both have to be identical across arms or they become the difference
    being measured, exactly like the snapshot flag."""
    m = _driver()
    seen = set()
    for arm in ("armada", "langchain", "isolated"):
        env = {e["name"]: e["value"] for e in m.build_overrides(
            0, 4, "c", "p/", _args(), backend=arm)["containerOverrides"][0]["environment"]}
        seen.add((env["REUSE_KEY_TOLERANCE"], env["OUTLIER_RESIDUAL_THRESHOLD"]))
    assert len(seen) == 1


def test_the_outlier_threshold_is_pinned_by_default():
    """Unpinned, the prompt generator derives it per shard and the same galaxy
    gets a different prompt at a different N — so the scaling curve would mix
    the isolation effect with a moving workload."""
    m = _driver()
    env = {e["name"]: e["value"] for e in m.build_overrides(
        0, 4, "c", "p/", _args(), backend="armada")["containerOverrides"][0]["environment"]}
    assert float(env["OUTLIER_RESIDUAL_THRESHOLD"]) > 0


def test_the_isolated_arm_runs_on_the_cylon_store():
    m = _driver()
    env = {e["name"]: e["value"] for e in m.build_overrides(
        0, 4, "c", "p/", _args(), backend="isolated")["containerOverrides"][0]["environment"]}
    assert env["CONTEXT_BACKEND"] == "cylon"
    assert env["EXECUTION_BACKEND"] == "isolated"


def test_strong_scaling_is_the_default():
    """Weak scaling holds per-rank load constant, so it structurally cannot show
    the isolation penalty — the sweep's headline needs a shrinking shard."""
    assert _driver().build_parser().parse_args([]).scaling == "strong"
