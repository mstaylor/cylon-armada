"""Sweep driver for the Experiment E Fargate runs.

Order alternation is the cheapest protection the headline number has: Bedrock
latency drifts over a session, so running every Arm A before every Arm B would
convert that drift into an apparent effect with a convincing error bar. And
the two arms of a run must never overlap in time — concurrent arms would share
Redis and Bedrock quota and throttle each other.

Run: pytest tests/test_fargate_cosmic_sweep.py -v
"""

import argparse
import importlib
import importlib.util
import json
import os

import pytest

_DRIVER = os.path.join(os.path.dirname(__file__), "..", "target", "aws", "scripts",
                       "experiment", "fargate_cosmic_poc.py")


def _driver():
    spec = importlib.util.spec_from_file_location("sweep", _DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SHARED_SCRIPTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "target", "shared", "scripts"))


def _shared_module(dotted):
    """Import a shared-scripts module by absolute path.

    Mirrors _driver's seam. Inserting a relative path into sys.path instead
    would depend on pytest's working directory and would leak into every test
    that ran afterwards in the session.
    """
    import sys

    added = _SHARED_SCRIPTS not in sys.path
    if added:
        sys.path.insert(0, _SHARED_SCRIPTS)
    try:
        return importlib.import_module(dotted)
    finally:
        if added:
            sys.path.remove(_SHARED_SCRIPTS)


def _args(**overrides):
    base = dict(listen_port=10000, timeout_ms=120000, device="cpu", batch_size=32,
                epoch_batch_size=4, live=True, galaxies=152,
                reuse_tolerance=0.0091, outlier_threshold=0.027677, gloo_socket_ifname=None, ray_dashboard_hold_s=0)
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
    n_orders = len(m._ORDERS)
    orders = [m.arm_order(i) for i in range(n_orders)]

    assert len(set(orders)) == n_orders
    assert m.arm_order(n_orders) == m.arm_order(0)


def test_no_arm_systematically_follows_the_same_arm():
    """Rotation alone leaves a fixed cyclic sequence where certain arms always
    follow the same arm. Both touch Redis and ContextTable state, and some arms
    supply headline numbers, so leftover-state bias would land on them in one
    constant undetected direction. A stride through permutations ensures variety."""
    m = _driver()
    n_orders = len(m._ORDERS)
    predecessors = {arm: set() for arm in m.ARMS}
    for i in range(n_orders):
        order = m.arm_order(i)
        for earlier, later in zip(order, order[1:]):
            predecessors[later].add(earlier)

    for arm, preds in predecessors.items():
        assert preds == set(m.ARMS) - {arm}, f"{arm} never follows {set(m.ARMS) - {arm} - preds}"


def test_grouped_launches_never_split_a_run():
    m = _driver()
    launches = m.plan_launches("all", 3)

    assert len(launches) == 3 * len(m.LAUNCHABLE_ARMS)
    for run_index in range(3):
        in_run = [arm for idx, arm in launches if idx == run_index]
        expected = tuple(a for a in m.arm_order(run_index) if a in m.LAUNCHABLE_ARMS)
        assert tuple(in_run) == expected
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
    assert args.runs == 4
    assert args.warmup_runs == 1
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

    for arm in m.ARMS:
        assert env_for(arm)["CONTEXT_TABLE_SNAPSHOT"] == "0"


def test_all_arms_launch_when_all_selected():
    """'all' launches every arm that has an executor. The Ray arms are
    registered names without one, so including them would spend a Fargate task
    per rank to produce an error record or nothing at all."""
    m = _driver()
    arms = {arm for _, arm in m.plan_launches("all", 1)}
    assert arms == set(m.LAUNCHABLE_ARMS)


def test_no_arm_is_systematically_first():
    """Arm order varies across runs so session drift cannot be mistaken for an
    effect. The full cycle is one pass through every permutation, so each arm leads
    exactly the same number of times over it."""
    from collections import Counter

    m = _driver()
    n_orders = len(m._ORDERS)
    leaders = Counter(m.arm_order(i)[0] for i in range(n_orders))

    assert set(leaders) == set(m.ARMS)
    assert len(set(leaders.values())) == 1


def test_leader_diversity_over_short_runs():
    """Production sweeps run only len(ARMS) repetitions, not the full permutation
    cycle. Over that short window, leader diversity must not degrade. A broken
    stride (e.g. consecutive stepping with 5 arms) would give [armada, armada,
    armada, armada, armada] — satisfying full-cycle fairness but failing the
    production constraint. This test catches that regression."""
    m = _driver()
    n_arms = len(m.ARMS)
    leaders = set(m.arm_order(i)[0] for i in range(n_arms))

    assert leaders == set(m.ARMS), \
        f"Over {n_arms} runs, not all arms led; got leaders {leaders}"


def test_every_arm_receives_the_same_reuse_tolerance_and_outlier_threshold():
    """Both have to be identical across arms or they become the difference
    being measured, exactly like the snapshot flag."""
    m = _driver()
    seen = set()
    for arm in m.ARMS:
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


# ---------------------------------------------------------------------------
# Task startup timing
# ---------------------------------------------------------------------------

def _task(created, pull_start, pull_stop, started, stopping, stopped):
    """An ECS describe_tasks entry with the lifecycle timestamps we read."""
    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
    def at(offset):
        return None if offset is None else base + timedelta(seconds=offset)
    out = {"taskArn": "arn:aws:ecs:us-east-1:1:task/c/abc", "createdAt": at(created)}
    for key, off in (("pullStartedAt", pull_start), ("pullStoppedAt", pull_stop),
                     ("startedAt", started), ("stoppingAt", stopping),
                     ("stoppedAt", stopped)):
        if off is not None:
            out[key] = at(off)
    return out


def test_timing_decomposes_a_task_lifecycle():
    """The point of the measurement: separate image pull from everything else."""
    sweep = _driver()
    t = sweep.task_timing(_task(created=0, pull_start=10, pull_stop=80,
                                started=85, stopping=200, stopped=205))
    assert t["provision_s"] == 10.0
    assert t["pull_s"] == 70.0
    assert t["run_s"] == 115.0
    assert t["teardown_s"] == 5.0
    assert t["total_s"] == 205.0


def test_timing_tolerates_a_task_that_never_pulled():
    """ECS omits pullStartedAt when provisioning fails. A failed run must not
    crash the driver on the way out, or the results it did produce are lost."""
    sweep = _driver()
    t = sweep.task_timing(_task(created=0, pull_start=None, pull_stop=None,
                                started=None, stopping=None, stopped=30))
    assert t["pull_s"] is None
    assert t["provision_s"] is None
    assert t["total_s"] == 30.0


def test_timing_summary_reports_the_slowest_task_not_the_mean():
    """An arm finishes when its slowest rank finishes, so the arm's startup cost
    is the max across ranks. A mean would understate what the sweep actually
    waits for."""
    sweep = _driver()
    tasks = [_task(0, 5, 40, 45, 100, 105), _task(0, 5, 70, 75, 100, 110)]
    s = sweep.summarize_timings([sweep.task_timing(t) for t in tasks])
    assert s["ranks"] == 2
    assert s["pull_s_max"] == 65.0
    assert s["total_s_max"] == 110.0


def test_every_arm_launches_on_the_cosmic_task_definition():
    """One image carries Ray for every arm, so no arm pulls a different image
    from the others."""
    m = _driver()

    assert m.ARMS_WITHOUT_EXECUTOR == ()
    for arm in ("ray-native", "ray-cylon"):
        assert m._selected_arms(arm) == (arm,)
    args = m.build_parser().parse_args(["--dry-run", "--backend", "all", "--world-sizes", "1", "2"])
    assert {row["task_definition"] for row in m.sweep_matrix(args)} == {m.TASK_DEFINITION}


def test_ray_comparison_selects_armada_and_every_ray_arm():
    m = _driver()
    assert m._selected_arms("ray-comparison") == (
        "armada", "ray-native", "ray-collective", "ray-cylon")
    assert "ray-comparison" in m.build_parser().parse_args(
        ["--backend", "ray-comparison"]).backend


def test_ray_comparison_leaders_over_the_four_measured_runs():
    """Whoever leads a run pays any leftover cold-start cost. Where ray-cylon
    runs, four arms over four runs lead once each. Elsewhere four runs cannot
    split evenly over three arms, and the extra lead falls on ray-collective
    rather than on either headline arm. Recorded so a change to it is noticed."""
    m = _driver()
    arms = m._selected_arms("ray-comparison")
    trimmed = tuple(arm for arm in arms if arm != "ray-cylon")
    assert [m.order_for(arms, i)[0] for i in range(4)] == [
        "armada", "ray-native", "ray-collective", "ray-cylon"]
    assert [m.order_for(trimmed, i)[0] for i in range(4)] == [
        "armada", "ray-collective", "ray-collective", "ray-native"]


@pytest.mark.parametrize("arms", [("armada", "langchain"), ("armada", "ray-native")])
def test_two_arm_comparisons_alternate_who_leads(arms):
    """Filtering the five-arm cycle down to two arms let one arm lead three of
    four runs. Ordering over the two arms' own permutations alternates them."""
    m = _driver()
    assert [m.order_for(arms, i) for i in range(4)] == [arms, arms[::-1], arms, arms[::-1]]


@pytest.mark.parametrize("arms", [("armada", "langchain"), ("armada", "ray-native", "ray-cylon"),
                                  ("armada", "langchain", "isolated", "ray-native", "ray-cylon")])
def test_every_selection_leads_equally_over_its_full_cycle(arms):
    from collections import Counter
    from math import factorial

    m = _driver()
    cycle = factorial(len(arms))
    leaders = Counter(m.order_for(arms, i)[0] for i in range(cycle))
    assert set(leaders) == set(arms)
    assert len(set(leaders.values())) == 1, dict(leaders)


def test_all_five_arms_keep_their_existing_order():
    m = _driver()
    assert [m.order_for(m.ARMS, i) for i in range(len(m._ORDERS))] == \
        [m.arm_order(i) for i in range(len(m._ORDERS))]


def test_plan_launches_orders_runs_over_the_selected_arms():
    m = _driver()
    launches = m.plan_launches("both", 2)
    assert launches == [(0, "armada"), (0, "langchain"), (1, "langchain"), (1, "armada")]


def test_ray_cylon_runs_only_at_its_configured_world_sizes():
    """ray-cylon is a portability check, not a sweep arm: its data plane is
    Armada's FMI bridge, so it runs only where --ray-cylon-world-sizes says."""
    m = _driver()
    args = m.build_parser().parse_args(
        ["--dry-run", "--backend", "ray-comparison", "--world-sizes", "1", "2", "4", "8"])
    rows = m.sweep_matrix(args)

    assert args.ray_cylon_world_sizes == [2, 8]
    ray_cylon_sizes = {r["world_size"] for r in rows if r["arm"] == "ray-cylon"}
    assert ray_cylon_sizes == {2, 8}
    assert {r["arm"] for r in rows if r["world_size"] in (1, 4)} == {
        "armada", "ray-native", "ray-collective"}


def test_ray_cylon_world_sizes_are_overridable():
    m = _driver()
    args = m.build_parser().parse_args(
        ["--dry-run", "--backend", "ray-comparison", "--world-sizes", "1", "2", "4",
         "--ray-cylon-world-sizes", "4"])
    assert {r["world_size"] for r in m.sweep_matrix(args) if r["arm"] == "ray-cylon"} == {4}


def test_naming_ray_cylon_alone_still_runs_it_at_every_requested_world_size():
    """The restriction trims ray-cylon out of a comparison; asking for the arm
    by itself is an explicit request and is honoured as given."""
    m = _driver()
    args = m.build_parser().parse_args(
        ["--dry-run", "--backend", "ray-cylon", "--world-sizes", "1", "2", "4"])
    assert {r["world_size"] for r in m.sweep_matrix(args)} == {1, 2, 4}


def test_warmups_precede_measured_runs_and_land_under_their_own_prefix():
    m = _driver()
    args = m.build_parser().parse_args(
        ["--dry-run", "--backend", "ray-comparison", "--runs", "2", "--warmup-runs", "1"])
    rows = m.sweep_matrix(args)
    ws2 = [r for r in rows if r["world_size"] == 2]

    phases = [r["phase"] for r in ws2]
    assert phases == ["warmup"] * 4 + ["measured"] * 8
    assert all("/warmup0/" in r["s3_prefix"] for r in ws2 if r["phase"] == "warmup")
    assert all("/warmup" not in r["s3_prefix"] for r in ws2 if r["phase"] == "measured")
    assert {r["arm"] for r in ws2} == {"armada", "ray-native", "ray-collective", "ray-cylon"}


def test_the_matrix_covers_every_world_size_and_counts_one_task_per_rank():
    m = _driver()
    args = m.build_parser().parse_args(["--dry-run", "--backend", "ray-comparison"])
    rows = m.sweep_matrix(args)

    assert sorted({r["world_size"] for r in rows})[:2] == [1, 2]
    assert all(r["tasks"] == r["world_size"] for r in rows)
    runs = args.runs + args.warmup_runs
    for n in args.world_sizes:
        arms = 4 if n in args.ray_cylon_world_sizes else 3
        assert sum(1 for r in rows if r["world_size"] == n) == arms * runs, n


def test_task_hours_estimate_multiplies_tasks_by_the_per_task_minutes():
    m = _driver()
    rows = [{"world_size": 4, "tasks": 4}, {"world_size": 2, "tasks": 2}]
    estimate = m.estimate_usage(rows, task_minutes=30, task_vcpu=4, task_memory_gb=8)
    assert estimate == {"launches": 2, "tasks": 6, "task_hours": 3.0,
                        "vcpu_hours": 12.0, "gb_hours": 24.0}


def test_dry_run_prints_the_matrix_and_the_estimate(capsys):
    m = _driver()
    args = m.build_parser().parse_args(
        ["--dry-run", "--backend", "ray-comparison", "--world-sizes", "1", "2",
         "--est-task-minutes", "10"])
    m.print_dry_run(args)
    out = capsys.readouterr().out
    assert "ray-native" in out and "ray-cylon" in out and "armada" in out
    assert "ray-collective" in out
    assert "task-hours" in out
    assert "55 tasks" in out
    assert "9.17 task-hours" in out


def test_all_includes_every_arm_but_both_still_means_two():
    """'both' has always meant exactly the two sharing arms. Folding the new
    arms into it would silently add Fargate and Bedrock cost to every caller
    that already passes it."""
    m = _driver()

    assert set(m._selected_arms("all")) == set(m.LAUNCHABLE_ARMS) == set(m.ARMS)
    assert m._selected_arms("both") == ("armada", "langchain")
    assert set(m.LAUNCHABLE_ARMS) | set(m.ARMS_WITHOUT_EXECUTOR) == set(m.ARMS)


def test_driver_and_runner_agree_on_every_arms_context_store():
    """The driver sets CONTEXT_BACKEND per arm and the runner refuses any
    store that is not its arm's own, so a disagreement fails every rank inside
    the task, which costs a Fargate launch to discover."""
    m = _driver()
    run_cosmic_local = _shared_module("armada.run_cosmic_local")

    for arm in m.ARMS:
        backend = run_cosmic_local.ExecutionBackend(arm)
        assert m._CONTEXT_STORE[arm] == run_cosmic_local._CONTEXT_STORE[backend], arm


def test_gating_the_ray_arms_does_not_unbalance_who_leads():
    """arm_order permutes all five registered arms and plan_launches then drops
    any without an executor (none today). Filtering a balanced sequence is not
    automatically balanced, and the arm that leads is the one that pays any
    cold-start or leftover-state cost. isolated supplies the headline
    isolation-penalty number, so a bias landing on it constantly would be
    invisible and would move that number."""
    from collections import Counter

    m = _driver()
    launchable = set(m.LAUNCHABLE_ARMS)
    leaders = Counter(
        next(a for a in m.arm_order(i) if a in launchable)
        for i in range(len(m._ORDERS))
    )

    assert set(leaders) == launchable
    assert len(set(leaders.values())) == 1, f"leads are uneven: {dict(leaders)}"

    short = [next(a for a in m.arm_order(i) if a in launchable)
             for i in range(len(m.LAUNCHABLE_ARMS))]
    assert set(short) == launchable, f"over {len(short)} runs the leaders were {short}"


class _RecordingEcs:
    def __init__(self):
        self.calls = []

    def run_task(self, **kwargs):
        self.calls.append(kwargs)
        return {"tasks": [{"taskArn": f"arn:{len(self.calls)}"}], "failures": []}


def test_ray_arm_tasks_launch_in_the_ray_security_group_and_others_do_not():
    """The VPC default group admits all TCP from anywhere and Ray's GCS and
    client ports are unauthenticated, so a Ray task must carry the
    members-only group. The other arms keep the network they were measured on."""
    m = _driver()
    for arm, expected in (("ray-native", ["sg-ray"]), ("ray-cylon", ["sg-ray"]),
                          ("armada", None), ("langchain", None)):
        ecs = _RecordingEcs()
        m._run_tasks(ecs, 2, lambda rank: {}, 0, 1, 60,
                     task_definition=m.TASK_DEFINITION,
                     security_groups=m.security_groups_for(arm, ["sg-ray"]))
        for call in ecs.calls:
            vpc = call["networkConfiguration"]["awsvpcConfiguration"]
            assert vpc.get("securityGroups") == expected, arm
            assert vpc["subnets"] == m.SUBNETS


def test_a_missing_ray_security_group_refuses_the_sweep():
    m = _driver()

    class Ec2:
        def __init__(self, groups):
            self.groups = groups

        def describe_security_groups(self, Filters):
            assert Filters == [{"Name": "group-name", "Values": [m.RAY_SECURITY_GROUP_NAME]}]
            return {"SecurityGroups": self.groups}

    assert m.resolve_ray_security_group(Ec2([{"GroupId": "sg-1"}])) == "sg-1"
    with pytest.raises(RuntimeError, match="terraform apply"):
        m.resolve_ray_security_group(Ec2([]))


def test_the_ray_security_group_name_matches_terraform_and_admits_only_members():
    m = _driver()
    main_tf = open(os.path.join(os.path.dirname(__file__), "..", "target", "aws", "scripts",
                                "terraform", "main.tf")).read()
    start = main_tf.index('resource "aws_security_group" "ray_tasks"')
    block = main_tf[start:main_tf.index("\n}\n", start)]
    assert m.RAY_SECURITY_GROUP_NAME == "cylon-armada-ray-tasks"
    assert 'name        = "${var.project_name}-ray-tasks"' in block
    ingress = block[block.index("ingress {"):block.index("egress {")]
    assert "self        = true" in ingress
    assert "cidr_blocks" not in ingress


def test_dry_run_prints_a_fargate_cost_estimate(capsys):
    m = _driver()
    args = m.build_parser().parse_args(
        ["--dry-run", "--backend", "ray-comparison", "--world-sizes", "1", "2",
         "--est-task-minutes", "10", "--vcpu-hour-usd", "0.04", "--gb-hour-usd", "0.004"])
    m.print_dry_run(args)
    out = capsys.readouterr().out
    assert "Estimated Fargate cost: $1.76" in out
    assert m.RAY_SECURITY_GROUP_NAME in out


def test_ray_collective_is_a_ray_arm_on_the_cylon_store():
    m = _driver()
    run_cosmic_local = _shared_module("armada.run_cosmic_local")

    assert "ray-collective" in m.ARMS and "ray-collective" in m.RAY_ARMS
    assert m.security_groups_for("ray-collective", ["sg-ray"]) == ["sg-ray"]
    assert run_cosmic_local.ExecutionBackend("ray-collective") is \
        run_cosmic_local.ExecutionBackend.RayCollective


def _env(overrides):
    return {e["name"]: e["value"] for e in overrides["containerOverrides"][0]["environment"]}


def test_the_gloo_interface_reaches_the_task_only_when_given():
    """Each exchanger derives Gloo's interface from its Ray node IP. The flag
    is an override and reaches the task only when given."""
    m = _driver()

    assert m.build_parser().parse_args(["--dry-run"]).gloo_socket_ifname is None
    assert m.build_parser().parse_args(
        ["--dry-run", "--gloo-socket-ifname", "eth1"]).gloo_socket_ifname == "eth1"
    assert "GLOO_SOCKET_IFNAME" not in _env(m.build_overrides(0, 2, "c", "p/", _args(), "ray-collective"))
    named = _args(gloo_socket_ifname="eth1")
    assert _env(m.build_overrides(0, 2, "c", "p/", named, "ray-collective"))["GLOO_SOCKET_IFNAME"] == "eth1"


def test_the_dashboard_hold_reaches_only_rank_zero_of_a_ray_arm():
    """The hold keeps the Ray head, and with it the dashboard, up after the
    rank has uploaded its result, so it never touches a measurement."""
    m = _driver()
    held = _args(ray_dashboard_hold_s=600)

    assert m.build_parser().parse_args(["--dry-run"]).ray_dashboard_hold_s == 0
    assert _env(m.build_overrides(0, 2, "c", "p/", held, "ray-native"))["RAY_DASHBOARD_HOLD_S"] == "600"
    assert "RAY_DASHBOARD_HOLD_S" not in _env(m.build_overrides(1, 2, "c", "p/", held, "ray-native"))
    assert "RAY_DASHBOARD_HOLD_S" not in _env(m.build_overrides(0, 2, "c", "p/", held, "armada"))
    assert "RAY_DASHBOARD_HOLD_S" not in _env(m.build_overrides(0, 2, "c", "p/", _args(), "ray-native"))


def test_the_runner_holds_rank_zero_only_after_uploading_its_result():
    m = _driver()
    script = m.RUNNER_SCRIPT
    compile(script, "runner", "exec")
    assert script.index("upload_file") < script.index("RAY_DASHBOARD_HOLD_S")


def test_only_rank_zero_of_a_ray_arm_launches_with_execute_command_for_the_dashboard():
    """The SSM agent ECS Exec adds is a process in the task, so it rides only
    on rank 0, where the dashboard lives, and only when asked for."""
    m = _driver()
    ecs = _RecordingEcs()
    m._run_tasks(ecs, 3, lambda rank: {"rank": rank}, 0, 1, 60, execute_command_ranks=(0,))
    assert {call["overrides"]["rank"]: call.get("enableExecuteCommand", False)
            for call in ecs.calls} == {0: True, 1: False, 2: False}

    plain = _RecordingEcs()
    m._run_tasks(plain, 2, lambda rank: {}, 0, 1, 60)
    assert not any(call.get("enableExecuteCommand") for call in plain.calls)


class _DescribingEcs:
    def __init__(self, responses):
        self.responses = list(responses)

    def describe_tasks(self, cluster, tasks):
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def _exec_task(status, agent_status=None, runtime_id="rt-1"):
    agents = [{"name": "ExecuteCommandAgent", "lastStatus": agent_status}] if agent_status else []
    return {"tasks": [{"taskArn": "arn:aws:ecs:us-east-1:1:task/CylonFargateExperiments/abc123",
                       "lastStatus": status,
                       "containers": [{"runtimeId": runtime_id, "managedAgents": agents}]}]}


def test_the_dashboard_target_is_rank_zeros_container_once_its_exec_agent_runs():
    m = _driver()
    ecs = _DescribingEcs([_exec_task("PENDING"), _exec_task("RUNNING", "PENDING"),
                          _exec_task("RUNNING", "RUNNING")])

    target = m.ray_dashboard_target(ecs, "arn:rank0", timeout_s=60, sleep=lambda s: None)

    assert target == "ecs:CylonFargateExperiments_abc123_rt-1"


def test_the_dashboard_target_is_none_when_rank_zero_stops_first_and_never_raises():
    m = _driver()

    class Broken:
        def describe_tasks(self, **kw):
            raise RuntimeError("throttled")

    stopped = _DescribingEcs([_exec_task("STOPPED")])
    assert m.ray_dashboard_target(stopped, "arn:rank0", timeout_s=60, sleep=lambda s: None) is None
    assert m.ray_dashboard_target(Broken(), "arn:rank0", timeout_s=60, sleep=lambda s: None) is None


def test_the_dashboard_tunnel_forwards_the_same_port_locally_through_ssm():
    m = _driver()
    command = m.dashboard_tunnel_command("ecs:c_t_r", 8265)
    assert command[:3] == ["aws", "ssm", "start-session"]
    assert command[command.index("--target") + 1] == "ecs:c_t_r"
    assert command[command.index("--document-name") + 1] == "AWS-StartPortForwardingSession"
    assert json.loads(command[command.index("--parameters") + 1]) == {
        "portNumber": ["8265"], "localPortNumber": ["8265"]}


def test_no_tunnel_is_opened_without_the_session_manager_plugin():
    m = _driver()
    opened = []
    tunnel = m.open_dashboard_tunnel("ecs:c_t_r", 8265, which=lambda name: None,
                                     popen=lambda *a, **k: opened.append(a))
    assert tunnel is None and not opened
