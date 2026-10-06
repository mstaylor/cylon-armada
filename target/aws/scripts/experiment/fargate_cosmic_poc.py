#!/usr/bin/env python3
"""Cosmic AI agentic pipeline on ECS Fargate (Experiment E proof of concept).

Launches one Fargate task per rank, each running one rank of the compiled
six-operator workflow (Preprocess | Embed | Retrieve | Reason | Bind |
MemoryUpsert) over the FMI direct-redis channel. Every rank runs AstroMAE over
its own contiguous shard of the SDSS partition first, so the workflow executes
with InputPlacement.PreDistributed and Preprocess's Scatter is skipped rather
than moving data that is already in place.

Modelled on fargate_expb_sweep.py — same cluster, subnets, and per-rank
run_task pattern.

Usage:
    python fargate_cosmic_poc.py --world-sizes 1 2 --dry-run
    python fargate_cosmic_poc.py --backend ray-comparison --dry-run
    python fargate_cosmic_poc.py --scaling weak --runs 4 --live
    python fargate_cosmic_poc.py --scaling strong --backend armada --world-sizes 4 --live

Every arm runs on one task definition and one image, which carries Ray, so
image size and pull time cannot differ between arms. Every rank is one task:
under the Ray arms rank 0 starts the Ray head and publishes its address
through Redis under COMM_NAME, the others discover it and join.

Each world size first runs --warmup-runs unmeasured runs, written under
warmup<k>/ and never aggregated, then --runs measured runs. A Fargate task is
always a fresh microVM, so a warm-up cannot warm a container; what it warms is
everything the tasks share (Redis, Bedrock, the S3 artifacts, the ECR
registry), so the first measured run is not also the first touch of those.
--dry-run prints the full matrix and an estimate of the task-hours it costs.

Each world size runs `--runs` measured paired runs; within a run the arms execute back
to back, never concurrently, and their order cycles through the permutations
of the arms being compared, so that Bedrock latency drift over the session
cannot masquerade as an effect. Two arms alternate; five lead once each over
five runs; three arms over four runs leave one extra lead, which falls on
ray-cylon. Inside a comparison ray-cylon runs only at --ray-cylon-world-sizes:
its data plane is Armada's, so it is a portability check, not a sweep arm.

Prerequisites:
    - cylon-armada-python image built with the AstroMAE deps and pushed to ECR
    - model + data uploaded to s3://<bucket>/<MODEL_KEY|DATA_KEY> (see --help)
"""

import argparse
import functools
import itertools
import json
import logging
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REGION = "us-east-1"
CLUSTER = "CylonFargateExperiments"
# Must match what terraform deploys: aws_ecs_task_definition.cosmic_armada is
# family "${project_name}-cosmic" (see the cosmic_task_definition_family output)
# and its container is var.ecs_container_name. Its own family, separate from
# "-python", because run_task cannot override a container image and only these
# tasks should carry the PyTorch layer. The standalone
# target/aws/scripts/ecs/task_definition_fargate.json is not a deployed
# definition — nothing reads it.
TASK_DEFINITION = "cylon-armada-cosmic"
RAY_SECURITY_GROUP_NAME = "cylon-armada-ray-tasks"
CONTAINER_NAME = "cylon-armada"
SUBNETS = ["subnet-07995eea6c462cd73", "subnet-0979c94513025746c"]
REDIS_ADDR = "dev-cylon-redis1.aws-cylondata.com:6379"
RESULTS_BUCKET = "staylor.dev2"
RESULTS_PREFIX = "cylon-armada/results/exp_e_cosmic"
# What the driver pays to notice an arm finished. See _wait_for_tasks.
WAITER_DELAY_S = 5
# ARTIFACT_BUCKET / ASTROMAE_* keys and paths, INFERENCE_DEVICE and
# INFERENCE_BATCH_SIZE all come from the task definition terraform deploys
# (local.ecs_env); only device and batch size are overridden here, and only
# when the caller asks for something other than the deployed default.

# Runs inside the task. Pulls the weights and the SDSS partition from S3 rather
# than baking ~90MB into the image, matching how the shared scripts are already
# hot-reloaded. cwd must be the scripts root: with armada/ as the working
# directory, armada/operator.py shadows the stdlib `operator` module and breaks
# almost every import.
RUNNER_SCRIPT = """import os, subprocess, sys, boto3
scripts = '/cylon-armada/scripts'
s3 = boto3.client('s3')
bucket = os.environ['ARTIFACT_BUCKET']
for key, dest in ((os.environ['ASTROMAE_MODEL_KEY'], os.environ['ASTROMAE_MODEL_PATH']),
                  (os.environ['ASTROMAE_DATA_KEY'], os.environ['ASTROMAE_DATA_PATH'])):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print('downloading', key, '->', dest, flush=True)
        s3.download_file(bucket, key, dest)
cmd = [sys.executable, '-m', 'armada.run_cosmic_local',
       '--real-inference', '--result-path', '/tmp/result.json']
if os.environ.get('GALAXIES'):
    cmd += ['--galaxies', os.environ['GALAXIES']]
if os.environ.get('LIVE') == '1':
    cmd += ['--live']
env = dict(os.environ, PYTHONPATH=scripts + ':' + os.environ.get('PYTHONPATH', ''))
result = subprocess.run(cmd, cwd=scripts, env=env)
if os.path.exists('/tmp/result.json'):
    s3.upload_file('/tmp/result.json', os.environ['RESULTS_BUCKET'],
                   os.environ['S3_RESULTS_PREFIX'] + 'rank%s.json' % os.environ['RANK'])
    print('uploaded result for rank', os.environ['RANK'], flush=True)
sys.exit(result.returncode)
"""


def galaxies_for(scaling, world_size, per_rank, total):
    """Population for this point.

    Weak scaling holds the per-rank load constant so the coordination cost per
    unit of work is what varies with N; strong scaling holds the total constant.
    Weak is capped at the real population because repeating galaxies would make
    them trivial cache hits and inflate the reuse rate by construction.
    """
    if scaling == "strong":
        return total
    return min(per_rank * world_size, total)


ARMS = ("armada", "langchain", "isolated", "ray-native", "ray-cylon")

ARMS_WITHOUT_EXECUTOR = ()
RAY_ARMS = ("ray-native", "ray-cylon")
LAUNCHABLE_ARMS = tuple(arm for arm in ARMS if arm not in ARMS_WITHOUT_EXECUTOR)

_ORDERS = tuple(itertools.permutations(ARMS))


def _order_stride(n_arms, n_orders):
    """Calculate a stride through permutations that cycles through different leaders.

    With n arms there are n! permutations. A naive modulo would cycle through them
    consecutively, but the first n_arms permutations all start with the same leader,
    so a short sweep would have one arm always leading. A stride skips by at least
    one block (n_orders // n_arms) and is coprime with n_orders, which guarantees
    the full cycle visits every permutation and so gives every arm an equal number
    of leads over n! runs.

    Distinct leaders over the first n runs is NOT guaranteed by that construction
    and must not be assumed for an arbitrary arm count. It holds at the current
    five arms (stride 29, first five runs led by all five arms) and is pinned by
    test_leader_diversity_over_short_runs; it is false at three arms, where
    stride 5 leads with arms 0, 2, 2. Anyone changing ARMS must re-check that
    test rather than trusting this function to generalise.
    """
    block = n_orders // n_arms
    stride = block
    while math.gcd(stride, n_orders) != 1:
        stride += 1
    return stride


_ORDER_STRIDE = _order_stride(len(ARMS), len(_ORDERS))


@functools.lru_cache(maxsize=None)
def _orders_for(arms):
    orders = tuple(itertools.permutations(arms))
    return orders, _order_stride(len(arms), len(orders))


def order_for(arms, run_index):
    """The order the given arms run in, cycling through their own permutations.

    Ordering over only the arms being compared keeps the leader balanced for
    that selection: filtering the five-arm cycle down to two arms let one arm
    lead three of four runs. Over all five arms this is exactly arm_order.
    """
    orders, stride = _orders_for(tuple(arms))
    return orders[(run_index * stride) % len(orders)]


def arm_order(run_index):
    """The arms in the order they run, cycling through every permutation.

    Rotation alone is not enough: it produces one fixed cyclic sequence, so certain
    arms would always follow the same arm. Both the store and collectives touch
    shared state, and `isolated` supplies the headline isolation-penalty number,
    so any warm-up or leftover-state bias from the preceding arm would land on it
    in one constant undetected direction. Permuting varies which arm precedes which.
    The stride spreads the leader across runs; see _order_stride for exactly what
    that does and does not guarantee.
    """
    return _ORDERS[(run_index * _ORDER_STRIDE) % len(_ORDERS)]


_ARM_SELECTIONS = {
    "all": LAUNCHABLE_ARMS,
    "both": ("armada", "langchain"),
    "ray-comparison": ("armada", "ray-native", "ray-cylon"),
}


def resolve_ray_security_group(ec2):
    """The id of the Ray tasks' security group terraform created, by name.

    Without it a Ray task lands in the VPC default group, which here admits
    all TCP from anywhere, and Ray's GCS and client ports have no
    authentication. Refusing to launch is cheaper than that exposure.
    """
    groups = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [RAY_SECURITY_GROUP_NAME]}])["SecurityGroups"]
    if len(groups) != 1:
        raise RuntimeError(
            f"expected exactly one security group named {RAY_SECURITY_GROUP_NAME!r}, found "
            f"{len(groups)}; terraform apply target/aws/scripts/terraform creates it")
    return groups[0]["GroupId"]


def security_groups_for(arm, ray_security_groups):
    """Ray arms get the members-only Ray group; every other arm keeps the
    default its results so far were measured under."""
    return ray_security_groups if arm in RAY_ARMS else None


def _selected_arms(backend):
    """Which arms a --backend value launches.

    "both" keeps meaning exactly the two sharing arms, as it always has.
    Folding it into "all" would silently add a third arm — 50% more Fargate and
    Bedrock than the name implies — to every caller and script that already
    passes it. "ray-comparison" is the headline comparison: Armada against
    both Ray arms.

    An arm listed in ARMS_WITHOUT_EXECUTOR is refused here rather than at the
    far end of a sweep, where it would cost a Fargate task per rank and yield
    no measurement. None is listed today.
    """
    selected = _ARM_SELECTIONS.get(backend, (backend,))
    pending = tuple(arm for arm in selected if arm in ARMS_WITHOUT_EXECUTOR)
    if pending:
        raise ValueError(
            f"arm(s) {list(pending)} have no executor and cannot be launched; "
            f"launchable arms are {list(LAUNCHABLE_ARMS)}"
        )
    return selected


def arms_at_world_size(backend, world_size, ray_cylon_world_sizes):
    """The arms a --backend value launches at one world size.

    ray-cylon's data plane is Armada's FMI bridge, so inside a comparison it
    runs only at ray_cylon_world_sizes, as a portability check rather than a
    sweep arm. Naming ray-cylon alone is an explicit request and is not trimmed.
    """
    selected = _selected_arms(backend)
    if len(selected) > 1 and world_size not in ray_cylon_world_sizes:
        selected = tuple(arm for arm in selected if arm != "ray-cylon")
    return selected


def plan_warmups(backend, warmup_runs, arms=None):
    """The ordered (warmup_index, backend) sequence run before the measured runs.

    Ordered exactly as measured runs are, so the warm-ups touch the shared
    services in the same arm order a measured run would.
    """
    return plan_launches(backend, warmup_runs, arms)


def plan_launches(backend, runs, arms=None):
    """The ordered (run_index, backend) sequence for one world size.

    Pure, so ordering and grouping are testable without AWS. Each run orders
    the launched arms by order_for over those arms alone; arms defaults to
    everything the backend selects.
    """
    selected = arms if arms is not None else _selected_arms(backend)
    return [(run_index, arm) for run_index in range(runs) for arm in order_for(selected, run_index)]


_CONTEXT_STORE = {"armada": "cylon", "langchain": "redis", "isolated": "cylon",
                  "ray-native": "plasma", "ray-cylon": "cylon"}


def build_overrides(rank, world_size, comm_name, s3_prefix, args, backend):
    environment = [
        {"name": "RANK", "value": str(rank)},
        {"name": "WORLD_SIZE", "value": str(world_size)},
        # Unique per run: Redis INCR assigns ranks under this name, so a reused
        # name makes concurrent or repeated runs collide on rank assignment.
        {"name": "COMM_NAME", "value": comm_name},
        {"name": "REDIS_HOST", "value": REDIS_ADDR.split(":")[0]},
        {"name": "REDIS_PORT", "value": REDIS_ADDR.split(":")[1]},
        {"name": "FMI_LISTEN_PORT", "value": str(args.listen_port)},
        {"name": "FMI_MAX_TIMEOUT", "value": str(args.timeout_ms)},
        {"name": "RESULTS_BUCKET", "value": RESULTS_BUCKET},
        {"name": "S3_RESULTS_PREFIX", "value": s3_prefix},
        {"name": "EXECUTION_BACKEND", "value": backend},
        {"name": "CONTEXT_BACKEND", "value": _CONTEXT_STORE[backend]},
        {"name": "CONTEXT_TABLE_SNAPSHOT", "value": "0"},
        {"name": "REUSE_KEY_TOLERANCE", "value": str(args.reuse_tolerance)},
        {"name": "OUTLIER_RESIDUAL_THRESHOLD", "value": str(args.outlier_threshold)},
        {"name": "EPOCH_BATCH_SIZE", "value": str(args.epoch_batch_size)},
        {"name": "INFERENCE_DEVICE", "value": args.device},
        {"name": "INFERENCE_BATCH_SIZE", "value": str(args.batch_size)},
        {"name": "LIVE", "value": "1" if args.live else "0"},
        {"name": "GALAXIES", "value": str(args.galaxies)},
    ]
    return {
        "containerOverrides": [
            {
                "name": CONTAINER_NAME,
                "command": ["python", "-c", RUNNER_SCRIPT],
                "environment": environment,
            }
        ]
    }


# run_task failure reason AWS returns when the AZ/instance family it picked has
# no Fargate capacity right now — transient, and retrying (often into the other
# subnet's AZ) succeeds within minutes. Any other failure reason (bad task
# definition, IAM, etc.) is not retried.
CAPACITY_FAILURE_REASON = "Capacity is unavailable"
CAPACITY_RETRY_MAX_DELAY_S = 300

CAPACITY_RETRYABLE_ERROR_CODES = (
    "ThrottlingException",
    "Throttling",
    "RequestLimitExceeded",
    "TooManyRequestsException",
    "ServiceUnavailable",
)

CAPACITY_BUDGET_FRACTION = 0.5


def capacity_budget_s(timeout_ms, fraction=CAPACITY_BUDGET_FRACTION):
    """How long one rank may spend retrying before its cohort is already dead.

    Every rank launched earlier is blocked inside FMI communicator construction
    with FMI_MAX_TIMEOUT = timeout_ms. A rank that retries past that point
    launches into a cohort whose peers have already given up: it pairs with
    nobody, and the driver then waits out the full arm timeout on tasks that
    cannot produce a measurement. Bounding retries at a fraction of the peers'
    own timeout keeps the failure fast and loud instead of slow and silent.
    """
    return (timeout_ms / 1000.0) * fraction


def _capacity_retryable(failures):
    return bool(failures) and all(
        CAPACITY_FAILURE_REASON in f.get("reason", "")
        or f.get("reason", "") in CAPACITY_RETRYABLE_ERROR_CODES
        for f in failures
    )


def _run_tasks(ecs, world_size, overrides_for_rank, capacity_retries,
               capacity_backoff_s, budget_s, task_definition=TASK_DEFINITION,
               security_groups=None):
    vpc_configuration = {"subnets": SUBNETS, "assignPublicIp": "ENABLED"}
    if security_groups:
        vpc_configuration["securityGroups"] = list(security_groups)

    def _launch(rank):
        attempt = 0
        deadline = time.monotonic() + budget_s
        while True:
            try:
                resp = ecs.run_task(
                    cluster=CLUSTER,
                    taskDefinition=task_definition,
                    launchType="FARGATE",
                    networkConfiguration={"awsvpcConfiguration": vpc_configuration},
                    overrides=overrides_for_rank(rank),
                )
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in CAPACITY_RETRYABLE_ERROR_CODES:
                    raise
                failures = [{"reason": code}]
            else:
                failures = resp.get("failures")
                if not failures:
                    return resp["tasks"][0]["taskArn"]
            if not _capacity_retryable(failures) or attempt >= capacity_retries:
                raise RuntimeError(f"rank {rank} run_task failed: {failures}")
            delay = min(capacity_backoff_s * (2 ** attempt), CAPACITY_RETRY_MAX_DELAY_S)
            if time.monotonic() + delay > deadline:
                raise RuntimeError(
                    f"rank {rank} still has no Fargate capacity with only "
                    f"{deadline - time.monotonic():.0f}s of its {budget_s:.0f}s budget "
                    f"left; abandoning the arm rather than launching into a cohort "
                    f"whose peers' FMI communicators have already timed out"
                )
            logger.warning("  rank %d hit Fargate capacity limits (attempt %d/%d), retrying in %ds",
                            rank, attempt + 1, capacity_retries, delay)
            time.sleep(delay)
            attempt += 1

    # Ranks must come up together — the channel establishes connections during
    # communicator construction, so a straggler stalls every peer it pairs with.
    with ThreadPoolExecutor(max_workers=min(world_size, 16)) as pool:
        return list(pool.map(_launch, range(world_size)))


def _wait_for_tasks(ecs, arns, timeout_s):
    """Block until every task in this arm has stopped.

    The paired arm must not start until this one is finished: two arms running
    concurrently would share Redis and Bedrock quota and throttle each other,
    which is exactly the confound the throttle gate exists to catch.

    The poll delay is what the driver pays to notice an arm has ended. At 15s it
    cost about 7s per arm-run on average, which is real time across a sweep of
    105 arm-runs and is also quantization noise on the startup measurement
    task_timing takes. MaxAttempts is derived from it, so the wall-clock timeout
    is unchanged.
    """
    waiter = ecs.get_waiter("tasks_stopped")
    delay = WAITER_DELAY_S
    waiter.wait(cluster=CLUSTER, tasks=arns,
                WaiterConfig={"Delay": delay, "MaxAttempts": max(1, timeout_s // delay)})


def task_timing(task):
    """Decompose one ECS task's lifecycle into the phases a sweep pays for.

    Fargate gives each task a fresh microVM with no shared layer cache, so the
    image is pulled per task. Separating pull from the rest is the whole point:
    it is the term that infrastructure changes can move, and the one that
    guessing gets wrong.

    Any timestamp may be absent. ECS omits pullStartedAt when a task never got
    far enough to pull, and a failed arm must still return its numbers rather
    than raise on the way out.
    """
    def span(a, b):
        first, second = task.get(a), task.get(b)
        if first is None or second is None:
            return None
        return round((second - first).total_seconds(), 3)

    return {
        "provision_s": span("createdAt", "pullStartedAt"),
        "pull_s": span("pullStartedAt", "pullStoppedAt"),
        "run_s": span("startedAt", "stoppingAt"),
        "teardown_s": span("stoppingAt", "stoppedAt"),
        "total_s": span("createdAt", "stoppedAt"),
    }


def summarize_timings(timings):
    """Reduce per-rank timings to what the arm actually waited for.

    An arm ends when its slowest rank ends, so the max is the quantity that
    sets sweep wall clock. A mean would understate it, and understating startup
    is how a sweep gets planned at a quarter of its real duration.
    """
    out = {"ranks": len(timings)}
    for phase in ("provision_s", "pull_s", "run_s", "teardown_s", "total_s"):
        values = [t[phase] for t in timings if t.get(phase) is not None]
        out[f"{phase}_max"] = max(values) if values else None
    return out


def collect_arm_timing(ecs, arns):
    """Per-rank lifecycle timings for one arm, or None if ECS has aged them out.

    Stopped tasks are retained for about an hour, so this has to run right after
    the waiter returns. It never raises: a timing record is diagnostic, and
    losing it must not cost the run's results.
    """
    try:
        described = []
        for i in range(0, len(arns), 100):
            resp = ecs.describe_tasks(cluster=CLUSTER, tasks=arns[i:i + 100])
            described.extend(resp.get("tasks", []))
        if not described:
            return None
        timings = [task_timing(task) for task in described]
        summary = summarize_timings(timings)
        summary["per_rank"] = timings
        return summary
    except Exception as exc:
        logger.warning("could not collect task timings: %s", exc)
        return None


def sweep_matrix(args):
    """Every launch the sweep makes, in order, one row per arm-run.

    Pure, so the dry run and the live run walk the same rows. Warm-ups at a
    world size precede its measured runs and are written under warmup<k>/.
    """
    rows = []
    for world_size in sorted(args.world_sizes):
        galaxies = galaxies_for(args.scaling, world_size, args.per_rank, args.total)
        arms = arms_at_world_size(args.backend, world_size, args.ray_cylon_world_sizes)
        phases = (("warmup", plan_warmups(args.backend, args.warmup_runs, arms)),
                  ("measured", plan_launches(args.backend, args.runs, arms)))
        for phase, launches in phases:
            run_dir = "warmup" if phase == "warmup" else "run"
            for run_index, arm in launches:
                rows.append({
                    "world_size": world_size, "galaxies": galaxies, "phase": phase,
                    "run": run_index, "arm": arm, "tasks": world_size,
                    "task_definition": TASK_DEFINITION,
                    "security_group": (RAY_SECURITY_GROUP_NAME if arm in RAY_ARMS
                                       else "vpc default"),
                    "s3_prefix": (f"{RESULTS_PREFIX}/{args.scaling}/{arm}/ws{world_size}/"
                                  f"{run_dir}{run_index}/"),
                })
    return rows


def estimate_usage(rows, task_minutes, task_vcpu, task_memory_gb):
    """Task-hours for a set of launches, assuming every task runs task_minutes.

    A planning figure only: refine task_minutes from a smoke run's
    _timing.json (total_s_max is the slowest task's whole lifecycle).
    """
    tasks = sum(row["tasks"] for row in rows)
    task_hours = tasks * task_minutes / 60.0
    return {"launches": len(rows), "tasks": tasks, "task_hours": round(task_hours, 2),
            "vcpu_hours": round(task_hours * task_vcpu, 2),
            "gb_hours": round(task_hours * task_memory_gb, 2)}


def estimate_cost_usd(usage, vcpu_hour_usd, gb_hour_usd):
    """Fargate compute cost of an estimate_usage result; Bedrock, Redis,
    ECR and S3 are not included."""
    return round(usage["vcpu_hours"] * vcpu_hour_usd + usage["gb_hours"] * gb_hour_usd, 2)


def print_dry_run(args):
    """Print the full sweep matrix and its estimated cost, launching nothing."""
    rows = sweep_matrix(args)

    def usage(subset):
        return estimate_usage(subset, args.est_task_minutes, args.task_vcpu,
                              args.task_memory_gb)

    print(f"Sweep matrix: scaling={args.scaling} backend={args.backend} "
          f"warm-up runs={args.warmup_runs} measured runs={args.runs}")
    def cost(cell):
        return estimate_cost_usd(cell, args.vcpu_hour_usd, args.gb_hour_usd)

    print(f"{'N':>4} {'galaxies':>8} {'phase':>8} {'run':>3}  {'arm':<11} "
          f"{'tasks':>5}  {'task definition':<20} security group")
    for row in rows:
        print(f"{row['world_size']:>4} {row['galaxies']:>8} {row['phase']:>8} {row['run']:>3}  "
              f"{row['arm']:<11} {row['tasks']:>5}  {row['task_definition']:<20} "
              f"{row['security_group']}")
    print()
    print(f"{'N':>4} {'launches':>8} {'tasks':>6} {'task-hours':>11} {'USD':>9}")
    for world_size in sorted(args.world_sizes):
        cell = usage([r for r in rows if r["world_size"] == world_size])
        print(f"{world_size:>4} {cell['launches']:>8} {cell['tasks']:>6} "
              f"{cell['task_hours']:>11} {cost(cell):>9.2f}")
    total = usage(rows)
    warmup = usage([r for r in rows if r["phase"] == "warmup"])
    print()
    print(f"Total: {total['launches']} launches, {total['tasks']} tasks, "
          f"{total['task_hours']} task-hours ({total['vcpu_hours']} vCPU-hours, "
          f"{total['gb_hours']} GB-hours) at {args.est_task_minutes} min/task, "
          f"{args.task_vcpu} vCPU / {args.task_memory_gb} GB per task; "
          f"warm-ups account for {warmup['task_hours']} task-hours")
    print(f"Estimated Fargate cost: ${cost(total):.2f} (warm-ups ${cost(warmup):.2f}) at "
          f"${args.vcpu_hour_usd}/vCPU-hour and ${args.gb_hour_usd}/GB-hour; excludes Bedrock, "
          f"Redis, ECR and S3")
    return rows


def launch_world_size(ecs, world_size, args, ray_security_groups=None):
    rows = [row for row in sweep_matrix(args) if row["world_size"] == world_size]
    galaxies = galaxies_for(args.scaling, world_size, args.per_rank, args.total)
    point_args = argparse.Namespace(**{**vars(args), "galaxies": galaxies})
    logger.info("world_size=%d scaling=%s galaxies=%d launches=%s",
                world_size, args.scaling, galaxies,
                [f"{row['phase']}{row['run']}:{row['arm']}" for row in rows])

    all_arns = []
    for row in rows:
        arm, run_index, s3_prefix = row["arm"], row["run"], row["s3_prefix"]
        phase_tag = "w" if row["phase"] == "warmup" else ""
        comm_name = (f"cosmic_{args.scaling}_{arm}_{world_size}_{phase_tag}{run_index}_"
                     f"{uuid.uuid4().hex[:8]}")
        logger.info("  %s %d arm %s comm_name=%s -> s3://%s/%s",
                    row["phase"], run_index, arm, comm_name, RESULTS_BUCKET, s3_prefix)
        if args.dry_run:
            overrides = build_overrides(0, world_size, comm_name, s3_prefix, point_args, arm)
            logger.info("  [dry-run] rank 0 task_definition=%s env=%s", row["task_definition"],
                        {e["name"]: e["value"] for e in
                         overrides["containerOverrides"][0]["environment"]})
            continue
        arns = _run_tasks(ecs, world_size, lambda rank: build_overrides(
            rank, world_size, comm_name, s3_prefix, point_args, arm),
            args.capacity_retries, args.capacity_backoff_s,
            capacity_budget_s(args.timeout_ms), task_definition=row["task_definition"],
            security_groups=security_groups_for(arm, ray_security_groups))
        logger.info("  launched %d tasks; waiting for the arm to finish", len(arns))
        _wait_for_tasks(ecs, arns, args.arm_timeout_s)
        _record_arm_timing(ecs, arns, s3_prefix)
        all_arns.extend(arns)
    return all_arns


def _record_arm_timing(ecs, arns, s3_prefix):
    """Write this arm's startup breakdown beside its results.

    Runs immediately after the waiter because ECS retains stopped tasks for
    about an hour. Never raises: this is diagnostic, and losing it must not
    cost the results the arm just produced.
    """
    summary = collect_arm_timing(ecs, arns)
    if summary is None:
        return
    logger.info("  startup: pull %ss, provision %ss, teardown %ss, total %ss (slowest rank)",
                summary.get("pull_s_max"), summary.get("provision_s_max"),
                summary.get("teardown_s_max"), summary.get("total_s_max"))
    try:
        boto3.client("s3", region_name=REGION).put_object(
            Bucket=RESULTS_BUCKET, Key=s3_prefix + "_timing.json",
            Body=json.dumps(summary, indent=1).encode())
    except Exception as exc:
        logger.warning("could not write timing record: %s", exc)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-sizes", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64],
                        help="must include 1 and 2 as baselines")
    parser.add_argument("--scaling", choices=["weak", "strong"], default="strong",
                        help="weak (fixed per-rank load, primary) or strong (fixed total)")
    parser.add_argument("--reuse-tolerance", type=float, default=0.0091,
                        help="redshift tolerance for the reuse validity gate; identical on "
                             "every arm or the policy becomes the difference being measured")
    parser.add_argument("--outlier-threshold", type=float, default=0.027677,
                        help="residual above which a galaxy gets the outlier prompt, pinned "
                             "from the whole population (p90 over 1253 SDSS galaxies). Unpinned, "
                             "the generator derives it per shard and the workload moves with N")
    parser.add_argument("--backend", choices=list(ARMS) + list(_ARM_SELECTIONS),
                        default="all",
                        help="which arm(s); 'ray-comparison' is armada against both Ray arms")
    parser.add_argument("--runs", type=int, default=4,
                        help="measured paired runs per world size; arm order alternates "
                             "between runs; mean and sample std are taken across them")
    parser.add_argument("--ray-cylon-world-sizes", type=int, nargs="+", default=[2, 8],
                        help="world sizes at which ray-cylon runs inside a comparison; its data "
                             "plane is Armada's, so it is a portability check, not a sweep arm")
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="unmeasured runs per world size before the measured ones, "
                             "written under warmup<k>/ and never aggregated")
    parser.add_argument("--est-task-minutes", type=float, default=10.0,
                        help="dry-run planning estimate of one task's whole lifecycle; refine "
                             "from a smoke run's _timing.json total_s_max")
    parser.add_argument("--task-vcpu", type=float, default=4.0,
                        help="dry-run estimate only; must match terraform ecs_python_cpu / 1024")
    parser.add_argument("--task-memory-gb", type=float, default=8.0,
                        help="dry-run estimate only; must match terraform ecs_python_memory_mb / 1024")
    parser.add_argument("--vcpu-hour-usd", type=float, default=0.04048,
                        help="dry-run estimate only; Fargate Linux/x86 on-demand vCPU-hour price")
    parser.add_argument("--gb-hour-usd", type=float, default=0.004445,
                        help="dry-run estimate only; Fargate Linux/x86 on-demand GB-hour price")
    parser.add_argument("--per-rank", type=int, default=19,
                        help="galaxies per rank under weak scaling (floor(1253/64))")
    parser.add_argument("--total", type=int, default=1253,
                        help="galaxies in the partition; caps weak scaling, is strong scaling")
    parser.add_argument("--epoch-batch-size", type=int, default=4,
                        help="galaxies per epoch inside each rank; identical across arms")
    parser.add_argument("--device", default="cpu",
                        help="torch device inside the task; Fargate has no GPU")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="AstroMAE inference batch size")
    parser.add_argument("--listen-port", type=int, default=10000)
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--arm-timeout-s", type=int, default=1800,
                        help="how long to wait for one arm's tasks to stop before the next")
    parser.add_argument("--capacity-retries", type=int, default=6,
                        help="per-rank retries on a transient Fargate 'Capacity is "
                             "unavailable' run_task failure before giving up")
    parser.add_argument("--capacity-backoff-s", type=int, default=30,
                        help="base delay for exponential backoff between capacity "
                             f"retries, capped at {CAPACITY_RETRY_MAX_DELAY_S}s")
    parser.add_argument("--live", action="store_true",
                        help="real Bedrock on every rank — costs money per galaxy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--settle-s", type=int, default=30,
                        help="pause between world sizes so tasks drain")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.device != "cpu":
        parser.error("Fargate has no GPU support — use the ECS EC2 GPU arm for cuda")
    if 1 not in args.world_sizes or 2 not in args.world_sizes:
        logger.warning("world sizes %s omit 1 and/or 2 — the sweep convention requires both "
                       "as baseline points", args.world_sizes)

    if args.dry_run:
        print_dry_run(args)

    ray_security_groups = None
    if not args.dry_run and any(arm in RAY_ARMS for arm in _selected_arms(args.backend)):
        ray_security_groups = [resolve_ray_security_group(boto3.client("ec2", region_name=REGION))]
        logger.info("Ray arms launch in security group %s (%s)",
                    ray_security_groups[0], RAY_SECURITY_GROUP_NAME)

    ecs = boto3.client("ecs", region_name=REGION)
    for i, world_size in enumerate(sorted(args.world_sizes)):
        launch_world_size(ecs, world_size, args, ray_security_groups)
        if not args.dry_run and i < len(args.world_sizes) - 1:
            time.sleep(args.settle_s)

    if args.dry_run:
        logger.info("dry run complete — nothing launched")
    else:
        logger.info("results land under s3://%s/%s", RESULTS_BUCKET, RESULTS_PREFIX)


if __name__ == "__main__":
    main()
