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
    python fargate_cosmic_poc.py --scaling weak --runs 5 --live
    python fargate_cosmic_poc.py --scaling strong --backend armada --world-sizes 4 --live

Each world size runs `--runs` paired runs; within a run the two arms execute
back to back, never concurrently, and their order alternates between runs so
that Bedrock latency drift over the session cannot masquerade as an effect.

Prerequisites:
    - cylon-armada-python image built with the AstroMAE deps and pushed to ECR
    - model + data uploaded to s3://<bucket>/<MODEL_KEY|DATA_KEY> (see --help)
"""

import argparse
import itertools
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import boto3

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


ARMS = ("armada", "langchain", "isolated")
_ORDERS = tuple(itertools.permutations(ARMS))


def arm_order(run_index):
    """The arms in the order they run, cycling through every permutation.

    Rotation alone is not enough with three arms: it produces one fixed cyclic
    sequence, so `isolated` would always immediately follow `langchain` and
    never `armada`. Both of those touch Redis and ContextTable state, and
    `isolated` supplies the headline isolation-penalty number, so any warm-up or
    leftover-state bias from the preceding arm would land on it in one constant
    undetected direction. Permuting varies which arm precedes which, not only
    which leads.
    """
    return _ORDERS[run_index % len(_ORDERS)]


_ARM_SELECTIONS = {
    "all": ARMS,
    "both": ("armada", "langchain"),
}


def _selected_arms(backend):
    """Which arms a --backend value launches.

    "both" keeps meaning exactly the two sharing arms, as it always has.
    Folding it into "all" would silently add a third arm — 50% more Fargate and
    Bedrock than the name implies — to every caller and script that already
    passes it.
    """
    return _ARM_SELECTIONS.get(backend, (backend,))


def plan_launches(backend, runs):
    """The ordered (run_index, backend) sequence for one world size.

    Pure, so ordering and grouping are testable without AWS. "all" yields every
    arm per run in arm_order; "both" the two sharing arms; a single backend
    yields one launch per run in the same slot it would have had.
    """
    launches = []
    for run_index in range(runs):
        for arm in arm_order(run_index):
            if arm in _selected_arms(backend):
                launches.append((run_index, arm))
    return launches


_CONTEXT_STORE = {"armada": "cylon", "langchain": "redis", "isolated": "cylon"}


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


def _run_tasks(ecs, world_size, overrides_for_rank):
    def _launch(rank):
        resp = ecs.run_task(
            cluster=CLUSTER,
            taskDefinition=TASK_DEFINITION,
            launchType="FARGATE",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": SUBNETS,
                    "assignPublicIp": "ENABLED",
                }
            },
            overrides=overrides_for_rank(rank),
        )
        if resp.get("failures"):
            raise RuntimeError(f"rank {rank} run_task failed: {resp['failures']}")
        return resp["tasks"][0]["taskArn"]

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


def launch_world_size(ecs, world_size, args):
    galaxies = galaxies_for(args.scaling, world_size, args.per_rank, args.total)
    point_args = argparse.Namespace(**{**vars(args), "galaxies": galaxies})
    launches = plan_launches(args.backend, args.runs)
    logger.info("world_size=%d scaling=%s galaxies=%d launches=%s",
                world_size, args.scaling, galaxies,
                [f"run{r}:{arm}" for r, arm in launches])

    all_arns = []
    for run_index, arm in launches:
        comm_name = f"cosmic_{args.scaling}_{arm}_{world_size}_{run_index}_{uuid.uuid4().hex[:8]}"
        s3_prefix = f"{RESULTS_PREFIX}/{args.scaling}/{arm}/ws{world_size}/run{run_index}/"
        logger.info("  run %d arm %s comm_name=%s -> s3://%s/%s",
                    run_index, arm, comm_name, RESULTS_BUCKET, s3_prefix)
        if args.dry_run:
            overrides = build_overrides(0, world_size, comm_name, s3_prefix, point_args, arm)
            logger.info("  [dry-run] rank 0 env=%s",
                        {e["name"]: e["value"] for e in
                         overrides["containerOverrides"][0]["environment"]})
            continue
        arns = _run_tasks(ecs, world_size, lambda rank: build_overrides(
            rank, world_size, comm_name, s3_prefix, point_args, arm))
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
    parser.add_argument("--backend", choices=["armada", "langchain", "isolated", "all", "both"],
                        default="all",
                        help="which arm(s); the experiment is a paired run, so both by default")
    parser.add_argument("--runs", type=int, default=5,
                        help="paired runs per world size; arm order alternates between runs")
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

    ecs = boto3.client("ecs", region_name=REGION)
    for i, world_size in enumerate(sorted(args.world_sizes)):
        launch_world_size(ecs, world_size, args)
        if not args.dry_run and i < len(args.world_sizes) - 1:
            time.sleep(args.settle_s)

    if args.dry_run:
        logger.info("dry run complete — nothing launched")
    else:
        logger.info("results land under s3://%s/%s", RESULTS_BUCKET, RESULTS_PREFIX)


if __name__ == "__main__":
    main()
