#!/usr/bin/env python3
"""Phase 1 cloud sweep — fires Step Functions per (arch, scenario, world_size, run).

Each execution is fully isolated: unique workflow_id, unique experiment_name,
results written to s3://staylor.dev2/results/{arch}/{scenario}/. Failures on
any single execution do not affect others.

Usage:
    # Single architecture, all scenarios, 4 runs
    python cloud_sweep.py --arch lambda-python --scenario all --runs 4

    # Specific scenario, vary world_size
    python cloud_sweep.py --arch ecs-fargate --scenario hydrology \
        --world-sizes 1 2 4 --runs 4

    # Dry-run — print configs without firing
    python cloud_sweep.py --arch all --scenario all --runs 4 --dry-run

    # All architectures sequentially
    python cloud_sweep.py --arch all --scenario all --runs 4
"""

import argparse
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCOUNT_ID   = "448324707516"
REGION       = "us-east-1"
RESULTS_BUCKET = "staylor.dev2"
S3_SCRIPTS_BUCKET = "staylor.dev2"
S3_SCRIPTS_PREFIX = "cylon-armada/scripts/"

WORKFLOW_ARNS = {
    "lambda-python": f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:cylon-armada-python-workflow",
    "lambda-nodejs": f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:cylon-armada-nodejs-workflow",
    "ecs-fargate":   f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:cylon-armada-ecs-fargate-workflow",
    "ecs-ec2-cpu":   f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:cylon-armada-ecs-ec2-cpu-workflow",
    "ecs-ec2-gpu":   f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:cylon-armada-ecs-ec2-gpu-workflow",
}

ALL_ARCHITECTURES = list(WORKFLOW_ARNS.keys())
ALL_SCENARIOS     = ["hydrology", "epidemiology", "seismology", "mixed_scientific"]

# Experiment A / A2 zero-copy benchmark: a single-node state machine with a
# different input schema (no scenarios / world_size / scaling). Kept separate
# from WORKFLOW_ARNS so `--arch all` reuse sweeps never fire it by accident;
# reached explicitly via `--arch benchmark`.
BENCHMARK_WORKFLOW_ARN = (
    f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:cylon-armada-benchmark-workflow"
)

# Experiment B collective benchmark: a Map-over-N state machine that runs
# exp_b_collectives.py per rank (redis OOB / FMI channel), one execution per world
# size N. Reached explicitly via `--arch collectives`. The state machine itself is a
# deployment follow-on (Map over world_size ranks); the sweep logic + payloads +
# S3 keys here are verified with `--dry-run`.
COLLECTIVES_WORKFLOW_ARN = (
    f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:cylon-armada-collectives-workflow"
)

SCENARIOS_DIR = Path(__file__).parent.parent.parent.parent / "shared" / "scripts" / "experiment" / "scenarios"

# ---------------------------------------------------------------------------
# Input builders — each architecture has a slightly different SFN schema
# ---------------------------------------------------------------------------

def _build_lambda_input(scenario, tasks, world_size, experiment_name, results_s3_dir,
                        workflow_id=None, scaling="weak", context_backend="redis",
                        fmi_channel="direct"):
    payload = {
        "workflow_id":      workflow_id or f"sweep-{experiment_name}-{str(uuid.uuid4())[:8]}",
        "tasks":            tasks,
        "scaling":          scaling,
        "world_size":       world_size,
        "results_s3_dir":   results_s3_dir,
        "experiment_name":  experiment_name,
        "context_backend":  context_backend,
    }
    if context_backend == "cylon":
        payload["fmi_channel_type"] = fmi_channel
    return payload


def _build_ecs_input(scenario, tasks, world_size, experiment_name, results_dir, workflow_id=None, scaling="weak"):
    return {
        "workflow_id":      workflow_id or f"sweep-{experiment_name}-{str(uuid.uuid4())[:8]}",
        "tasks":            tasks,
        "scaling":          scaling,
        "world_size":       world_size,
        "results_dir":      results_dir,
        "experiment_name":  experiment_name,
        "s3_scripts_bucket": S3_SCRIPTS_BUCKET,
        "s3_scripts_prefix": S3_SCRIPTS_PREFIX,
    }


def build_sfn_input(arch, scenario, tasks, world_size, experiment_name, workflow_id=None,
                    scaling="weak", context_backend="redis", results_scaling=None,
                    fmi_channel="direct"):
    """Build the Step Functions input payload for the given architecture.

    workflow_id is shared across runs of the same (arch, scenario, scaling, world_size)
    so that run 2-4 can reuse contexts stored by run 1.
    context_backend controls similarity search: "redis" (numpy, concurrent-safe) or
    "cylon" (Arrow SIMD, for FMI broadcast path in Phase 2).
    results_scaling: scaling label used for the S3 results path (defaults to scaling).
                     Needed because SFN always uses scaling="weak" for chunk() behavior,
                     but the results path should reflect the actual experiment type.
    """
    rs = results_scaling or scaling
    results_dir = f"results/{arch}/{scenario}/{rs}/"
    if arch in ("lambda-python", "lambda-nodejs"):
        return _build_lambda_input(scenario, tasks, world_size, experiment_name,
                                   results_dir, workflow_id, scaling, context_backend,
                                   fmi_channel)
    else:
        return _build_ecs_input(scenario, tasks, world_size, experiment_name,
                                results_dir, workflow_id, scaling)


# ---------------------------------------------------------------------------
# Experiment A / A2 benchmark input — single-node, no scenarios/world_size
# ---------------------------------------------------------------------------

def build_benchmark_input(experiment_name, results_s3_dir, batch_sizes, dims,
                          warmup, reps, seed, matrix_runs):
    """Payload for the cylon-armada-benchmark state machine.

    Forwarded verbatim to armada_benchmark.handler, which applies its own
    defaults for any omitted key. matrix_runs is the number of times the whole
    matrix repeats *inside one Lambda* (writes run_N/ keys); the sweep-level
    --runs fires separate executions (cold-Lambda variance) instead.
    """
    return {
        "experiment_name": experiment_name,
        "results_s3_dir":  results_s3_dir,
        "batch_sizes":     batch_sizes,
        "dims":            dims,
        "warmup":          warmup,
        "reps":            reps,
        "seed":            seed,
        "runs":            matrix_runs,
    }


# ---------------------------------------------------------------------------
# Experiment B collective benchmark input — Map over `world_size` ranks
# ---------------------------------------------------------------------------

def build_collectives_input(experiment_name, results_s3_dir, world_size, channel,
                            collectives, msg_sizes, warmup, reps, runs):
    """Payload for the Experiment B collective-benchmark state machine.

    Forwarded to the Map-over-N workflow, which runs exp_b_collectives.py on each of
    `world_size` ranks over the given `channel` (redis-OOB UCC / FMI). The `runs`
    warmed measurement passes happen inside one execution on the same warm workers
    (writing run{n}_exp_b_collectives_results.csv), mirroring the local launcher; the
    world_size sweep fires separate executions.
    """
    return {
        "experiment_name": experiment_name,
        "results_s3_dir":  results_s3_dir,
        "world_size":      world_size,
        "channel":         channel,
        "collectives":     collectives,
        "msg_sizes":       msg_sizes,
        "warmup":          warmup,
        "reps":            reps,
        "runs":            runs,
    }


# ---------------------------------------------------------------------------
# Task sampling — stratified, same seed across runs for reproducibility
# ---------------------------------------------------------------------------

def sample_tasks(scenario_file: Path, n: int, seed: int = 42) -> list:
    """Sample n tasks from a scenario file, tiling if n > scenario size.

    Mirrors the cylon scaling.py pattern where `rows` is fixed per worker
    (weak) or total (strong), and the scenario file is the task pool.
    Tiling is valid — production LLM workloads naturally see recurring task
    types, and context reuse is most valuable when similar tasks repeat.
    """
    data = json.loads(scenario_file.read_text())
    all_tasks = data.get("tasks", [])
    if not all_tasks:
        raise ValueError(f"No tasks in {scenario_file}")
    if n <= len(all_tasks):
        if n == len(all_tasks):
            return all_tasks[:]
        import random
        rng = random.Random(seed)
        bucket_size = len(all_tasks) / n
        return [
            all_tasks[int(i * bucket_size) + rng.randint(0, int(bucket_size) - 1)]
            for i in range(n)
        ]
    # n > available tasks: tile the list (realistic for recurring task patterns)
    import math
    tiled = (all_tasks * math.ceil(n / len(all_tasks)))[:n]
    return tiled


# ---------------------------------------------------------------------------
# Execution — EXPRESS vs STANDARD handled separately
# ---------------------------------------------------------------------------


def fire_execution(sfn_client, arn, execution_name, sfn_input):
    """Fire a state machine execution asynchronously. Returns (name,)."""
    sfn_client.start_execution(
        stateMachineArn=arn,
        name=execution_name,
        input=json.dumps(sfn_input),
    )
    logger.info("  → fired %s", execution_name)


def poll_s3_results(pending: dict, poll_interval: int = 20, timeout_seconds: int = 600):
    """Poll S3 for _metrics.json completion files.

    pending: {execution_name: s3_key}
    Returns: {execution_name: "SUCCEEDED" | "FAILED"}
    """
    s3 = boto3.client("s3", region_name=REGION)
    remaining = dict(pending)
    results = {}
    elapsed = 0

    while remaining and elapsed < timeout_seconds:
        time.sleep(poll_interval)
        elapsed += poll_interval
        done = []
        for name, key in remaining.items():
            try:
                s3.head_object(Bucket=RESULTS_BUCKET, Key=key)
                results[name] = "SUCCEEDED"
                done.append(name)
                logger.info("✓ %s — SUCCEEDED", name)
            except Exception:
                pass
        for name in done:
            del remaining[name]
        if remaining:
            logger.info("  %d still running...", len(remaining))

    for name in remaining:
        logger.warning("✗ %s — FAILED (no S3 result after %ds)", name, timeout_seconds)
        results[name] = "FAILED"

    return results


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(args, sweep_tag=""):
    sfn = boto3.client("stepfunctions", region_name=REGION)
    tag = f"_{sweep_tag}" if sweep_tag else ""

    architectures = ALL_ARCHITECTURES if args.arch == "all" else [args.arch]
    scenarios     = ALL_SCENARIOS if args.scenario == "all" else [args.scenario]

    for arch in architectures:
        arn = WORKFLOW_ARNS[arch]
        logger.info("=" * 60)
        logger.info("Architecture: %s", arch)
        logger.info("=" * 60)

        # Collect all configs for this architecture
        all_configs  = []   # [(exp_name, sfn_input, s3_key), ...]
        for scenario in scenarios:
            scenario_file = SCENARIOS_DIR / f"{scenario}.json"
            if not scenario_file.exists():
                logger.error("Scenario file not found: %s", scenario_file)
                continue
            for scaling in args.scaling:
                for world_size in args.world_sizes:
                    if scaling == "weak":
                        n_tasks = args.task_count * world_size
                    else:
                        n_tasks = args.task_count
                    tasks = sample_tasks(scenario_file, n_tasks, seed=42)
                    shared_workflow_id = (
                        f"{arch.replace('-','_')}_{scenario}_{scaling}_ws{world_size}_{sweep_tag}"
                    )
                    for run in range(1, args.runs + 1):
                        exp_name = (
                            f"{arch.replace('-','_')}_{scenario}_{scaling}_ws{world_size}_run{run}{tag}"
                        )[:80]
                        sfn_input = build_sfn_input(
                            arch, scenario, tasks, world_size, exp_name,
                            workflow_id=shared_workflow_id,
                            scaling="weak",
                            context_backend=getattr(args, "context_backend", "redis"),
                            results_scaling=scaling,
                            fmi_channel=getattr(args, "fmi_channel", "direct"),
                        )
                        # Expected S3 key written by armada_aggregate on completion
                        rs = scaling  # results_scaling
                        s3_key = f"results/{arch}/{scenario}/{rs}/{exp_name}_metrics.json"
                        all_configs.append((exp_name, sfn_input, s3_key))

        logger.info("Firing %d executions for %s", len(all_configs), arch)

        if args.dry_run:
            for name, _, key in all_configs:
                logger.info("  [dry-run] %s → s3://%s/%s", name, RESULTS_BUCKET, key)
            continue

        # Fire in batches of max_parallel, then poll S3 for completion
        batch_size = args.max_parallel
        all_results = {}
        for i in range(0, len(all_configs), batch_size):
            batch = all_configs[i:i + batch_size]
            pending = {}
            for name, sfn_input, s3_key in batch:
                try:
                    fire_execution(sfn, arn, name, sfn_input)
                    pending[name] = s3_key
                except Exception as e:
                    logger.error("Failed to fire %s: %s", name, e)
                    all_results[name] = "FAILED"
            batch_results = poll_s3_results(
                pending,
                poll_interval=getattr(args, "poll_interval", 20),
                timeout_seconds=getattr(args, "timeout", 600),
            )
            all_results.update(batch_results)

        succeeded = sum(1 for s in all_results.values() if s == "SUCCEEDED")
        failed    = sum(1 for s in all_results.values() if s != "SUCCEEDED")
        logger.info("Architecture %s complete: %d succeeded, %d failed",
                    arch, succeeded, failed)


# ---------------------------------------------------------------------------
# Benchmark sweep (Experiment A / A2) — fires the single-node state machine
# ---------------------------------------------------------------------------

def run_benchmark_sweep(args, sweep_tag=""):
    """Fire the Experiment A/A2 benchmark state machine `--runs` times and poll
    S3 for each run's results CSV. No scenario/world_size/scaling loop — the
    data axis (N x D) is swept inside the Lambda."""
    sfn = boto3.client("stepfunctions", region_name=REGION)
    arn = BENCHMARK_WORKFLOW_ARN
    tag = f"_{sweep_tag}" if sweep_tag else ""
    results_prefix = args.results_prefix.rstrip("/")

    logger.info("=" * 60)
    logger.info("Experiment A / A2 zero-copy benchmark")
    logger.info("  batch_sizes=%s dims=%s warmup=%d reps=%d matrix_runs=%d",
                args.batch_sizes, args.dims, args.warmup, args.reps, args.matrix_runs)
    logger.info("=" * 60)

    all_configs = []  # [(exp_name, sfn_input, s3_key), ...]
    for run in range(1, args.runs + 1):
        exp_name = f"benchmark_exp_a_run{run}{tag}"[:80]
        results_s3_dir = f"{results_prefix}/"
        sfn_input = build_benchmark_input(
            experiment_name=exp_name,
            results_s3_dir=results_s3_dir,
            batch_sizes=args.batch_sizes,
            dims=args.dims,
            warmup=args.warmup,
            reps=args.reps,
            seed=args.seed + run - 1,
            matrix_runs=args.matrix_runs,
        )
        # Completion marker: the results CSV the handler writes. With matrix_runs>1
        # the handler nests under run_N/, so the last matrix run is the marker.
        run_seg = "" if args.matrix_runs == 1 else f"run_{args.matrix_runs}/"
        s3_key = f"{results_prefix}/{run_seg}{exp_name}_exp_a_zerocopy_results.csv"
        all_configs.append((exp_name, sfn_input, s3_key))

    logger.info("Firing %d benchmark execution(s)", len(all_configs))

    if args.dry_run:
        for name, sfn_input, key in all_configs:
            logger.info("  [dry-run] %s → s3://%s/%s", name, RESULTS_BUCKET, key)
            logger.info("            input=%s", json.dumps(sfn_input))
        return

    batch_size = args.max_parallel
    all_results = {}
    for i in range(0, len(all_configs), batch_size):
        batch = all_configs[i:i + batch_size]
        pending = {}
        for name, sfn_input, s3_key in batch:
            try:
                fire_execution(sfn, arn, name, sfn_input)
                pending[name] = s3_key
            except Exception as e:
                logger.error("Failed to fire %s: %s", name, e)
                all_results[name] = "FAILED"
        # The benchmark Lambda runs up to its 900s timeout, so poll at least
        # that long regardless of the shared --timeout default (600s).
        batch_results = poll_s3_results(
            pending,
            poll_interval=getattr(args, "poll_interval", 20),
            timeout_seconds=max(getattr(args, "timeout", 600), 900),
        )
        all_results.update(batch_results)

    succeeded = sum(1 for s in all_results.values() if s == "SUCCEEDED")
    failed    = sum(1 for s in all_results.values() if s != "SUCCEEDED")
    logger.info("Benchmark sweep complete: %d succeeded, %d failed", succeeded, failed)


def run_collectives_sweep(args, sweep_tag=""):
    """Fire the Experiment B collective benchmark once per world_size N, poll S3 for
    each run's results CSV. World size is the sweep axis; the warmed --runs passes
    happen inside each execution (run{n}_ keys), like the local launcher."""
    sfn = boto3.client("stepfunctions", region_name=REGION)
    arn = COLLECTIVES_WORKFLOW_ARN
    tag = f"_{sweep_tag}" if sweep_tag else ""
    results_prefix = args.collectives_prefix.rstrip("/")

    logger.info("=" * 60)
    logger.info("Experiment B collective benchmark  channel=%s runs=%d", args.channel, args.runs)
    logger.info("  collectives=%s msg_sizes=%s world_sizes=%s",
                args.collectives, args.msg_sizes, args.world_sizes)
    logger.info("=" * 60)

    all_configs = []  # [(exp_name, sfn_input, s3_key), ...]
    for ws in args.world_sizes:
        exp_name = f"expb_{args.channel.replace('-', '_')}_ws{ws}{tag}"[:80]
        results_s3_dir = f"{results_prefix}/{args.channel}/ws{ws}/"
        sfn_input = build_collectives_input(
            experiment_name=exp_name,
            results_s3_dir=results_s3_dir,
            world_size=ws,
            channel=args.channel,
            collectives=args.collectives,
            msg_sizes=args.msg_sizes,
            warmup=args.warmup,
            reps=args.reps,
            runs=args.runs,
        )
        # Completion marker: rank 0's last warmed run CSV (name matches the aggregator
        # glob `*_exp_b_collectives_results.csv`).
        marker = ("exp_b_collectives_results.csv" if args.runs == 1
                  else f"run{args.runs}_exp_b_collectives_results.csv")
        s3_key = f"{results_s3_dir}{marker}"
        all_configs.append((exp_name, sfn_input, s3_key))

    logger.info("Firing %d collective execution(s)", len(all_configs))

    if args.dry_run:
        for name, sfn_input, key in all_configs:
            logger.info("  [dry-run] %s → s3://%s/%s", name, RESULTS_BUCKET, key)
            logger.info("            input=%s", json.dumps(sfn_input))
        return

    batch_size = args.max_parallel
    all_results = {}
    for i in range(0, len(all_configs), batch_size):
        batch = all_configs[i:i + batch_size]
        pending = {}
        for name, sfn_input, s3_key in batch:
            try:
                fire_execution(sfn, arn, name, sfn_input)
                pending[name] = s3_key
            except Exception as e:
                logger.error("Failed to fire %s: %s", name, e)
                all_results[name] = "FAILED"
        batch_results = poll_s3_results(
            pending,
            poll_interval=getattr(args, "poll_interval", 20),
            timeout_seconds=max(getattr(args, "timeout", 600), 900),
        )
        all_results.update(batch_results)

    succeeded = sum(1 for s in all_results.values() if s == "SUCCEEDED")
    failed    = sum(1 for s in all_results.values() if s != "SUCCEEDED")
    logger.info("Collective sweep complete: %d succeeded, %d failed", succeeded, failed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 cloud experiment sweep")

    parser.add_argument("--arch", required=True,
                        choices=ALL_ARCHITECTURES + ["all", "benchmark", "collectives"],
                        help="Architecture to sweep, 'all' (reuse workflows), "
                             "'benchmark' (Experiment A/A2 zero-copy state machine), or "
                             "'collectives' (Experiment B collective benchmark, N-sweep)")
    parser.add_argument("--scenario", default="all",
                        choices=ALL_SCENARIOS + ["all"],
                        help="Scenario to run, or 'all'")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1],
                        help="World sizes (parallel workers)")
    parser.add_argument("--scaling", nargs="+", default=["weak"],
                        choices=["weak", "strong"],
                        help="Scaling mode(s) — weak: each worker gets task_count tasks; "
                             "strong: all workers share task_count tasks")
    parser.add_argument("--task-count", type=int, default=16,
                        help="Number of tasks per worker (weak) or total tasks (strong)")
    parser.add_argument("--runs", type=int, default=4,
                        help="Number of runs per config (for error bars)")
    parser.add_argument("--max-parallel", type=int, default=10,
                        help="Max concurrent Step Functions executions before waiting")
    parser.add_argument("--context-backend", default="redis",
                        choices=["redis", "cylon"],
                        help="Context similarity backend: redis (numpy, concurrent-safe) "
                             "or cylon (Arrow SIMD, for FMI Phase 2)")
    parser.add_argument("--fmi-channel", default="direct",
                        choices=["direct", "redis", "s3"],
                        help="FMI channel type for cylon backend: direct (TCPunch P2P), "
                             "redis, or s3 (default: direct)")
    parser.add_argument("--poll-interval", type=int, default=20,
                        help="Seconds between S3 result polls (default 20)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Max seconds to wait per batch for S3 results (default 600)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without firing executions")

    # --- Experiment A / A2 benchmark options (only used with --arch benchmark) ---
    bench = parser.add_argument_group("benchmark (--arch benchmark)")
    bench.add_argument("--batch-sizes", type=int, nargs="+", default=[100, 500, 1000, 5000],
                       help="Embedding batch sizes N (default: 100 500 1000 5000)")
    bench.add_argument("--dims", type=int, nargs="+", default=[256, 512, 1024],
                       help="Embedding dimensions D (default: 256 512 1024)")
    bench.add_argument("--warmup", type=int, default=3,
                       help="Warmup reps discarded before timing (default 3)")
    bench.add_argument("--reps", type=int, default=20,
                       help="Measured reps per cell (default 20)")
    bench.add_argument("--seed", type=int, default=42,
                       help="Base RNG seed; run k uses seed+k-1 (default 42)")
    bench.add_argument("--matrix-runs", type=int, default=1,
                       help="Times the whole matrix repeats inside one Lambda "
                            "(writes run_N/ keys); use --runs for separate executions")
    bench.add_argument("--results-prefix", default="results/benchmark/exp_a",
                       help="S3 prefix for benchmark result CSVs "
                            "(default: results/benchmark/exp_a)")

    # --- Experiment B collective options (only used with --arch collectives) ---
    # Reuses --world-sizes (the N sweep), --runs (warmed passes), --warmup, --reps.
    coll = parser.add_argument_group("collectives (--arch collectives)")
    coll.add_argument("--collectives", nargs="+",
                      default=["scatter", "scatterv", "gather", "allgather",
                               "reduce", "broadcast", "allreduce", "barrier"],
                      help="Collectives to benchmark (default: all eight)")
    coll.add_argument("--msg-sizes", type=int, nargs="+",
                      default=[8, 64, 512, 4096, 32768, 262144, 1048576],
                      help="Message sizes in bytes (default: 8B..1MB powers of two)")
    coll.add_argument("--channel", default="ucc",
                      choices=["ucc", "fmi-redis", "fmi-direct"],
                      help="Collective channel (default: ucc)")
    coll.add_argument("--collectives-prefix", default="results/collectives",
                      help="S3 prefix for collective result CSVs (default: results/collectives)")

    args = parser.parse_args()

    # Unique sweep tag prevents ExecutionAlreadyExists on retries
    sweep_tag = datetime.utcnow().strftime("%m%d%H%M")

    # Experiment A / A2 benchmark: separate flow (no scenarios/world_size/scaling).
    if args.arch == "benchmark":
        logger.info("Benchmark plan: %d execution(s) (dry_run=%s, tag=%s)",
                    args.runs, args.dry_run, sweep_tag)
        run_benchmark_sweep(args, sweep_tag)
        return

    # Experiment B collective benchmark: N-sweep (one execution per world_size), no
    # scenarios/scaling. Warmed --runs happen inside each execution.
    if args.arch == "collectives":
        logger.info("Collective plan: %d execution(s) over N=%s (dry_run=%s, tag=%s)",
                    len(args.world_sizes), args.world_sizes, args.dry_run, sweep_tag)
        run_collectives_sweep(args, sweep_tag)
        return

    total = (
        (len(ALL_ARCHITECTURES) if args.arch == "all" else 1) *
        (len(ALL_SCENARIOS) if args.scenario == "all" else 1) *
        len(args.world_sizes) *
        len(args.scaling) *
        args.runs
    )
    logger.info("Sweep plan: %d total executions (dry_run=%s, tag=%s)",
                total, args.dry_run, sweep_tag)

    run_sweep(args, sweep_tag)


if __name__ == "__main__":
    main()