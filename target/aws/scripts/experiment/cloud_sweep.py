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
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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

SCENARIOS_DIR = Path(__file__).parent.parent.parent.parent / "shared" / "scripts" / "experiment" / "scenarios"

# ---------------------------------------------------------------------------
# Input builders — each architecture has a slightly different SFN schema
# ---------------------------------------------------------------------------

def _build_lambda_input(scenario, tasks, world_size, experiment_name, results_s3_dir):
    return {
        "workflow_id":      f"sweep-{experiment_name}-{str(uuid.uuid4())[:8]}",
        "tasks":            tasks,
        "scaling":          "weak",
        "world_size":       world_size,
        "results_s3_dir":   results_s3_dir,
        "experiment_name":  experiment_name,
    }


def _build_ecs_input(scenario, tasks, world_size, experiment_name, results_dir):
    return {
        "workflow_id":      f"sweep-{experiment_name}-{str(uuid.uuid4())[:8]}",
        "tasks":            tasks,
        "scaling":          "weak",
        "world_size":       world_size,
        "results_dir":      results_dir,
        "experiment_name":  experiment_name,
        "s3_scripts_bucket": S3_SCRIPTS_BUCKET,
        "s3_scripts_prefix": S3_SCRIPTS_PREFIX,
    }


def build_sfn_input(arch, scenario, tasks, world_size, experiment_name):
    """Build the Step Functions input payload for the given architecture."""
    if arch in ("lambda-python", "lambda-nodejs"):
        results_dir = f"results/{arch}/{scenario}/weak/"
        return _build_lambda_input(scenario, tasks, world_size, experiment_name, results_dir)
    else:
        results_dir = f"results/{arch}/{scenario}/weak/"
        return _build_ecs_input(scenario, tasks, world_size, experiment_name, results_dir)


# ---------------------------------------------------------------------------
# Task sampling — stratified, same seed across runs for reproducibility
# ---------------------------------------------------------------------------

def sample_tasks(scenario_file: Path, n: int, seed: int = 42) -> list:
    """Stratified sample of n tasks from a scenario file."""
    data = json.loads(scenario_file.read_text())
    all_tasks = data.get("tasks", [])
    if not all_tasks:
        raise ValueError(f"No tasks in {scenario_file}")
    if n >= len(all_tasks):
        return all_tasks[:n]
    # Stratified: divide list into n buckets, pick one from each
    import random
    rng = random.Random(seed)
    bucket_size = len(all_tasks) / n
    return [
        all_tasks[int(i * bucket_size) + rng.randint(0, int(bucket_size) - 1)]
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Execution — EXPRESS vs STANDARD handled separately
# ---------------------------------------------------------------------------

EXPRESS_ARCHS = {"lambda-python", "lambda-nodejs"}


def _get_workflow_type(sfn_client, arn):
    return sfn_client.describe_state_machine(stateMachineArn=arn)["type"]


def run_express(sfn_client, arn, execution_name, sfn_input):
    """Run an EXPRESS state machine synchronously. Returns (name, status)."""
    try:
        resp = sfn_client.start_sync_execution(
            stateMachineArn=arn,
            name=execution_name,
            input=json.dumps(sfn_input),
        )
        status = resp["status"]
    except Exception as e:
        logger.error("EXPRESS execution failed %s: %s", execution_name, e)
        status = "FAILED"
    icon = "✓" if status == "SUCCEEDED" else "✗"
    logger.info("%s %s — %s", icon, execution_name, status)
    return execution_name, status


def run_express_parallel(sfn_client, arn, configs, max_parallel):
    """Run multiple EXPRESS executions in parallel threads. Returns {name: status}."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(run_express, sfn_client, arn, name, inp): name
            for name, inp in configs
        }
        for future in as_completed(futures):
            name, status = future.result()
            results[name] = status
    return results


def fire_standard(sfn_client, arn, execution_name, sfn_input):
    """Start a STANDARD execution. Returns (name, execution_arn)."""
    resp = sfn_client.start_execution(
        stateMachineArn=arn,
        name=execution_name,
        input=json.dumps(sfn_input),
    )
    logger.info("  → %s", execution_name)
    return execution_name, resp["executionArn"]


def poll_standard(sfn_client, execution_arns: dict, poll_interval: int = 30):
    """Poll STANDARD executions until all complete. Returns {name: status}."""
    pending = dict(execution_arns)
    results = {}

    while pending:
        done = []
        for name, arn in pending.items():
            resp = sfn_client.describe_execution(executionArn=arn)
            status = resp["status"]
            if status in ("SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"):
                results[name] = status
                done.append(name)
                icon = "✓" if status == "SUCCEEDED" else "✗"
                logger.info("%s %s — %s", icon, name, status)
        for name in done:
            del pending[name]
        if pending:
            logger.info("%d executions still running...", len(pending))
            time.sleep(poll_interval)

    return results


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(args):
    sfn = boto3.client("stepfunctions", region_name=REGION)

    architectures = ALL_ARCHITECTURES if args.arch == "all" else [args.arch]
    scenarios     = ALL_SCENARIOS if args.scenario == "all" else [args.scenario]

    for arch in architectures:
        arn = WORKFLOW_ARNS[arch]
        is_express = arch in EXPRESS_ARCHS
        logger.info("=" * 60)
        logger.info("Architecture: %s (%s)", arch,
                    "EXPRESS" if is_express else "STANDARD")
        logger.info("=" * 60)

        # Collect all configs for this architecture
        all_configs = []
        for scenario in scenarios:
            scenario_file = SCENARIOS_DIR / f"{scenario}.json"
            if not scenario_file.exists():
                logger.error("Scenario file not found: %s", scenario_file)
                continue
            for world_size in args.world_sizes:
                tasks = sample_tasks(scenario_file, args.task_count, seed=42)
                for run in range(1, args.runs + 1):
                    exp_name = (
                        f"{arch.replace('-','_')}_{scenario}_ws{world_size}_run{run}"
                    )[:80]
                    sfn_input = build_sfn_input(arch, scenario, tasks, world_size, exp_name)
                    all_configs.append((exp_name, sfn_input))

        logger.info("Firing %d executions for %s", len(all_configs), arch)

        if args.dry_run:
            for name, _ in all_configs:
                logger.info("  [dry-run] %s", name)
            continue

        if is_express:
            # EXPRESS — run synchronously in parallel threads
            results = run_express_parallel(sfn, arn, all_configs, args.max_parallel)
        else:
            # STANDARD — fire all, then poll
            execution_arns = {}
            for name, sfn_input in all_configs:
                n, exec_arn = fire_standard(sfn, arn, name, sfn_input)
                execution_arns[n] = exec_arn
                # Throttle if needed
                if len(execution_arns) >= args.max_parallel:
                    batch = poll_standard(sfn, execution_arns)
                    failed = [n for n, s in batch.items() if s != "SUCCEEDED"]
                    if failed:
                        logger.warning("%d failed: %s", len(failed), failed)
                    execution_arns = {}
            if execution_arns:
                results = poll_standard(sfn, execution_arns)
            else:
                results = {}

        succeeded = sum(1 for s in results.values() if s == "SUCCEEDED")
        failed    = sum(1 for s in results.values() if s != "SUCCEEDED")
        logger.info("Architecture %s complete: %d succeeded, %d failed",
                    arch, succeeded, failed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 cloud experiment sweep")

    parser.add_argument("--arch", required=True,
                        choices=ALL_ARCHITECTURES + ["all"],
                        help="Architecture to sweep, or 'all'")
    parser.add_argument("--scenario", default="all",
                        choices=ALL_SCENARIOS + ["all"],
                        help="Scenario to run, or 'all'")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1],
                        help="World sizes (parallel workers)")
    parser.add_argument("--task-count", type=int, default=16,
                        help="Number of tasks sampled per scenario per run")
    parser.add_argument("--runs", type=int, default=4,
                        help="Number of runs per config (for error bars)")
    parser.add_argument("--max-parallel", type=int, default=10,
                        help="Max concurrent Step Functions executions before waiting")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without firing executions")

    args = parser.parse_args()

    total = (
        (len(ALL_ARCHITECTURES) if args.arch == "all" else 1) *
        (len(ALL_SCENARIOS) if args.scenario == "all" else 1) *
        len(args.world_sizes) *
        args.runs
    )
    logger.info("Sweep plan: %d total executions (dry_run=%s)", total, args.dry_run)

    run_sweep(args)


if __name__ == "__main__":
    main()