"""Experiment B collectives sweep on Fargate — direct-redis channel.

Drives exp_b_collectives.py across a world-size grid by launching one ECS
Fargate task per rank per world size (all ranks for a given N launched
concurrently, since the mesh needs everyone up at once). Mirrors the
already-completed Rivanna sweep's methodology exactly (same collectives,
message-size grid, reps, and 4 runs per world size — exp_b_collectives.py's
own defaults already match) so the two result sets are directly comparable.
Results are written locally inside each container by exp_b_collectives.py,
then uploaded to S3 by rank 0 only.

Usage:
    python fargate_expb_sweep.py --world-sizes 1 2 --dry-run
    python fargate_expb_sweep.py --world-sizes 1 2 4 8 16 32 64
"""

import argparse
import concurrent.futures
import json
import logging
import time
import uuid

import boto3

logger = logging.getLogger(__name__)

REGION = "us-east-1"
CLUSTER = "CylonFargateExperiments"
TASK_DEFINITION = "cylon-armada-python"
CONTAINER_NAME = "cylon-armada"
SUBNETS = ["subnet-07995eea6c462cd73", "subnet-0979c94513025746c"]
REDIS_ADDR = "dev-cylon-redis1.aws-cylondata.com:6379"
RESULTS_BUCKET = "staylor.dev2"
RESULTS_PREFIX = "cylon-armada/results/exp_b_collectives_fargate"

RUNNER_SCRIPT = """import sys, os, subprocess, boto3
sys.path.insert(0, '/cylon-armada/scripts')
world_size = int(os.environ['WORLD_SIZE'])
output_dir = '/tmp/expb_results'
boto3.client('s3').download_file(
    'staylor.dev2', 'cylon-armada/manual-scripts/exp_b_collectives.py',
    '/cylon-armada/scripts/experiment/exp_b_collectives.py',
)
cmd = [
    'python', '/cylon-armada/scripts/experiment/exp_b_collectives.py',
    '--channel', 'fmi-direct-redis',
    '--world-size', str(world_size),
    '--runs', os.environ.get('EXPB_RUNS', '4'),
    '--redis-addr', os.environ['REDIS_ADDR'],
    '--output', output_dir,
]
result = subprocess.run(cmd)
if os.path.isdir(output_dir) and os.listdir(output_dir):
    s3 = boto3.client('s3')
    bucket = os.environ.get('RESULTS_BUCKET', 'staylor.dev2')
    prefix = os.environ['S3_RESULTS_PREFIX']
    for fname in os.listdir(output_dir):
        local_path = os.path.join(output_dir, fname)
        s3.upload_file(local_path, bucket, prefix + fname)
        print('uploaded', fname, 'to s3://' + bucket + '/' + prefix + fname)
sys.exit(result.returncode)
"""


def build_overrides(rank, world_size, runs, s3_prefix, session_id):
    return {
        "containerOverrides": [
            {
                "name": CONTAINER_NAME,
                "command": ["python", "-c", RUNNER_SCRIPT],
                "environment": [
                    {"name": "RANK", "value": str(rank)},
                    {"name": "WORLD_SIZE", "value": str(world_size)},
                    {"name": "REDIS_ADDR", "value": REDIS_ADDR},
                    {"name": "EXPB_RUNS", "value": str(runs)},
                    {"name": "RESULTS_BUCKET", "value": RESULTS_BUCKET},
                    {"name": "S3_RESULTS_PREFIX", "value": s3_prefix},
                    {"name": "CYLON_SESSION_ID", "value": session_id},
                ],
            }
        ]
    }


def launch_world_size(ecs, world_size, runs, dry_run):
    """Launch world_size concurrent tasks for one world size, wait for all to
    stop, return per-rank exit codes."""
    s3_prefix = f"{RESULTS_PREFIX}/ws{world_size}/"
    session_id = f"expb_ws{world_size}_{uuid.uuid4().hex[:12]}"
    logger.info("world_size=%d: launching %d tasks (dry_run=%s) session_id=%s",
                world_size, world_size, dry_run, session_id)

    if dry_run:
        for rank in range(world_size):
            overrides = build_overrides(rank, world_size, runs, s3_prefix, session_id)
            logger.info("[dry-run] rank=%d overrides=%s", rank, json.dumps(overrides))
        return {}

    task_arns = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, world_size)) as pool:
        futures = []
        for rank in range(world_size):
            overrides = build_overrides(rank, world_size, runs, s3_prefix, session_id)

            def _launch(overrides=overrides, rank=rank):
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
                    overrides=overrides,
                )
                if resp.get("failures"):
                    raise RuntimeError(f"rank {rank} run_task failed: {resp['failures']}")
                return resp["tasks"][0]["taskArn"]

            futures.append(pool.submit(_launch))
            time.sleep(0.1)  # avoid bursting the RunTask API

        for f in concurrent.futures.as_completed(futures):
            task_arns.append(f.result())

    logger.info("world_size=%d: %d tasks launched, waiting for completion", world_size, len(task_arns))

    # ecs.get_waiter('tasks_stopped') caps at 100 tasks per call and polls
    # every 6s up to 100 attempts (10 min) by default — fine up to ws=64.
    waiter = ecs.get_waiter("tasks_stopped")
    waiter.wait(cluster=CLUSTER, tasks=task_arns, WaiterConfig={"Delay": 6, "MaxAttempts": 100})

    described = ecs.describe_tasks(cluster=CLUSTER, tasks=task_arns)["tasks"]
    exit_codes = {}
    for t in described:
        containers = t.get("containers", [])
        exit_code = containers[0].get("exitCode") if containers else None
        exit_codes[t["taskArn"]] = exit_code

    failed = {arn: code for arn, code in exit_codes.items() if code != 0}
    if failed:
        logger.error("world_size=%d: %d/%d tasks failed: %s", world_size, len(failed), len(task_arns), failed)
    else:
        logger.info("world_size=%d: all %d tasks exited 0", world_size, len(task_arns))
    return exit_codes


def main():
    parser = argparse.ArgumentParser(description="Experiment B direct-redis sweep on Fargate")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ecs = boto3.client("ecs", region_name=REGION)

    results = {}
    for ws in args.world_sizes:
        results[ws] = launch_world_size(ecs, ws, args.runs, args.dry_run)

    if not args.dry_run:
        any_failed = any(code != 0 for codes in results.values() for code in codes.values())
        if any_failed:
            logger.error("sweep completed with failures — see above")
            raise SystemExit(1)
        logger.info("sweep completed cleanly across world sizes: %s", args.world_sizes)


if __name__ == "__main__":
    main()