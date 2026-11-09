"""Experiment B collectives sweep on Fargate.

Channels: fmi-direct-redis (Cylon, the direct Redis communicator), ray-gloo and
ray-objectstore (Ray, exp_b_ray.py, on the Ray tasks' security group).

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
import os
import sys
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

RAY_CHANNELS = ("ray-gloo", "ray-objectstore")
CHANNELS = ("fmi-direct-redis",) + RAY_CHANNELS
PAYLOADS = ("float32", "kv_pairs")

RUNNER_SCRIPT = """import sys, os, subprocess, boto3
channel = os.environ.get('EXPB_CHANNEL', 'fmi-direct-redis')
is_ray = channel.startswith('ray-')
bucket = os.environ.get('RESULTS_BUCKET', 'staylor.dev2')
scripts = '/cylon-armada/scripts'
if os.environ.get('S3_SCRIPTS_PREFIX'):
    sys.path.insert(0, '/cylon-armada')
    from s3_loader import load_scripts
    if not load_scripts(os.environ.get('SCRIPTS_BUCKET', bucket), os.environ['S3_SCRIPTS_PREFIX'],
                        '/tmp/armada_scripts'):
        print('could not load scripts from', os.environ['S3_SCRIPTS_PREFIX'], flush=True)
        sys.exit(3)
    scripts = '/tmp/armada_scripts'
else:
    modules = ['experiment/exp_b_collectives.py', 'experiment/exp_b_payloads.py']
    if is_ray:
        modules += ['experiment/exp_b_ray.py', 'ray_arm/collective.py', 'ray_arm/exchange_board.py']
    manual_prefix = os.environ.get('MANUAL_SCRIPTS_PREFIX', 'cylon-armada/manual-scripts/')
    s3 = boto3.client('s3')
    for module in modules:
        s3.download_file(os.environ.get('SCRIPTS_BUCKET', bucket),
                         manual_prefix + os.path.basename(module), scripts + '/' + module)
world_size = os.environ['WORLD_SIZE']
output_dir = os.environ.get('EXPB_OUTPUT_DIR', '/tmp/expb_results')
cmd = [sys.executable, '-m', 'experiment.exp_b_ray' if is_ray else 'experiment.exp_b_collectives',
       '--channel', channel,
       '--payload', os.environ.get('EXPB_PAYLOAD', 'float32'),
       '--world-size', world_size,
       '--runs', os.environ.get('EXPB_RUNS', '4'),
       '--output', output_dir]
if not is_ray:
    cmd += ['--redis-addr', os.environ['REDIS_ADDR']]
env = dict(os.environ, PYTHONPATH=scripts + ':' + os.environ.get('PYTHONPATH', ''))
result = subprocess.run(cmd, cwd=scripts, env=env)
if os.path.isdir(output_dir) and os.listdir(output_dir):
    s3 = boto3.client('s3')
    prefix = os.environ['S3_RESULTS_PREFIX']
    for fname in os.listdir(output_dir):
        local_path = os.path.join(output_dir, fname)
        s3.upload_file(local_path, bucket, prefix + fname)
        print('uploaded', fname, 'to s3://' + bucket + '/' + prefix + fname)
sys.exit(result.returncode)
"""


def results_prefix_for(results_prefix, channel, payload, world_size):
    return f"{results_prefix}/{channel}/{payload}/ws{world_size}/"


def security_groups_for_channel(channel, ray_group_id):
    """Ray channels get the members-only Ray group; the Cylon channel keeps
    the default its earlier results were measured under."""
    return [ray_group_id] if channel in RAY_CHANNELS else None


def resolve_ray_group_id(ec2):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fargate_cosmic_poc import resolve_ray_security_group

    return resolve_ray_security_group(ec2)


def build_overrides(rank, world_size, runs, s3_prefix, session_id, channel, payload,
                    scripts_prefix):
    environment = [
        {"name": "RANK", "value": str(rank)},
        {"name": "WORLD_SIZE", "value": str(world_size)},
        {"name": "REDIS_ADDR", "value": REDIS_ADDR},
        {"name": "EXPB_RUNS", "value": str(runs)},
        {"name": "RESULTS_BUCKET", "value": RESULTS_BUCKET},
        {"name": "S3_RESULTS_PREFIX", "value": s3_prefix},
        {"name": "CYLON_SESSION_ID", "value": session_id},
        {"name": "EXPB_CHANNEL", "value": channel},
        {"name": "EXPB_PAYLOAD", "value": payload},
    ]
    if channel in RAY_CHANNELS:
        redis_host, redis_port = REDIS_ADDR.split(":")
        environment += [
            {"name": "COMM_NAME", "value": session_id},
            {"name": "REDIS_HOST", "value": redis_host},
            {"name": "REDIS_PORT", "value": redis_port},
        ]
    if scripts_prefix:
        environment.append({"name": "S3_SCRIPTS_PREFIX", "value": scripts_prefix})
    return {
        "containerOverrides": [
            {
                "name": CONTAINER_NAME,
                "command": ["python", "-c", RUNNER_SCRIPT],
                "environment": environment,
            }
        ]
    }


def launch_world_size(ecs, world_size, args, security_groups):
    """Launch world_size concurrent tasks for one world size, wait for all to
    stop, return per-rank exit codes."""
    runs, dry_run = args.runs, args.dry_run
    s3_prefix = results_prefix_for(args.results_prefix, args.channel, args.payload, world_size)
    session_id = f"expb_ws{world_size}_{uuid.uuid4().hex[:12]}"
    logger.info("world_size=%d: launching %d tasks (dry_run=%s) session_id=%s",
                world_size, world_size, dry_run, session_id)

    if dry_run:
        for rank in range(world_size):
            overrides = build_overrides(rank, world_size, runs, s3_prefix, session_id,
                                        args.channel, args.payload, args.scripts_prefix)
            logger.info("[dry-run] rank=%d overrides=%s", rank, json.dumps(overrides))
        return {}

    vpc_configuration = {"subnets": SUBNETS, "assignPublicIp": "ENABLED"}
    if security_groups:
        vpc_configuration["securityGroups"] = list(security_groups)

    task_arns = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, world_size)) as pool:
        futures = []
        for rank in range(world_size):
            overrides = build_overrides(rank, world_size, runs, s3_prefix, session_id,
                                        args.channel, args.payload, args.scripts_prefix)

            def _launch(overrides=overrides, rank=rank):
                resp = ecs.run_task(
                    cluster=CLUSTER,
                    taskDefinition=args.task_definition,
                    launchType="FARGATE",
                    networkConfiguration={"awsvpcConfiguration": vpc_configuration},
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
    parser = argparse.ArgumentParser(description="Experiment B collectives sweep on Fargate")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--channel", default="fmi-direct-redis", choices=CHANNELS)
    parser.add_argument("--payload", default="float32", choices=PAYLOADS)
    parser.add_argument("--task-definition", default=TASK_DEFINITION)
    parser.add_argument("--results-prefix", default=RESULTS_PREFIX)
    parser.add_argument("--scripts-prefix", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ecs = boto3.client("ecs", region_name=REGION)
    ray_group_id = None
    if args.channel in RAY_CHANNELS and not args.dry_run:
        ray_group_id = resolve_ray_group_id(boto3.client("ec2", region_name=REGION))
    security_groups = security_groups_for_channel(args.channel, ray_group_id)

    results = {}
    for ws in args.world_sizes:
        results[ws] = launch_world_size(ecs, ws, args, security_groups)

    if not args.dry_run:
        any_failed = any(code != 0 for codes in results.values() for code in codes.values())
        if any_failed:
            logger.error("sweep completed with failures — see above")
            raise SystemExit(1)
        logger.info("sweep completed cleanly across world sizes: %s", args.world_sizes)


if __name__ == "__main__":
    main()