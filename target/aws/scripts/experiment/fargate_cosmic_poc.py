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
    python fargate_cosmic_poc.py --world-sizes 1 2 4 8 --galaxies 256
    python fargate_cosmic_poc.py --world-sizes 4 --live      # spends Bedrock money

Prerequisites:
    - cylon-armada-python image built with the AstroMAE deps and pushed to ECR
    - model + data uploaded to s3://<bucket>/<MODEL_KEY|DATA_KEY> (see --help)
"""

import argparse
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
RESULTS_PREFIX = "cylon-armada/results/exp_e_cosmic_fargate"
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


def build_overrides(rank, world_size, comm_name, s3_prefix, args):
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
        {"name": "INFERENCE_DEVICE", "value": args.device},
        {"name": "INFERENCE_BATCH_SIZE", "value": str(args.batch_size)},
        {"name": "LIVE", "value": "1" if args.live else "0"},
    ]
    if args.galaxies:
        environment.append({"name": "GALAXIES", "value": str(args.galaxies)})
    return {
        "containerOverrides": [
            {
                "name": CONTAINER_NAME,
                "command": ["python", "-c", RUNNER_SCRIPT],
                "environment": environment,
            }
        ]
    }


def launch_world_size(ecs, world_size, args):
    comm_name = f"cosmic_poc_{world_size}_{uuid.uuid4().hex[:8]}"
    s3_prefix = f"{RESULTS_PREFIX}/ws{world_size}/{comm_name}/"
    logger.info("world_size=%d comm_name=%s -> s3://%s/%s",
                world_size, comm_name, RESULTS_BUCKET, s3_prefix)

    if args.dry_run:
        overrides = build_overrides(0, world_size, comm_name, s3_prefix, args)
        logger.info("[dry-run] rank 0 overrides=%s", json.dumps(overrides)[:900])
        return []

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
            overrides=build_overrides(rank, world_size, comm_name, s3_prefix, args),
        )
        if resp.get("failures"):
            raise RuntimeError(f"rank {rank} run_task failed: {resp['failures']}")
        return resp["tasks"][0]["taskArn"]

    # Ranks must come up together — the channel establishes connections during
    # communicator construction, so a straggler stalls every peer it pairs with.
    with ThreadPoolExecutor(max_workers=min(world_size, 16)) as pool:
        arns = list(pool.map(_launch, range(world_size)))
    logger.info("launched %d tasks for world_size=%d", len(arns), world_size)
    return arns


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="must include 1 and 2 as baselines")
    parser.add_argument("--galaxies", type=int, default=None,
                        help="cap the population (default: the whole partition)")
    parser.add_argument("--device", default="cpu",
                        help="torch device inside the task; Fargate has no GPU")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--listen-port", type=int, default=10000)
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--live", action="store_true",
                        help="real Bedrock on every rank — costs money per galaxy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--settle-s", type=int, default=30,
                        help="pause between world sizes so tasks drain")
    args = parser.parse_args()

    if args.device != "cpu":
        parser.error("Fargate has no GPU support — use the ECS EC2 GPU arm for cuda")

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