#!/usr/bin/env python3
"""Locate the memory/latency crossover for Retrieve's stored-embedding index.

Retrieve today runs PointToPoint against a shared ContextManager that every
rank can already see in full (see cosmic_workflow.py's COSMIC_PATTERNS
docstring), so a REDUCE-merged, genuinely partitioned layout only has
something to win if the stored-embedding set gets too large for one rank to
hold and search alone. This script finds that point empirically instead of
guessing from memory-size arithmetic: it inserts synthetic 1024-dim
embeddings into a ContextManager(backend="cylon") in increasing steps and
records process RSS and ContextRouter.find_similar() latency at each step.

--live loads the real AstroMAE model first, inside the deployed cosmic
Fargate task, so the model's own memory footprint is included in the
baseline rather than measured in isolation. --dry-run-local skips AWS and
the model entirely, to sanity-check the ContextManager/ContextRouter path in
cylon_dev before spending anything (needs the usual
LD_LIBRARY_PATH/PYTHONPATH exports from CLAUDE.md's Environment Setup).

Usage:
    python embedding_scale_probe.py --dry-run-local --sizes 1000 5000 20000
    python embedding_scale_probe.py --live --sizes 50000 200000 1000000
"""

import argparse
import json
import os
import time
import uuid

import boto3

from fargate_cosmic_poc import (
    CLUSTER, CONTAINER_NAME, REGION, RESULTS_BUCKET, _run_tasks, _wait_for_tasks,
)

RESULTS_PREFIX = "cylon-armada/results/exp_e_embedding_scale_probe"

CONDA_ENV_LIB = os.environ.get("PROBE_CONDA_ENV_LIB", "/opt/conda/envs/cylon_dev/lib")

# Runs inside the task via `python -c`, the same convention fargate_cosmic_poc.py
# uses for its own RUNNER_SCRIPT — avoids needing this file to exist inside the
# image, so no rebuild or S3 script sync is required to change what gets probed.
INNER_SCRIPT = """import gc, json, os, time
import numpy as np
import boto3

def rss_mb():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

dimensions = int(os.environ['PROBE_DIMENSIONS'])
sizes = [int(s) for s in os.environ['PROBE_SIZES'].split(',')]
num_queries = int(os.environ['PROBE_NUM_QUERIES'])
device = os.environ.get('INFERENCE_DEVICE', 'cpu')
seed = int(os.environ.get('PROBE_SEED', '7'))

s3 = boto3.client('s3')
bucket = os.environ['ARTIFACT_BUCKET']
model_path = os.environ['ASTROMAE_MODEL_PATH']
model_key = os.environ['ASTROMAE_MODEL_KEY']
os.makedirs(os.path.dirname(model_path), exist_ok=True)
rss_before_model = rss_mb()
if not os.path.exists(model_path):
    print('downloading', model_key, '->', model_path, flush=True)
    s3.download_file(bucket, model_key, model_path)

from cosmic_ai.inference import load_model
# Kept alive for the whole probe — an unassigned load_model() call is freed
# the instant it returns, which would measure import overhead, not a
# resident model, and defeat the point of this baseline.
model = load_model(model_path, device=device)
rss_after_model = rss_mb()
print('model loaded, rss_after_model_mb=%.1f' % rss_after_model, flush=True)

from context.manager import ContextManager
from context.router import ContextRouter, SIMDBackend

results_bucket = os.environ['RESULTS_BUCKET']
results_prefix = os.environ['S3_RESULTS_PREFIX']

def checkpoint(sizes_so_far, complete):
    summary = {
        'dimensions': dimensions, 'device': device,
        'rss_before_model_mb': rss_before_model,
        'rss_after_model_mb': rss_after_model,
        'complete': complete,
        'sizes': sizes_so_far,
    }
    with open('/tmp/result.json', 'w') as f:
        json.dump(summary, f, indent=1)
    s3.upload_file('/tmp/result.json', results_bucket, results_prefix + 'result.json')

# Uploaded before any insertion starts too, so a failure during the first
# step's insert/query loop still leaves a record that the task got this far,
# instead of surfacing only as an S3 404 to the driver.
checkpoint([], complete=False)
print('uploaded started marker', flush=True)

manager = ContextManager(
    embedding_dim=dimensions, backend='cylon',
    persist_to_redis=False, snapshot_context_table=False,
)
router = ContextRouter(manager, backend=SIMDBackend.NUMPY)

rng = np.random.default_rng(seed)

def synthetic_vec():
    v = rng.standard_normal(dimensions).astype(np.float32)
    return v / np.linalg.norm(v)

results = []
inserted = 0
for target in sizes:
    t0 = time.perf_counter()
    for _ in range(target - inserted):
        manager.store_context(
            workflow_id='probe', task_description='', embedding=synthetic_vec(),
            response='x', cost_metadata={},
        )
    insert_s = time.perf_counter() - t0
    inserted = target
    gc.collect()
    rss_pre_query = rss_mb()

    # The first query after an insert step materializes the ContextTable's
    # Arrow buffers (old + new rows coexist briefly), so RSS sampled only
    # before querying can miss that peak — sample again after the first one.
    query_times = []
    match_counts = []
    for i in range(num_queries):
        q = synthetic_vec()
        qt0 = time.perf_counter()
        matches = router.find_similar(q, workflow_id='probe')
        query_times.append(time.perf_counter() - qt0)
        match_counts.append(len(matches))
        if i == 0:
            rss_post_first_query = rss_mb()

    row = {
        'galaxies': inserted,
        'rss_pre_query_mb': rss_pre_query,
        'rss_post_first_query_mb': rss_post_first_query,
        'insert_s_for_step': insert_s,
        'query_s_mean': float(np.mean(query_times)),
        'query_s_max': float(np.max(query_times)),
        'matches_mean': float(np.mean(match_counts)),
    }
    results.append(row)
    print('galaxies=%d rss_pre_query_mb=%.1f rss_post_first_query_mb=%.1f '
          'query_s_mean=%.4f query_s_max=%.4f matches_mean=%.2f' %
          (inserted, rss_pre_query, rss_post_first_query, row['query_s_mean'],
           row['query_s_max'], row['matches_mean']), flush=True)

    # Written and uploaded after EVERY step, not once at the end: the largest
    # size is the one most likely to OOM or time out, and losing every
    # smaller size's measurement to that would throw away good data.
    checkpoint(results, complete=(target == sizes[-1]))
    print('uploaded result through galaxies=%d' % inserted, flush=True)
"""


def build_overrides(sizes, dimensions, num_queries, s3_prefix, device):
    environment = [
        # Set explicitly rather than relying on the task definition's deployed
        # default agreeing with the RESULTS_BUCKET constant this driver reads
        # back from below — those are two independent sources of truth today.
        {"name": "RESULTS_BUCKET", "value": RESULTS_BUCKET},
        {"name": "S3_RESULTS_PREFIX", "value": s3_prefix},
        {"name": "PROBE_SIZES", "value": ",".join(str(s) for s in sizes)},
        {"name": "PROBE_DIMENSIONS", "value": str(dimensions)},
        {"name": "PROBE_NUM_QUERIES", "value": str(num_queries)},
        {"name": "INFERENCE_DEVICE", "value": device},
        # torch loads the system libstdc++ first, and the loader will not then
        # admit a second copy of that SONAME, so libarrow's GLIBCXX_3.4.32 is
        # unsatisfiable by the time context.manager imports. Preloading conda's
        # newer libstdc++ makes it resident before torch, which is backward
        # compatible for torch and sufficient for arrow. Removing this line
        # reintroduces an ImportError that only appears after the model loads.
        {"name": "LD_PRELOAD", "value": os.path.join(CONDA_ENV_LIB, "libstdc++.so.6")},
    ]
    return {
        "containerOverrides": [
            {
                "name": CONTAINER_NAME,
                "command": ["python", "-c", INNER_SCRIPT],
                "environment": environment,
            }
        ]
    }


def run_local(sizes, dimensions, num_queries, seed=7):
    """ContextManager/ContextRouter path only — no AWS, no AstroMAE model.

    For sanity-checking the probe logic in cylon_dev before spending on
    Fargate; the RSS numbers here exclude the model, so they are not the
    crossover answer, only a check that insertion and search behave.
    """
    import numpy as np
    from context.manager import ContextManager
    from context.router import ContextRouter, SIMDBackend

    manager = ContextManager(embedding_dim=dimensions, backend="cylon",
                              persist_to_redis=False, snapshot_context_table=False)
    router = ContextRouter(manager, backend=SIMDBackend.NUMPY)
    rng = np.random.default_rng(seed)

    def synthetic_vec():
        v = rng.standard_normal(dimensions).astype(np.float32)
        return v / np.linalg.norm(v)

    inserted = 0
    results = []
    for target in sizes:
        t0 = time.perf_counter()
        for _ in range(target - inserted):
            manager.store_context(workflow_id="probe", task_description="",
                                   embedding=synthetic_vec(), response="x",
                                   cost_metadata={})
        insert_s = time.perf_counter() - t0
        inserted = target
        query_times = []
        for _ in range(num_queries):
            q = synthetic_vec()
            qt0 = time.perf_counter()
            router.find_similar(q, workflow_id="probe")
            query_times.append(time.perf_counter() - qt0)
        row = {"galaxies": inserted, "insert_s_for_step": insert_s,
               "query_s_mean": sum(query_times) / len(query_times),
               "query_s_max": max(query_times)}
        results.append(row)
        print(row, flush=True)
    return results


def _fetch_checkpoint(s3, s3_prefix):
    local_path = "/tmp/embedding_scale_probe_result.json"
    try:
        s3.download_file(RESULTS_BUCKET, s3_prefix + "result.json", local_path)
    except Exception as exc:
        print(f"no checkpoint found at s3://{RESULTS_BUCKET}/{s3_prefix}result.json: {exc}")
        return None
    with open(local_path) as f:
        return json.load(f)


def run_live(sizes, dimensions, num_queries, device, arm_timeout_s):
    comm_name = f"embedding_scale_probe_{uuid.uuid4().hex[:8]}"
    s3_prefix = f"{RESULTS_PREFIX}/{comm_name}/"
    print(f"results will land at s3://{RESULTS_BUCKET}/{s3_prefix}result.json", flush=True)
    overrides = build_overrides(sizes, dimensions, num_queries, s3_prefix, device)
    ecs = boto3.client("ecs", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)
    arns = _run_tasks(ecs, 1, lambda rank: overrides,
                       capacity_retries=6, capacity_backoff_s=30, budget_s=arm_timeout_s)
    print(f"launched task {arns[0]}; waiting up to {arm_timeout_s}s", flush=True)
    try:
        _wait_for_tasks(ecs, arns, arm_timeout_s)
    except Exception as exc:
        # A timed-out or failed wait must not leave the task running and
        # billing with nobody watching it — stop it, then still try to
        # salvage whatever the most recent completed step checkpointed.
        print(f"wait failed ({exc}); stopping task {arns[0]}", flush=True)
        ecs.stop_task(cluster=CLUSTER, task=arns[0],
                       reason="embedding_scale_probe: driver wait failed")
        result = _fetch_checkpoint(s3, s3_prefix)
        if result is not None:
            print("partial result recovered despite wait failure:")
            print(json.dumps(result, indent=1))
        raise

    described = ecs.describe_tasks(cluster=CLUSTER, tasks=arns)
    for task in described.get("tasks", []):
        for container in task.get("containers", []):
            exit_code = container.get("exitCode")
            if exit_code not in (0, None):
                print(f"task container exited non-zero: exitCode={exit_code} "
                      f"stopCode={task.get('stopCode')} reason={task.get('stoppedReason')}",
                      flush=True)

    result = _fetch_checkpoint(s3, s3_prefix)
    if result is None:
        raise RuntimeError(f"task stopped but no checkpoint exists at "
                            f"s3://{RESULTS_BUCKET}/{s3_prefix}result.json")
    if not result.get("complete"):
        print("WARNING: result is PARTIAL — task stopped before the largest "
              "size finished; see the last entry in 'sizes' for how far it got",
              flush=True)
    print(json.dumps(result, indent=1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[50000, 200000, 1000000],
                         help="cumulative embedding-store sizes to measure at, increasing")
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arm-timeout-s", type=int, default=900)
    parser.add_argument("--live", action="store_true",
                         help="launch on the real cosmic Fargate task")
    parser.add_argument("--dry-run-local", action="store_true",
                         help="ContextManager/ContextRouter path only, no AWS/model")
    args = parser.parse_args()

    if args.live == args.dry_run_local:
        parser.error("pass exactly one of --live or --dry-run-local")
    if len(args.sizes) < 1 or any(s <= 0 for s in args.sizes):
        parser.error("--sizes must be positive")
    if any(b <= a for a, b in zip(args.sizes, args.sizes[1:])):
        parser.error("--sizes must be strictly increasing")
    if args.dimensions <= 0:
        parser.error("--dimensions must be positive")
    if args.num_queries < 1:
        parser.error("--num-queries must be at least 1")
    if args.device != "cpu":
        # Matches fargate_cosmic_poc.py's own check — the cosmic Fargate task
        # has no GPU, so a cuda device would fail inside the container, not here.
        parser.error("Fargate has no GPU support — use cpu")

    if args.dry_run_local:
        run_local(args.sizes, args.dimensions, args.num_queries)
    else:
        run_live(args.sizes, args.dimensions, args.num_queries, args.device,
                  args.arm_timeout_s)


if __name__ == "__main__":
    main()