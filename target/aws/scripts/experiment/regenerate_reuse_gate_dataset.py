#!/usr/bin/env python3
"""Regenerate the real SDSS dataset the reuse-validity-gate analysis needs.

The prior session's cached `real_pred.npy` / `real_embeddings.npy` /
`real_prompts.pkl` (used for docs/Retrieve_Scaling_Findings_2026-09-22.md)
lived in a session scratchpad and no longer exist. This rebuilds the same
inputs from scratch: real AstroMAE inference over the full SDSS partition
(needs torch, so it runs inside the deployed cosmic Fargate task, not
locally), real per-galaxy prompts via cosmic_ai.task_generator, and real
Bedrock Titan embeddings of those prompts (~$0.002 total at Titan embed v2
pricing for ~1,253 short prompts — negligible, but real spend, hence --live).

Output lands at s3://<RESULTS_BUCKET>/<RESULTS_PREFIX>/<run>/:
    predictions.npy, true_redshifts.npy, magnitudes.npy — (N,) / (N, 5)
    embeddings.npy — (N, D) real Bedrock embeddings, one per prompt
    prompts.json — the N prompt strings, index-aligned with the arrays above

Downstream (validate_reuse_gate.py, not yet written) recomputes which
template each galaxy used the same way task_generator.py does — residual vs.
the 90th-percentile outlier threshold, then index parity — rather than this
script persisting it, since that's a pure function of predictions/
true_redshifts already saved and doesn't need a second source of truth.

Usage:
    python regenerate_reuse_gate_dataset.py --live
"""

import argparse
import json

import boto3

from fargate_cosmic_poc import (
    CLUSTER, CONTAINER_NAME, REGION, RESULTS_BUCKET, _run_tasks, _wait_for_tasks,
)

RESULTS_PREFIX = "cylon-armada/results/exp_e_reuse_gate_validation/dataset"

# Runs inside the task via `python -c`, same convention as
# fargate_cosmic_poc.py's own RUNNER_SCRIPT and embedding_scale_probe.py's
# INNER_SCRIPT — no new file needs to exist inside the image.
INNER_SCRIPT = """import json, os
import numpy as np
import boto3

s3 = boto3.client('s3')
bucket = os.environ['ARTIFACT_BUCKET']
model_path = os.environ['ASTROMAE_MODEL_PATH']
model_key = os.environ['ASTROMAE_MODEL_KEY']
data_path = os.environ['ASTROMAE_DATA_PATH']
data_key = os.environ['ASTROMAE_DATA_KEY']
for key, dest in ((model_key, model_path), (data_key, data_path)):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print('downloading', key, '->', dest, flush=True)
        s3.download_file(bucket, key, dest)

device = os.environ.get('INFERENCE_DEVICE', 'cpu')
batch_size = int(os.environ.get('INFERENCE_BATCH_SIZE', '512'))

from cosmic_ai.inference import load_data, load_model, run_inference

print('loading model and full SDSS partition', flush=True)
dataset = load_data(data_path, device=device)
model = load_model(model_path, device=device)
result = run_inference(model, dataset, batch_size=batch_size, device=device)
predictions = result['predictions']
true_redshifts = result['true_redshifts']
magnitudes = result['magnitudes']
n = len(predictions)
print('inference done, n=%d' % n, flush=True)

from cosmic_ai.task_generator import generate_tasks_from_results

# metrics=None so only the n per-galaxy tasks are generated, in galaxy order —
# no trailing batch_summary/cost_analysis tasks to slice off downstream.
prompts = generate_tasks_from_results(
    predictions, true_redshifts, magnitudes,
    metrics=None, max_tasks=None, seed=7, index_offset=0,
)
assert len(prompts) == n, 'expected one prompt per galaxy, got %d for n=%d' % (len(prompts), n)
print('generated %d prompts' % len(prompts), flush=True)

from context.embedding import EmbeddingService

embedder = EmbeddingService()
embeddings = np.zeros((n, embedder.dimensions), dtype=np.float32)
for i, (vec, _meta) in enumerate(embedder.embed_batch(prompts)):
    embeddings[i] = vec
    if (i + 1) % 200 == 0:
        print('embedded %d/%d' % (i + 1, n), flush=True)
print('embedded %d/%d' % (n, n), flush=True)

os.makedirs('/tmp/dataset', exist_ok=True)
np.save('/tmp/dataset/predictions.npy', np.asarray(predictions, dtype=np.float64))
np.save('/tmp/dataset/true_redshifts.npy', np.asarray(true_redshifts, dtype=np.float64))
np.save('/tmp/dataset/magnitudes.npy', np.asarray(magnitudes, dtype=np.float64))
np.save('/tmp/dataset/embeddings.npy', embeddings)
with open('/tmp/dataset/prompts.json', 'w') as f:
    json.dump(prompts, f)

prefix = os.environ['S3_RESULTS_PREFIX']
for name in ('predictions.npy', 'true_redshifts.npy', 'magnitudes.npy',
             'embeddings.npy', 'prompts.json'):
    s3.upload_file('/tmp/dataset/' + name, os.environ['RESULTS_BUCKET'], prefix + name)
    print('uploaded', name, flush=True)
"""


def build_overrides(device, batch_size, s3_prefix):
    environment = [
        {"name": "RESULTS_BUCKET", "value": RESULTS_BUCKET},
        {"name": "S3_RESULTS_PREFIX", "value": s3_prefix},
        {"name": "INFERENCE_DEVICE", "value": device},
        {"name": "INFERENCE_BATCH_SIZE", "value": str(batch_size)},
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


def run_live(device, batch_size, arm_timeout_s):
    s3_prefix = f"{RESULTS_PREFIX}/"
    print(f"dataset will land at s3://{RESULTS_BUCKET}/{s3_prefix}", flush=True)
    overrides = build_overrides(device, batch_size, s3_prefix)
    ecs = boto3.client("ecs", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)
    arns = _run_tasks(ecs, 1, lambda rank: overrides,
                       capacity_retries=6, capacity_backoff_s=30, budget_s=arm_timeout_s)
    print(f"launched task {arns[0]}; waiting up to {arm_timeout_s}s", flush=True)
    try:
        _wait_for_tasks(ecs, arns, arm_timeout_s)
    except Exception as exc:
        print(f"wait failed ({exc}); stopping task {arns[0]}", flush=True)
        ecs.stop_task(cluster=CLUSTER, task=arns[0],
                       reason="regenerate_reuse_gate_dataset: driver wait failed")
        raise

    described = ecs.describe_tasks(cluster=CLUSTER, tasks=arns)
    for task in described.get("tasks", []):
        for container in task.get("containers", []):
            exit_code = container.get("exitCode")
            if exit_code not in (0, None):
                raise RuntimeError(
                    f"task container exited non-zero: exitCode={exit_code} "
                    f"stopCode={task.get('stopCode')} reason={task.get('stoppedReason')}"
                )

    manifest = []
    for name in ("predictions.npy", "true_redshifts.npy", "magnitudes.npy",
                 "embeddings.npy", "prompts.json"):
        key = s3_prefix + name
        s3.head_object(Bucket=RESULTS_BUCKET, Key=key)
        manifest.append(f"s3://{RESULTS_BUCKET}/{key}")
    print(json.dumps({"dataset_files": manifest}, indent=1))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--arm-timeout-s", type=int, default=900)
    parser.add_argument("--live", action="store_true", required=True,
                         help="required — this makes real Bedrock embedding calls")
    args = parser.parse_args()

    if args.device != "cpu":
        parser.error("Fargate has no GPU support — use cpu")

    run_live(args.device, args.batch_size, args.arm_timeout_s)


if __name__ == "__main__":
    main()