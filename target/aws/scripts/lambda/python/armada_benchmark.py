"""armada_benchmark — run Experiment A / A2 (zero-copy data plane) in Lambda.

Single-node microbenchmark handler. Unlike armada_init/executor/aggregate,
there is no Map fan-out: one Lambda runs the whole serialization matrix
(arrow_ipc / json / pickle / protobuf / flatbuffers / ... × N × D) under the
10 GB / Firecracker configuration named in the proposal, then writes the same
CSV artifacts the local `exp_a_zerocopy.py --output` run produces — only to S3.

The compute is delegated verbatim to `experiment.exp_a_zerocopy.run(config)`,
so the Lambda measurement and the local `cylon_dev` measurement come from one
code path (Code Change Rule 5 — trace the deployment path; no forked logic).

Input (from the benchmark Step Functions Task state / direct invoke):
    {
        "batch_sizes":     [100, 500, 1000, 5000],   # optional, else defaults
        "dims":            [256, 512, 1024],          # optional
        "warmup":          3,                          # optional
        "reps":            20,                         # optional
        "seed":            42,                         # optional
        "runs":            1,                          # optional (>1 -> run_N/ keys)
        "results_s3_dir":  "results/benchmark/exp_a/",# where CSVs land
        "experiment_name": "lambda_exp_a_10240mb"      # filename stem + poll key
    }

Config resolution (env var > event > default), consistent with the framework:
    RESULTS_BUCKET      — S3 bucket for result CSVs (env, Terraform-managed)

Returns a small JSON summary (row counts + the S3 keys written) for the
Step Functions execution record. The bulk artifacts are the CSVs in S3.
"""

import json
import logging
import os
import sys
import time

import boto3

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def _ensure_shared_scripts_on_path():
    """Put the shared scripts dir on sys.path so `experiment.exp_a_zerocopy`
    imports. When invoked via lambda_entry.py the S3 copy is already on
    sys.path; the baked-in path below is the fallback for direct invokes."""
    _shared = os.environ.get(
        "SHARED_SCRIPTS_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared", "scripts"),
    )
    abs_shared = os.path.abspath(_shared)
    if abs_shared not in sys.path:
        sys.path.insert(0, abs_shared)


def _upload_csv_map(csv_map, results_bucket, results_s3_dir, experiment_name, run_idx, total_runs):
    """Upload each CSV to s3://<bucket>/<dir>/[run_N/]<experiment>_<filename>.

    Naming mirrors armada_aggregate (`{dir}{experiment_name}_{artifact}`) so the
    results pipeline discovers benchmark artifacts the same way it finds sweep
    artifacts. Returns the list of keys written.
    """
    s3 = boto3.client("s3")
    dir_prefix = results_s3_dir.rstrip("/") + "/" if results_s3_dir else ""
    run_seg = "" if total_runs == 1 else f"run_{run_idx}/"
    keys = []
    for filename, text in csv_map.items():
        if not text:
            continue
        key = f"{dir_prefix}{run_seg}{experiment_name}_{filename}"
        s3.put_object(
            Bucket=results_bucket,
            Key=key,
            Body=text.encode(),
            ContentType="text/csv",
        )
        keys.append(key)
        logger.info("Wrote s3://%s/%s", results_bucket, key)
    return keys


def handler(event, context):
    """Lambda entry point — run the Experiment A/A2 matrix and write CSVs to S3."""
    _ensure_shared_scripts_on_path()

    # Import here (not at module top) so lambda_entry's S3 sys.path insert wins.
    from experiment.exp_a_zerocopy import run  # noqa: E402

    # --- Config resolution (event > default); env only for the S3 bucket ----
    batch_sizes = event.get("batch_sizes") or [100, 500, 1000, 5000]
    dims        = event.get("dims")        or [256, 512, 1024]
    warmup      = int(event.get("warmup", 3))
    reps        = int(event.get("reps", 20))
    seed        = int(event.get("seed", 42))
    total_runs  = int(event.get("runs", 1))

    results_s3_dir  = event.get("results_s3_dir", "")
    experiment_name = event.get("experiment_name", "lambda_exp_a")
    results_bucket  = os.environ.get("RESULTS_BUCKET", "").strip()

    # Fail fast at the boundary — a benchmark that can't persist is useless.
    if not results_bucket:
        raise ValueError(
            "RESULTS_BUCKET env var not set — configure it in Terraform "
            "(local.lambda_env). Cannot persist Experiment A results."
        )

    # Memory provenance: the proposal pins this benchmark to a 10 GB Lambda, so
    # record the actual configured size alongside the machine info.
    try:
        mem_mb = int(context.memory_limit_in_mb)
    except Exception:
        mem_mb = None

    logger.info(
        "armada_benchmark: experiment=%s runs=%d batch_sizes=%s dims=%s "
        "warmup=%d reps=%d mem=%sMB -> s3://%s/%s",
        experiment_name, total_runs, batch_sizes, dims, warmup, reps,
        mem_mb, results_bucket, results_s3_dir,
    )

    all_keys = []
    row_counts = []
    t0 = time.time()
    for run_idx in range(1, total_runs + 1):
        config = {
            "batch_sizes": batch_sizes,
            "dims": dims,
            "warmup": warmup,
            "reps": reps,
            "seed": seed + run_idx - 1,
            "meta": {
                "run": run_idx,
                "warmup": warmup,
                "reps": reps,
                "platform": "lambda",
                "lambda_memory_mb": mem_mb,
                "experiment_name": experiment_name,
            },
        }
        csv_map = run(config)
        keys = _upload_csv_map(
            csv_map, results_bucket, results_s3_dir, experiment_name, run_idx, total_runs,
        )
        all_keys.extend(keys)
        # results CSV = one row per (format, N, D); count lines minus header.
        results_csv = csv_map.get("exp_a_zerocopy_results.csv", "")
        row_counts.append(max(results_csv.count("\n") - 1, 0))

    elapsed_s = round(time.time() - t0, 2)
    logger.info(
        "armada_benchmark complete: %d runs, %d CSV(s) in %.2fs",
        total_runs, len(all_keys), elapsed_s,
    )

    return {
        "experiment_name": experiment_name,
        "results_bucket": results_bucket,
        "results_s3_dir": results_s3_dir,
        "runs": total_runs,
        "result_rows_per_run": row_counts,
        "keys_written": all_keys,
        "lambda_memory_mb": mem_mb,
        "elapsed_s": elapsed_s,
    }