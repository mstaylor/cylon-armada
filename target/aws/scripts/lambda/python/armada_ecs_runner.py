"""armada_ecs_runner — ECS / Rivanna entry point for cylon-armada experiments.

Mirrors the original cylon experiment pattern: reads all configuration
from environment variables (injected per-run by Step Functions
ContainerOverrides or Slurm --export), runs the full experiment, writes
results to S3, and exits with code 0 on success.

Scaling modes (set via SCALING env var):
    weak:   each worker thread gets its own equal slice of the task list.
            Total work grows proportionally with world_size — measures
            throughput at scale.
    strong: all worker threads process the same fixed task list.
            Same total work done faster with more workers — measures speedup.

File naming follows the cylon_init.py pattern: RESULTS_DIR and
EXPERIMENT_NAME are format strings that accept {scaling} and {world_size}
substitutions, giving the Step Functions caller full control over the
output location and file naming convention.

Results written to S3:
    s3://<RESULTS_BUCKET>/<RESULTS_DIR>/<EXPERIMENT_NAME>_stopwatch.csv
    s3://<RESULTS_BUCKET>/<RESULTS_DIR>/<EXPERIMENT_NAME>_metrics.json
    s3://<RESULTS_BUCKET>/<RESULTS_DIR>/<EXPERIMENT_NAME>_summary.csv

Environment variables (static — from ECS task definition or Slurm environment):
    REDIS_HOST, REDIS_PORT, DYNAMO_TABLE_NAME, RESULTS_BUCKET
    BEDROCK_LLM_MODEL_ID, BEDROCK_EMBEDDING_MODEL_ID
    BEDROCK_EMBEDDING_DIMENSIONS, SIMILARITY_THRESHOLD
    CONTEXT_BACKEND, AWS_DEFAULT_REGION

Environment variables (dynamic — injected per-run by Step Functions or sbatch --export):
    WORKFLOW_ID          unique identifier for this run
    TASKS_JSON           JSON array of task description strings
    SCALING              "weak" or "strong"
    WORLD_SIZE           number of parallel worker threads
    RESULTS_DIR          S3 key directory prefix; supports {scaling} placeholder
                         e.g. "results/ecs-fargate/{scaling}/"
    EXPERIMENT_NAME      File name stem; supports {scaling} and {world_size}
                         e.g. "ecs_fargate_{scaling}_ws{world_size}"
    COMPUTE_PLATFORM     "ecs-fargate", "ecs-ec2", or "rivanna" (metadata only)
    S3_SCRIPTS_BUCKET    optional — bucket for hot-loaded scripts
    S3_SCRIPTS_PREFIX    optional — prefix for hot-loaded scripts
"""

import csv
import io
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import boto3

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


def _load_scripts() -> None:
    """Hot-load shared scripts from S3 if S3_SCRIPTS_BUCKET is set."""
    s3_bucket = os.environ.get("S3_SCRIPTS_BUCKET", "").strip()
    s3_prefix = os.environ.get("S3_SCRIPTS_PREFIX", "scripts/").strip()
    if s3_bucket:
        import s3_loader
        ok = s3_loader.load_scripts(s3_bucket, s3_prefix)
        if not ok:
            logger.warning("S3 script load failed — using baked-in scripts")


def _setup_path() -> None:
    """Prepend shared scripts directory to sys.path."""
    shared = os.environ.get(
        "SHARED_SCRIPTS_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared", "scripts"),
    )
    abs_shared = os.path.abspath(shared)
    if abs_shared not in sys.path:
        sys.path.insert(0, abs_shared)


def _route_task(router, chain_executor, cost_tracker, task: str,
                query_embedding, workflow_id: str, embedding_metadata: dict,
                rank: int) -> dict:
    """Route a single task — called from each worker thread."""
    result = router.route(
        task_description=task,
        query_embedding=query_embedding,
        workflow_id=workflow_id,
        chain_executor=chain_executor,
        cost_tracker=cost_tracker,
        embedding_metadata=embedding_metadata,
    )
    result["task_description"] = task
    result["rank"] = rank
    result["workflow_id"] = workflow_id
    return result


def _write_results_to_s3(
    results: List[dict],
    workflow_id: str,
    scaling: str,
    world_size: int,
    platform: str,
    total_elapsed_ms: float,
    results_bucket: str,
    results_dir: str,
    experiment_name: str,
) -> None:
    """Write stopwatch CSV, metrics JSON, and summary CSV to S3.

    Files are written as:
        s3://{results_bucket}/{results_dir}/{experiment_name}_stopwatch.csv
        s3://{results_bucket}/{results_dir}/{experiment_name}_metrics.json
        s3://{results_bucket}/{results_dir}/{experiment_name}_summary.csv
    """
    s3 = boto3.client("s3")
    dir_prefix = results_dir.rstrip("/") + "/" if results_dir else ""

    cache_hits = sum(1 for r in results if r.get("source") == "cache")
    llm_calls = len(results) - cache_hits
    reuse_rate = cache_hits / len(results) if results else 0.0
    total_cost = sum(r.get("cost_usd", 0.0) for r in results)
    latencies = [r.get("total_latency_ms", 0.0) for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)] if latencies_sorted else 0.0
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else 0.0
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)] if latencies_sorted else 0.0
    throughput = len(results) / (total_elapsed_ms / 1000) if total_elapsed_ms > 0 else 0.0

    # --- stopwatch.csv --------------------------------------------------
    stopwatch_rows = []
    for r in results:
        stopwatch_rows.append({
            "workflow_id":        workflow_id,
            "rank":               r.get("rank", 0),
            "world_size":         world_size,
            "scaling":            scaling,
            "platform":           platform,
            "task_description":   r.get("task_description", ""),
            "source":             r.get("source", ""),
            "search_latency_ms":  r.get("search_latency_ms", 0.0),
            "llm_latency_ms":     r.get("llm_latency_ms", 0.0),
            "total_latency_ms":   r.get("total_latency_ms", 0.0),
            "cost_usd":           r.get("cost_usd", 0.0),
            "similarity":         r.get("similarity", 0.0),
            "backend":            r.get("backend", ""),
        })

    buf = io.StringIO()
    if stopwatch_rows:
        writer = csv.DictWriter(buf, fieldnames=list(stopwatch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stopwatch_rows)
    s3.put_object(
        Bucket=results_bucket,
        Key=f"{dir_prefix}{experiment_name}_stopwatch.csv",
        Body=buf.getvalue().encode(),
        ContentType="text/csv",
    )

    # --- metrics.json ---------------------------------------------------
    metrics = {
        "workflow_id":              workflow_id,
        "platform":                 platform,
        "scaling":                  scaling,
        "world_size":               world_size,
        "task_count":               len(results),
        "cache_hits":               cache_hits,
        "llm_calls":                llm_calls,
        "reuse_rate":               reuse_rate,
        "total_cost":               total_cost,
        "baseline_cost":            0.0,
        "savings_pct":              0.0,
        "total_ms":                 total_elapsed_ms,
        "avg_latency_ms":           avg_latency,
        "p50_latency_ms":           p50,
        "p95_latency_ms":           p95,
        "p99_latency_ms":           p99,
        "throughput_tasks_per_sec": throughput,
    }
    s3.put_object(
        Bucket=results_bucket,
        Key=f"{dir_prefix}{experiment_name}_metrics.json",
        Body=json.dumps(metrics, indent=2).encode(),
        ContentType="application/json",
    )

    # --- summary.csv (one row — for the results aggregator) -------------
    summary = dict(metrics)
    summary["experiment_name"] = experiment_name

    summary_buf = io.StringIO()
    writer = csv.DictWriter(summary_buf, fieldnames=list(summary.keys()))
    writer.writeheader()
    writer.writerow(summary)
    s3.put_object(
        Bucket=results_bucket,
        Key=f"{dir_prefix}{experiment_name}_summary.csv",
        Body=summary_buf.getvalue().encode(),
        ContentType="text/csv",
    )

    logger.info(
        "Results written to s3://%s/%s%s_* (tasks=%d reuse=%.1f%% cost=$%.4f elapsed=%.0fms)",
        results_bucket, dir_prefix, experiment_name, len(results),
        reuse_rate * 100, total_cost, total_elapsed_ms,
    )


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Hot-load scripts from S3 (if configured), then set up sys.path  #
    # ------------------------------------------------------------------ #
    _load_scripts()
    _setup_path()

    # Lazy imports — must come after sys.path is configured
    import numpy as np
    from context.manager import ContextManager
    from context.router import ContextRouter, SIMDBackend
    from chain.executor import ChainExecutor
    from cost.bedrock_pricing import BedrockConfig, BedrockCostTracker

    # ------------------------------------------------------------------ #
    # 2. Read per-run config from environment                             #
    # ------------------------------------------------------------------ #
    workflow_id = os.environ.get("WORKFLOW_ID", "").strip() or str(uuid.uuid4())
    tasks_json  = os.environ.get("TASKS_JSON", "[]").strip()
    scaling     = os.environ.get("SCALING", "weak").strip().lower()
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    platform    = os.environ.get("COMPUTE_PLATFORM", "ecs").strip()

    results_bucket = os.environ.get("RESULTS_BUCKET", "").strip()

    # --- File naming (cylon_init.py pattern) ----------------------------
    # RESULTS_DIR and EXPERIMENT_NAME are format strings — {scaling} and
    # {world_size} are substituted at runtime, giving the caller full
    # control over the output location without hardcoding values.
    scaling_str = "strong" if scaling.startswith("s") else "weak"
    scaling = scaling_str  # normalise "s"/"w" to full word

    raw_results_dir     = os.environ.get("RESULTS_DIR", "results/{scaling}/").strip()
    raw_experiment_name = os.environ.get(
        "EXPERIMENT_NAME", f"{platform}_{{scaling}}_ws{{world_size}}"
    ).strip()

    results_dir     = raw_results_dir.format(scaling=scaling, world_size=world_size)
    experiment_name = raw_experiment_name.format(scaling=scaling, world_size=world_size)

    tasks: List[str] = json.loads(tasks_json)
    if not tasks:
        logger.error("TASKS_JSON is empty — nothing to run")
        sys.exit(1)
    if not results_bucket:
        logger.error("RESULTS_BUCKET is not set")
        sys.exit(1)

    logger.info(
        "armada_ecs_runner: workflow=%s platform=%s scaling=%s world_size=%d tasks=%d "
        "experiment=%s dir=%s",
        workflow_id, platform, scaling, world_size, len(tasks),
        experiment_name, results_dir,
    )

    # ------------------------------------------------------------------ #
    # 3. Build shared infrastructure objects                             #
    # ------------------------------------------------------------------ #
    config = BedrockConfig.resolve()

    redis_host      = os.environ.get("REDIS_HOST", "localhost")
    redis_port      = int(os.environ.get("REDIS_PORT", 6379))
    dynamo_table    = os.environ.get("DYNAMO_TABLE_NAME")
    context_backend = os.environ.get("CONTEXT_BACKEND", "redis")

    context_manager = ContextManager(
        dynamo_table=dynamo_table,
        redis_host=redis_host,
        redis_port=redis_port,
        region=config.region,
        embedding_dim=config.embedding_dimensions,
        backend=context_backend,
    )

    raw_backend = os.environ.get("SIMD_BACKEND", "numpy").lower()
    simd_backend = {
        "pycylon": SIMDBackend.PYCYLON,
        "cython":  SIMDBackend.CYTHON_BATCH,
        "numpy":   SIMDBackend.NUMPY,
    }.get(raw_backend, SIMDBackend.NUMPY)

    # Pre-embed all tasks using Bedrock (one batch call via BedrockConfig)
    bedrock = boto3.client("bedrock-runtime", region_name=config.region)
    embedding_metadata = {
        "model_id":    config.embedding_model_id,
        "dimensions":  config.embedding_dimensions,
    }

    logger.info("Embedding %d tasks...", len(tasks))
    embeddings = []
    total_tokens = 0
    for task in tasks:
        response = bedrock.invoke_model(
            modelId=config.embedding_model_id,
            body=json.dumps({"inputText": task, "dimensions": config.embedding_dimensions}),
        )
        body = json.loads(response["body"].read())
        embeddings.append(np.array(body["embedding"], dtype=np.float32))
        total_tokens += body.get("inputTextTokenCount", 0)
    embedding_metadata["token_count"] = total_tokens

    # ------------------------------------------------------------------ #
    # 4. Assign tasks to workers                                         #
    #                                                                    #
    # weak:   each of the world_size workers gets an equal share of the  #
    #         full task list (ceiling division).                         #
    # strong: same total task list; workers run tasks in parallel and    #
    #         results are collected — fixed work, more workers → speedup #
    # ------------------------------------------------------------------ #
    def chunk(lst, n):
        size = max(1, -(-len(lst) // n))  # ceiling division
        return [lst[i:i + size] for i in range(0, len(lst), size)]

    embedding_pairs = list(zip(tasks, embeddings))

    if scaling == "weak":
        # Each worker gets its own slice — total work scales with world_size
        worker_assignments = chunk(embedding_pairs, world_size)
    else:
        # Strong: all workers share the same task list (parallel execution)
        worker_assignments = [embedding_pairs] * world_size

    # ------------------------------------------------------------------ #
    # 5. Run experiment — one thread per worker                          #
    # ------------------------------------------------------------------ #
    all_results = []
    experiment_start = time.monotonic()

    def run_worker(worker_id: int, pairs: list) -> List[dict]:
        chain_executor = ChainExecutor(config=config)
        cost_tracker   = BedrockCostTracker.create(region=config.region)
        router         = ContextRouter(context_manager, config=config, backend=simd_backend)
        worker_results = []
        for task, embedding in pairs:
            result = _route_task(
                router, chain_executor, cost_tracker,
                task, embedding, workflow_id, embedding_metadata, worker_id,
            )
            worker_results.append(result)
        return worker_results

    with ThreadPoolExecutor(max_workers=world_size) as pool:
        futures = {
            pool.submit(run_worker, wid, pairs): wid
            for wid, pairs in enumerate(worker_assignments)
            if pairs
        }
        for future in as_completed(futures):
            wid = futures[future]
            try:
                worker_results = future.result()
                all_results.extend(worker_results)
                logger.info("Worker %d completed %d tasks", wid, len(worker_results))
            except Exception as exc:
                logger.error("Worker %d failed: %s", wid, exc)
                raise

    total_elapsed_ms = (time.monotonic() - experiment_start) * 1000
    logger.info(
        "Experiment complete: %d tasks in %.0fms (world_size=%d scaling=%s)",
        len(all_results), total_elapsed_ms, world_size, scaling,
    )

    # ------------------------------------------------------------------ #
    # 6. Write results to S3                                             #
    # ------------------------------------------------------------------ #
    _write_results_to_s3(
        results=all_results,
        workflow_id=workflow_id,
        scaling=scaling,
        world_size=world_size,
        platform=platform,
        total_elapsed_ms=total_elapsed_ms,
        results_bucket=results_bucket,
        results_dir=results_dir,
        experiment_name=experiment_name,
    )


if __name__ == "__main__":
    main()