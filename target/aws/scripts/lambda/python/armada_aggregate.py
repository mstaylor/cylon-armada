"""armada_aggregate — aggregate results from Map state and write to S3.

Receives the full array of per-task results from armada_executor,
computes cost/reuse summary, writes result files to S3, and returns
the final workflow output.

File naming follows the cylon_init.py pattern: results_s3_dir and
experiment_name are formatted by armada_init with {scaling}/{world_size}
substitutions and forwarded here by the ASL.

Input (from Step Functions state):
    {
        "workflow_id": "...",
        "task_results": [ <armada_executor result>, ... ],
        "scaling": "weak",
        "world_size": 4,
        "results_s3_dir": "results/lambda/weak/",
        "experiment_name": "lambda_weak_ws4",
        "prepare_cost": { ... },         # from armada_init
    }

Returns:
    {
        "workflow_id": "...",
        "scaling": "...",
        "world_size": ...,
        "experiment_name": "...",
        "results_s3_dir": "...",
        "cost_summary": { ... },
        "reuse_stats": { ... },
        "latency": { ... }
    }
"""

import csv
import io
import json
import logging
import os
import sys
import time

import boto3

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def _write_results_to_s3(
    aggregate: dict,
    task_results: list,
    results_bucket: str,
    results_s3_dir: str,
    experiment_name: str,
) -> None:
    """Write stopwatch CSV, metrics JSON, and summary CSV to S3.

    Follows the same naming convention as armada_ecs_runner so the
    shared results pipeline can aggregate both Lambda and ECS results.
    """
    s3 = boto3.client("s3")
    dir_prefix = results_s3_dir.rstrip("/") + "/" if results_s3_dir else ""

    scaling    = aggregate.get("scaling", "")
    world_size = aggregate.get("world_size", len(task_results))

    cost_summary = aggregate.get("cost_summary", {})
    reuse_stats  = aggregate.get("reuse_stats",  {})
    latency      = aggregate.get("latency",       {})

    total_cost   = cost_summary.get("total_cost",   0.0)
    baseline_cost = cost_summary.get("baseline_cost", 0.0)
    savings_pct  = cost_summary.get("savings_pct",  0.0)
    reuse_rate   = reuse_stats.get("reuse_rate",    0.0)
    cache_hits   = reuse_stats.get("cache_hits",    0)
    llm_calls    = reuse_stats.get("llm_calls",     len(task_results))

    avg_latency_ms = latency.get("avg_ms", 0.0)
    p50            = latency.get("p50_ms", 0.0)
    p95            = latency.get("p95_ms", 0.0)
    p99            = latency.get("p99_ms", 0.0)
    total_ms       = latency.get("total_ms", 0.0)

    # --- stopwatch.csv --------------------------------------------------
    stopwatch_rows = []
    for r in task_results:
        stopwatch_rows.append({
            "experiment_name":   experiment_name,
            "workflow_id":       aggregate.get("workflow_id", ""),
            "rank":              r.get("rank", 0),
            "world_size":        world_size,
            "scaling":           scaling,
            "platform":          "lambda",
            "task_description":  r.get("task_description", ""),
            "source":            r.get("source", ""),
            "search_latency_ms": r.get("search_latency_ms", 0.0),
            "llm_latency_ms":    r.get("llm_latency_ms", 0.0),
            "total_latency_ms":  r.get("total_latency_ms", 0.0),
            "cost_usd":          r.get("cost_usd", 0.0),
            "similarity":        r.get("similarity", 0.0),
            "backend":           r.get("backend", ""),
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
        "experiment_name":          experiment_name,
        "workflow_id":              aggregate.get("workflow_id", ""),
        "platform":                 "lambda",
        "scaling":                  scaling,
        "world_size":               world_size,
        "task_count":               len(task_results),
        "cache_hits":               cache_hits,
        "llm_calls":                llm_calls,
        "reuse_rate":               reuse_rate,
        "total_cost":               total_cost,
        "baseline_cost":            baseline_cost,
        "savings_pct":              savings_pct,
        "total_ms":                 total_ms,
        "avg_latency_ms":           avg_latency_ms,
        "p50_latency_ms":           p50,
        "p95_latency_ms":           p95,
        "p99_latency_ms":           p99,
    }
    s3.put_object(
        Bucket=results_bucket,
        Key=f"{dir_prefix}{experiment_name}_metrics.json",
        Body=json.dumps(metrics, indent=2).encode(),
        ContentType="application/json",
    )

    # --- summary.csv (one row — compatible with pipeline aggregator) ----
    summary = dict(metrics)

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
        "Results written to s3://%s/%s%s_* (tasks=%d reuse=%.1f%% cost=$%.4f)",
        results_bucket, dir_prefix, experiment_name,
        len(task_results), reuse_rate * 100, total_cost,
    )


def handler(event, context):
    """Lambda entry point — aggregate Map state results and write to S3."""
    # --- Shared script imports -------------------------------------------
    # When invoked via lambda_entry.py, S3 scripts are already downloaded
    # and on sys.path.  The baked-in path below is the fallback for direct
    # invocations (local dev, Rivanna, smoke tests).
    _shared = os.environ.get(
        "SHARED_SCRIPTS_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared", "scripts"),
    )
    if _shared not in sys.path:
        sys.path.insert(0, os.path.abspath(_shared))

    from coordinator.agent_coordinator import AgentCoordinator  # noqa: E402
    from cost.bedrock_pricing import BedrockConfig  # noqa: E402

    # --- Main logic -----------------------------------------------------
    workflow_id    = event.get("workflow_id", "")
    scaling        = event.get("scaling", "weak")
    world_size     = int(event.get("world_size", 1))
    results_s3_dir = event.get("results_s3_dir", "")
    experiment_name = event.get("experiment_name", f"lambda_{scaling}_ws{world_size}")
    task_results   = event.get("task_results", [])

    logger.info(
        "armada_aggregate: workflow=%s scaling=%s world_size=%d tasks=%d experiment=%s",
        workflow_id, scaling, world_size, len(task_results), experiment_name,
    )

    config = BedrockConfig.resolve()
    coordinator = AgentCoordinator(config=config)
    aggregate = coordinator.aggregate_results(event)

    # Enrich aggregate with scaling metadata for downstream consumers
    aggregate["scaling"]         = scaling
    aggregate["world_size"]      = world_size
    aggregate["results_s3_dir"]  = results_s3_dir
    aggregate["experiment_name"] = experiment_name

    # --- Write to S3 if bucket is configured ----------------------------
    results_bucket = os.environ.get("RESULTS_BUCKET", "").strip()
    if results_bucket and results_s3_dir:
        _write_results_to_s3(
            aggregate=aggregate,
            task_results=task_results,
            results_bucket=results_bucket,
            results_s3_dir=results_s3_dir,
            experiment_name=experiment_name,
        )
    else:
        logger.info(
            "RESULTS_BUCKET or results_s3_dir not set — skipping S3 write "
            "(bucket=%r dir=%r)", results_bucket, results_s3_dir,
        )

    return aggregate