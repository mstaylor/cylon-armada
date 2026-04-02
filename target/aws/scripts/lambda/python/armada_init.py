"""armada_init — Step Functions init Lambda for cylon-armada.

Follows the cylon_init.py pattern from the cylon paper:
  1. Receives the workflow input payload
  2. Embeds all task descriptions (Bedrock)
  3. Builds one payload per task for the Map state
  4. Returns {statusCode: 200, body: [per-task payloads]}

The Step Functions Map state iterates over $.body, invoking
armada_executor once per task (analogous to cylon_executor per rank).

Infrastructure config (Redis, DynamoDB) is read from Lambda environment
variables — not embedded in the payload — so it stays out of Step
Functions execution history.

S3 script loading (optional):
    Pass "s3_scripts_bucket" (and optionally "s3_scripts_prefix") in the
    event payload to have each Lambda download the shared scripts folder
    from S3 at cold-start time instead of using the baked-in image copy.
    armada_init propagates these fields into every per-task payload so
    armada_executor and armada_aggregate can call s3_loader.load_scripts()
    the same way.

    Deploy script changes in seconds:
        aws s3 sync target/shared/scripts/ s3://<bucket>/scripts/

    Omit "s3_scripts_bucket" to use the scripts baked into the image
    (default for local dev, smoke tests, Rivanna).

Event payload:
    {
        "workflow_id": "<optional — generated if absent>",
        "tasks": ["task description 1", "task description 2", ...],
        "s3_scripts_bucket": "<optional>",   # enables S3 script loading
        "s3_scripts_prefix": "scripts/",     # optional, default "scripts/"
        "config": {                           # optional overrides
            "llm_model_id": "...",
            "embedding_model_id": "...",
            "embedding_dimensions": 256,
            "similarity_threshold": 0.8
        }
    }

Returns:
    {
        "statusCode": 200,
        "body": [ <task_payload>, ... ],   # Map state iterates this
        "workflow_id": "...",
        "prepare_cost": { ... },
        "prepare_latency_ms": ...
    }
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def handler(event, context):
    """Lambda entry point — prepare tasks for Map state.

    Analogous to cylon_init.lambda_handler: receives the top-level
    workflow payload, fans out into per-task payloads, returns the
    array in 'body' for Step Functions Map iteration.
    """
    # --- S3 script loading (optional) -----------------------------------
    # Read S3 coordinates from the event payload (cylon_init.py pattern).
    # If present, download the shared scripts folder before importing them.
    s3_scripts_bucket = event.get("s3_scripts_bucket", "")
    s3_scripts_prefix = event.get("s3_scripts_prefix", "scripts/")

    if s3_scripts_bucket:
        import s3_loader
        ok = s3_loader.load_scripts(s3_scripts_bucket, s3_scripts_prefix)
        if not ok:
            logger.warning("armada_init: S3 script load failed — falling back to baked-in scripts")

    # --- Shared script imports (lazy so S3 path is on sys.path first) ---
    _shared = os.environ.get(
        "SHARED_SCRIPTS_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared", "scripts"),
    )
    if _shared not in sys.path:
        sys.path.insert(0, os.path.abspath(_shared))

    from coordinator.agent_coordinator import AgentCoordinator  # noqa: E402
    from cost.bedrock_pricing import BedrockConfig  # noqa: E402

    # --- Main logic -----------------------------------------------------
    logger.info("armada_init invoked: %d tasks", len(event.get("tasks", [])))

    config = BedrockConfig.resolve(payload=event.get("config", {}))
    coordinator = AgentCoordinator(config=config)

    result = coordinator.prepare_tasks(event)

    # Propagate S3 script coordinates into every per-task payload so
    # armada_executor and armada_aggregate can load the same scripts.
    for payload in result["task_payloads"]:
        if s3_scripts_bucket:
            payload["s3_scripts_bucket"] = s3_scripts_bucket
            payload["s3_scripts_prefix"] = s3_scripts_prefix

    return {
        "statusCode": 200,
        "body": result["task_payloads"],
        "workflow_id": result["workflow_id"],
        "prepare_cost": result["prepare_cost"],
        "prepare_latency_ms": result["prepare_latency_ms"],
        # Echo S3 script coords so AggregateResults can forward them to
        # armada_aggregate without an extra path in the execution input.
        "s3_scripts_bucket": s3_scripts_bucket,
        "s3_scripts_prefix": s3_scripts_prefix,
    }