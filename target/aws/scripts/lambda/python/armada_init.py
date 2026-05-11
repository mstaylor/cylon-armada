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

S3 script loading:
    When invoked via lambda_entry.py (the default), scripts are downloaded
    from S3 at cold-start time.  S3_SCRIPTS_BUCKET and S3_SCRIPTS_PREFIX
    are configured via Lambda env vars (Terraform) or event payload override.

    Deploy script changes in seconds:
        aws s3 sync target/shared/scripts/ s3://<bucket>/scripts/
        aws s3 sync target/aws/scripts/lambda/python/ s3://<bucket>/scripts/lambda/

    When S3_SCRIPTS_BUCKET is unset, the baked-in image scripts are used
    (default for local dev, smoke tests, Rivanna).

Event payload:
    {
        "workflow_id": "<optional — generated if absent>",
        "tasks": ["task description 1", "task description 2", ...],
        "scaling": "weak",               # optional: "weak" | "strong" | "w" | "s"
        "world_size": 1,                 # optional: number of concurrent workers
        "results_s3_dir": "results/lambda/{scaling}/",  # optional; {scaling}/{world_size} substituted
        "experiment_name": "lambda_{scaling}_ws{world_size}",  # optional; same substitutions
        "config": {                          # optional overrides
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
        "scaling": "weak",
        "world_size": 1,
        "results_s3_dir": "results/lambda/weak/",
        "experiment_name": "lambda_weak_ws1",
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
    # --- Scaling / result naming (cylon_init.py pattern) ----------------
    # Accept "w"/"s" abbreviations as well as full words.
    raw_scaling = str(event.get("scaling", "weak")).strip().lower()
    scaling = "strong" if raw_scaling.startswith("s") else "weak"
    world_size = int(event.get("world_size", 1))

    raw_results_s3_dir  = event.get("results_s3_dir",  "results/lambda/{scaling}/")
    raw_experiment_name = event.get("experiment_name", "lambda_{scaling}_ws{world_size}")
    results_s3_dir  = raw_results_s3_dir.format(scaling=scaling, world_size=world_size)
    experiment_name = raw_experiment_name.format(scaling=scaling, world_size=world_size)

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
    logger.info("armada_init invoked: %d tasks", len(event.get("tasks", [])))

    config = BedrockConfig.resolve(payload=event.get("config", {}))
    coordinator = AgentCoordinator(config=config)

    result = coordinator.prepare_tasks(event)

    # FMI channel config — passed through from caller to every executor
    fmi_channel_type = event.get("fmi_channel_type", "redis")
    fmi_hint         = event.get("fmi_hint", "fast")

    # Propagate scaling metadata and FMI config into every per-task payload
    # so armada_executor and armada_aggregate receive them via the Map state.
    # S3 script coordinates are handled by lambda_entry.py via env vars
    # (or event payload override) — no need to propagate per-task.
    # --- Embedding offload to Redis -----------------------------------------
    # Lambda Step Functions payloads are capped at 256KB (input) / 6MB (output).
    # Passing embedding_b64 inline for large task counts (64+ tasks × 1024 dims)
    # can exceed this limit and cause HTTP 413 errors.
    #
    # Non-FMI path: store each embedding in Redis keyed by
    #   embedding:{workflow_id}:{rank}
    # and replace embedding_b64 in the payload with embedding_key.
    # The executor retrieves the embedding from Redis before routing.
    #
    # FMI path (Phase 2): embeddings are broadcast peer-to-peer via TCPunch,
    # eliminating the Redis round-trip entirely. The FMI path is the key
    # research contribution — Redis offload is the non-FMI baseline.
    #
    # This architectural split is documented in docs/ARCHITECTURE_DECISIONS.md.
    redis_host = os.environ.get("REDIS_HOST", "")
    redis_port = int(os.environ.get("REDIS_PORT", 6379))
    workflow_id = result["workflow_id"]
    offload_to_redis = bool(redis_host)

    if offload_to_redis:
        try:
            import redis as _redis
            _rc = _redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
            _ttl = 3600  # 1 hour — embeddings only needed for one experiment run
        except Exception as e:
            logger.warning("Redis unavailable for embedding offload, keeping inline: %s", e)
            offload_to_redis = False

    for i, payload in enumerate(result["task_payloads"]):
        if offload_to_redis and "embedding_b64" in payload:
            emb_key = f"embedding:{workflow_id}:{i}"
            try:
                _rc.setex(emb_key, _ttl, payload["embedding_b64"].encode())
                payload["embedding_key"] = emb_key
                del payload["embedding_b64"]  # remove from SFN payload
            except Exception as e:
                logger.warning("Failed to offload embedding %d to Redis: %s", i, e)
                payload["embedding_key"] = None  # executor will use inline fallback

        payload["scaling"]          = scaling
        payload["world_size"]       = world_size
        payload["rank"]             = i
        payload["results_s3_dir"]   = results_s3_dir
        payload["experiment_name"]  = experiment_name
        payload["fmi_channel_type"] = fmi_channel_type
        payload["fmi_hint"]         = fmi_hint

    return {
        "statusCode": 200,
        "body": result["task_payloads"],
        "workflow_id": result["workflow_id"],
        "scaling":          scaling,
        "world_size":       world_size,
        "results_s3_dir":   results_s3_dir,
        "experiment_name":  experiment_name,
        "fmi_channel_type": fmi_channel_type,
        "fmi_hint":         fmi_hint,
        "prepare_cost": result["prepare_cost"],
        "prepare_latency_ms": result["prepare_latency_ms"],
    }