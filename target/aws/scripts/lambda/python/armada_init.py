"""armada_init — Step Functions init Lambda for cylon-armada.

Follows the cylon_init.py pattern: embed all tasks, build per-task payloads,
return for Map state. Uses ResultPath: "$" in the SFN definition so this
output REPLACES the full state (dropping the original 'tasks' array).

SFN state-size strategy
-----------------------
256 KB per-state limit is hit at ws16+ when task descriptions pass inline.
Fix: route large data through Redis, keep SFN state tiny.

  task:{workflow_id}:{rank}        — task description (fetched by executor)
  embedding:{workflow_id}:{rank}   — query embedding  (fetched by executor)

Per-task SFN payload: only rank + embedding_key (~110 chars each).
Shared fields (config, world_size, etc.) live at top level and are injected
per-task by Map Parameters in the ASL definition.

FMI path (Phase 2): executor broadcasts embeddings peer-to-peer via TCPunch,
eliminating the Redis embedding round-trip. Task descriptions still come from
Redis since they must reach each worker somehow.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def handler(event, context):
    """Lambda entry point — prepare tasks for Map state."""
    raw_scaling = str(event.get("scaling", "weak")).strip().lower()
    scaling = "strong" if raw_scaling.startswith("s") else "weak"
    world_size = int(event.get("world_size", 1))

    raw_results_s3_dir  = event.get("results_s3_dir",  "results/lambda/{scaling}/")
    raw_experiment_name = event.get("experiment_name", "lambda_{scaling}_ws{world_size}")
    results_s3_dir  = raw_results_s3_dir.format(scaling=scaling, world_size=world_size)
    experiment_name = raw_experiment_name.format(scaling=scaling, world_size=world_size)

    _shared = os.environ.get(
        "SHARED_SCRIPTS_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared", "scripts"),
    )
    if _shared not in sys.path:
        sys.path.insert(0, os.path.abspath(_shared))

    from coordinator.agent_coordinator import AgentCoordinator  # noqa: E402
    from cost.bedrock_pricing import BedrockConfig  # noqa: E402

    logger.info(
        "armada_init: %d tasks, world_size=%d",
        len(event.get("tasks", [])), world_size,
    )

    config = BedrockConfig.resolve(payload=event.get("config", {}))
    coordinator = AgentCoordinator(config=config)
    result = coordinator.prepare_tasks(event)

    fmi_channel_type = event.get("fmi_channel_type", "redis")
    fmi_hint         = event.get("fmi_hint", "fast")
    workflow_id      = result["workflow_id"]

    # --- Redis client -------------------------------------------------------
    redis_host = os.environ.get("REDIS_HOST", "")
    redis_port = int(os.environ.get("REDIS_PORT", 6379))
    _rc        = None

    if redis_host:
        try:
            import redis as _redis
            _rc = _redis.Redis(
                host=redis_host, port=redis_port,
                decode_responses=False, socket_connect_timeout=2,
            )
            _rc.ping()
        except Exception as e:
            logger.warning("Redis unavailable: %s", e)
            _rc = None

    _ttl = 3600  # 1 hour

    # --- Build minimal per-task payloads ------------------------------------
    # Only rank + embedding_key + embedding_metadata per task (~110 chars).
    # Task descriptions stored in Redis so they never pass through SFN state.
    minimal_body = []
    for i, payload in enumerate(result["task_payloads"]):
        task_description   = payload.get("task_description", "")
        embedding_metadata = payload.get("embedding_metadata", {})

        if _rc is not None:
            try:
                _rc.setex(f"task:{workflow_id}:{i}", _ttl, task_description.encode())
            except Exception as e:
                logger.warning("task store %d failed: %s", i, e)

        embedding_key = None
        if _rc is not None and "embedding_b64" in payload:
            try:
                emb_key = f"embedding:{workflow_id}:{i}"
                _rc.setex(emb_key, _ttl, payload["embedding_b64"].encode())
                embedding_key = emb_key
            except Exception as e:
                logger.warning("embedding offload %d failed: %s", i, e)

        entry = {"rank": i, "embedding_metadata": embedding_metadata}
        if embedding_key:
            entry["embedding_key"] = embedding_key
        elif "embedding_b64" in payload:
            entry["embedding_b64"] = payload["embedding_b64"]

        minimal_body.append(entry)

    # Config serialised once at top level; injected per-task via Map Parameters
    config_dict = {
        "llm_model_id":         config.llm_model_id,
        "embedding_model_id":   config.embedding_model_id,
        "embedding_dimensions": config.embedding_dimensions,
        "similarity_threshold": config.similarity_threshold,
        "region":               config.region,
    }

    logger.info("armada_init done: %d payloads, redis=%s", len(minimal_body), _rc is not None)

    # ResultPath: "$" in the ASL replaces the full SFN state with this value,
    # dropping the original 'tasks' array and preventing state accumulation.
    return {
        "statusCode":         200,
        "body":               minimal_body,
        "workflow_id":        workflow_id,
        "scaling":            scaling,
        "world_size":         world_size,
        "config":             config_dict,
        "results_s3_dir":     results_s3_dir,
        "experiment_name":    experiment_name,
        "fmi_channel_type":   fmi_channel_type,
        "fmi_hint":           fmi_hint,
        "prepare_cost":       result["prepare_cost"],
        "prepare_latency_ms": result["prepare_latency_ms"],
    }