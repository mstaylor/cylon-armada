"""armada_executor — per-task Lambda invoked by the Map state.

Follows the cylon_executor pattern from the cylon paper:
  - Receives a single task payload (one Map item from armada_init's body)
  - Runs the context-reuse routing pipeline for that task
  - Returns the routing result

Infrastructure config (Redis host/port, DynamoDB table) is read from
Lambda environment variables so it stays out of Step Functions history
and can be managed independently of workflow inputs.

Task payload (each item from armada_init's body array):
    {
        "task_description": "...",
        "embedding_b64": "<base64-encoded float32 array>",
        "embedding_metadata": { "model_id": "...", "token_count": ... },
        "workflow_id": "...",
        "rank": 0,
        "world_size": 4,
        "s3_scripts_bucket": "<optional>",
        "s3_scripts_prefix": "scripts/",
        "config": {
            "llm_model_id": "...",
            "embedding_model_id": "...",
            "embedding_dimensions": 256,
            "similarity_threshold": 0.8,
            "region": "us-east-1"
        }
    }

Returns:
    {
        "response": "...",
        "source": "cache" | "llm",
        "cost_usd": ...,
        "similarity": ...,
        "search_latency_ms": ...,
        "llm_latency_ms": ...,
        "total_latency_ms": ...,
        "backend": "...",
        "context_id": "...",
        "task_description": "...",
        "rank": ...,
        "workflow_id": "..."
    }
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def handler(event, context):
    """Lambda entry point — route a single task.

    Analogous to cylon_executor.lambda_handler: receives one rank's
    payload, executes the workload (here: context-reuse routing),
    returns the result.

    'event' is one item from armada_init's body array, passed directly
    by the Step Functions Map state via $$.Map.Item.Value.
    """
    # --- S3 script loading (optional) -----------------------------------
    # S3 coordinates were propagated from armada_init into this payload.
    s3_scripts_bucket = event.get("s3_scripts_bucket", "")
    s3_scripts_prefix = event.get("s3_scripts_prefix", "scripts/")

    if s3_scripts_bucket:
        import s3_loader
        ok = s3_loader.load_scripts(s3_scripts_bucket, s3_scripts_prefix)
        if not ok:
            logger.warning("armada_executor: S3 script load failed — falling back to baked-in scripts")

    # --- Shared script imports (lazy so S3 path is on sys.path first) ---
    _shared = os.environ.get(
        "SHARED_SCRIPTS_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared", "scripts"),
    )
    if _shared not in sys.path:
        sys.path.insert(0, os.path.abspath(_shared))

    import numpy as np
    from context.manager import ContextManager  # noqa: E402
    from context.router import ContextRouter, SIMDBackend  # noqa: E402
    from chain.executor import ChainExecutor  # noqa: E402
    from cost.bedrock_pricing import BedrockConfig, BedrockCostTracker  # noqa: E402

    # --- Main logic -----------------------------------------------------
    task_description = event["task_description"]
    workflow_id = event["workflow_id"]
    rank = event.get("rank", 0)
    world_size = event.get("world_size", 1)

    logger.info(
        "armada_executor rank=%d/%d workflow=%s",
        rank, world_size, workflow_id,
    )

    # Decode embedding from base64 (encoded by armada_init)
    import base64
    embedding = np.frombuffer(
        base64.b64decode(event["embedding_b64"]), dtype=np.float32
    )
    embedding_metadata = event["embedding_metadata"]

    # Resolve config — task payload carries model/threshold; infra from env
    task_config = event.get("config", {})
    config = BedrockConfig.resolve(payload=task_config)

    # Infrastructure from Lambda environment variables
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", 6379))
    dynamo_table = os.environ.get("DYNAMO_TABLE_NAME")  # None → DynamoDB disabled
    dynamo_endpoint_url = os.environ.get("DYNAMO_ENDPOINT_URL")
    context_backend = os.environ.get("CONTEXT_BACKEND", "redis")

    context_manager = ContextManager(
        dynamo_table=dynamo_table,
        dynamo_endpoint_url=dynamo_endpoint_url,
        redis_host=redis_host,
        redis_port=redis_port,
        region=config.region,
        embedding_dim=config.embedding_dimensions,
        backend=context_backend,
    )

    # Resolve SIMD backend from environment
    raw = os.environ.get("SIMD_BACKEND", "numpy").lower()
    backend = {
        "pycylon": SIMDBackend.PYCYLON,
        "cython": SIMDBackend.CYTHON_BATCH,
        "numpy": SIMDBackend.NUMPY,
    }.get(raw, SIMDBackend.NUMPY)

    chain_executor = ChainExecutor(config=config)
    cost_tracker = BedrockCostTracker.create(region=config.region)
    router = ContextRouter(context_manager, config=config, backend=backend)

    result = router.route(
        task_description=task_description,
        query_embedding=embedding,
        workflow_id=workflow_id,
        chain_executor=chain_executor,
        cost_tracker=cost_tracker,
        embedding_metadata=embedding_metadata,
    )

    result["task_description"] = task_description
    result["rank"] = rank
    result["workflow_id"] = workflow_id

    return result