"""armada_executor — per-task Lambda invoked by the Map state.

Follows the cylon_executor pattern from the cylon paper:
  - Receives a single task payload (one Map item from armada_init's body)
  - Runs the context-reuse routing pipeline for that task
  - Returns the routing result

Infrastructure config (Redis host/port, DynamoDB table) is read from
Lambda environment variables so it stays out of Step Functions history
and can be managed independently of workflow inputs.

FMI integration (when world_size > 1 and FMI is configured):
  - Context Broadcast: rank 0 serializes the context store and broadcasts
    to all workers before routing. Cylon backend uses Arrow IPC (Redis-free);
    Redis backend broadcasts (context_id, embedding_b64) pairs.
  - Progressive Context Sync: after each new LLM call, the executing worker
    broadcasts the new embedding to all peers so subsequent similar tasks in
    the same run can reuse it without a Redis round-trip.

Task payload (each item from armada_init's body array):
    {
        "task_description": "...",
        "embedding_b64": "<base64-encoded float32 array>",
        "embedding_metadata": { "model_id": "...", "token_count": ... },
        "workflow_id": "...",
        "rank": 0,
        "world_size": 4,
        "fmi_channel_type": "redis",   # "redis" | "direct" | "s3"
        "fmi_hint": "fast",            # "fast" | "low_latency"
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

import base64
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def _broadcast_context(fmi, context_manager, workflow_id, context_backend, rank, np):
    """Broadcast context store from rank 0 to all workers via FMI.

    Cylon backend: serializes the full ContextTable as Arrow IPC bytes.
        Rank 0 calls to_ipc(), broadcasts raw bytes; all other ranks call
        load_from_ipc(). This is Redis-free — no ElastiCache required for
        the Cylon context backend when FMI is available.

    Redis backend: rank 0 fetches (context_id, embedding_b64) pairs and
        broadcasts as JSON. All workers pre-populate their in-memory cache,
        avoiding N independent Redis embedding lookups per worker.
    """
    if context_backend == "cylon":
        ipc_data = context_manager.to_ipc() if fmi.rank == 0 else b""
        received = fmi.broadcast_bytes(ipc_data, root=0)
        if fmi.rank != 0 and received:
            try:
                context_manager.load_from_ipc(received)
                logger.info(
                    "FMI: loaded ContextTable from Arrow IPC broadcast (rank=%d)", rank
                )
            except Exception as e:
                logger.warning("FMI: ContextTable IPC load failed: %s", e)
    else:
        if fmi.rank == 0:
            all_embeddings = context_manager.get_all_embeddings(workflow_id=workflow_id)
            cache_data = [
                (ctx_id, base64.b64encode(emb.tobytes()).decode("ascii"))
                for ctx_id, emb in all_embeddings
            ]
        else:
            cache_data = []

        cache_data = fmi.broadcast_embeddings(cache_data, root=0)
        for ctx_id, emb_b64 in cache_data:
            emb = np.frombuffer(base64.b64decode(emb_b64), dtype=np.float32)
            context_manager.cache_embedding(ctx_id, emb, workflow_id=workflow_id)

        logger.info(
            "FMI: pre-populated %d embeddings via broadcast (rank=%d)",
            len(cache_data), rank,
        )


def _sync_new_embedding(fmi, context_manager, result, query_embedding, workflow_id, rank, np):
    """Progressive context sync — push a newly-computed embedding to all peers.

    Called immediately after a new LLM call completes. The executing worker
    broadcasts the (context_id, embedding) pair so all peers can cache it
    and reuse it for any subsequent similar tasks in the same Map execution.

    This is a within-run optimization: it improves reuse rates when Map
    workers process related tasks and one worker's LLM response is useful
    to another. Without this, reuse across workers only happens in the
    next workflow run after Redis/Cylon persistence has been written.
    """
    context_id = result.get("context_id", "")
    if not context_id:
        return

    payload_bytes = json.dumps({
        "context_id": context_id,
        "embedding_b64": base64.b64encode(query_embedding.tobytes()).decode("ascii"),
        "workflow_id": workflow_id,
        "src_rank": rank,
    }).encode("utf-8")

    received = fmi.broadcast_bytes(payload_bytes, root=rank)

    if fmi.rank != rank:
        try:
            data = json.loads(received.decode("utf-8"))
            emb = np.frombuffer(
                base64.b64decode(data["embedding_b64"]), dtype=np.float32
            )
            context_manager.cache_embedding(
                data["context_id"], emb, workflow_id=data.get("workflow_id")
            )
            logger.info(
                "FMI: progressive sync — cached new embedding from rank %d "
                "(context_id=%s, local_rank=%d)",
                data["src_rank"], data["context_id"], rank,
            )
        except Exception as e:
            logger.warning("FMI: progressive sync receive failed: %s", e)


def handler(event, context):
    """Lambda entry point — route a single task.

    Analogous to cylon_executor.lambda_handler: receives one rank's
    payload, executes the workload (here: context-reuse routing),
    returns the result.

    'event' is one item from armada_init's body array, passed directly
    by the Step Functions Map state via $$.Map.Item.Value.
    """
    # --- S3 script loading (optional) -----------------------------------
    s3_scripts_bucket = event.get("s3_scripts_bucket", "")
    s3_scripts_prefix = event.get("s3_scripts_prefix", "scripts/")

    if s3_scripts_bucket:
        import s3_loader
        ok = s3_loader.load_scripts(s3_scripts_bucket, s3_scripts_prefix)
        if not ok:
            logger.warning(
                "armada_executor: S3 script load failed — falling back to baked-in scripts"
            )

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
    from communicator.fmi_bridge import FMIBridge  # noqa: E402

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

    # --- FMI Context Broadcast ------------------------------------------
    # FMIBridge.from_payload reads rank, world_size, fmi_channel_type, and
    # fmi_hint from the task payload. fmi.available is True only when the
    # Cylon FMI library (fmilib) is present in the container — otherwise
    # all FMI calls are no-ops and execution falls back to Redis-only paths.
    fmi = FMIBridge.from_payload(event)

    if fmi.available and world_size > 1:
        _broadcast_context(fmi, context_manager, workflow_id, context_backend, rank, np)

    # Resolve SIMD backend from environment
    raw = os.environ.get("SIMD_BACKEND", "numpy").lower()
    backend = {
        "pycylon": SIMDBackend.PYCYLON,
        "cython":  SIMDBackend.CYTHON_BATCH,
        "numpy":   SIMDBackend.NUMPY,
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

    # --- Progressive Context Sync ---------------------------------------
    # After a new LLM call, immediately broadcast the new embedding to all
    # peer workers. Workers processing later tasks in the same Map batch
    # can reuse it without waiting for the next workflow run.
    if fmi.available and world_size > 1 and result.get("source") == "llm":
        _sync_new_embedding(
            fmi, context_manager, result, embedding, workflow_id, rank, np
        )

    if fmi.available and world_size > 1:
        fmi.finalize()

    result["task_description"] = task_description
    result["rank"] = rank
    result["workflow_id"] = workflow_id

    return result