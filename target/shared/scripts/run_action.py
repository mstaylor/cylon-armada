"""Action dispatcher — entry point executed by the Lambda S3 script runner.

Downloaded from S3 at runtime along with the rest of the shared scripts.
Reads ACTION and ACTION_PAYLOAD_PATH from environment variables, dispatches
to the appropriate function, and writes the result to ACTION_RESULT_PATH.

Actions:
    prepare_tasks     — embed all tasks, generate per-task payloads
    route_task        — similarity search + reuse or LLM call for one task
    aggregate_results — collect per-task results, compute cost summary
"""

import json
import logging
import os
import sys

# Scripts are downloaded to /tmp preserving S3 key structure.
# Add the scripts root to sys.path so imports like `from context.router import ...` work.
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from coordinator.agent_coordinator import AgentCoordinator, b64_to_ndarray
from context.embedding import EmbeddingService
from context.router import ContextRouter, SIMDBackend
from context.manager import ContextManager
from chain.executor import ChainExecutor
from communicator.fmi_bridge import FMIBridge
from cost.bedrock_pricing import BedrockConfig, BedrockCostTracker

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _load_payload():
    """Load the action payload from the JSON file written by the handler."""
    payload_path = os.environ.get("ACTION_PAYLOAD_PATH", "/tmp/action_payload.json")
    if not os.path.exists(payload_path):
        raise FileNotFoundError(f"Action payload not found at {payload_path}")
    with open(payload_path) as f:
        return json.load(f)


def _write_result(result):
    """Write the action result to a JSON file for the handler to return."""
    result_path = os.environ.get("ACTION_RESULT_PATH", "/tmp/action_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f)
    logger.info("Result written to %s", result_path)


def action_prepare_tasks(payload):
    """Prepare tasks — called by the PrepareTasks Step Functions state."""
    coordinator = AgentCoordinator()
    return coordinator.prepare_tasks(payload)


def action_route_task(payload):
    """Route a single task — called by each Map iteration.

    If FMI is available and world_size > 1, uses the communicator to:
    1. Broadcast context cache from rank 0 (avoids N independent Redis lookups)
    2. Reduce cost metrics back to rank 0 after routing
    """
    config = BedrockConfig.resolve(payload=payload.get("config", {}))
    context_manager = ContextManager(
        redis_host=os.environ.get("REDIS_HOST", "localhost"),
        redis_port=int(os.environ.get("REDIS_PORT", 6379)),
        region=config.region,
    )
    router = ContextRouter(context_manager, config=config)
    chain_executor = ChainExecutor(config=config)
    cost_tracker = BedrockCostTracker.create(region=config.region)

    # Initialize FMI if rank/world_size present
    fmi = FMIBridge.from_payload(payload)

    # If FMI is available, rank 0 loads embeddings and broadcasts to all workers.
    # This avoids each worker independently querying Redis for the full embedding set.
    if fmi.available and fmi.world_size > 1:
        workflow_id = payload["workflow_id"]
        if fmi.rank == 0:
            all_embeddings = context_manager.get_all_embeddings(workflow_id=workflow_id)
            # Serialize embeddings for broadcast
            import base64
            cache_data = [
                (ctx_id, base64.b64encode(emb.tobytes()).decode("ascii"))
                for ctx_id, emb in all_embeddings
            ]
        else:
            cache_data = None

        cache_data = fmi.broadcast_embeddings(cache_data, root=0)
        logger.info("FMI: received %d cached embeddings via broadcast (rank=%d)",
                     len(cache_data), fmi.rank)

        # Pre-populate the context manager's cache from broadcast data
        import base64
        import numpy as np
        for ctx_id, emb_b64 in cache_data:
            emb = np.frombuffer(base64.b64decode(emb_b64), dtype=np.float32)
            context_manager.cache_embedding(ctx_id, emb, workflow_id=workflow_id)

    # Decode embedding from base64
    import numpy as np
    query_embedding = b64_to_ndarray(payload["embedding_b64"])

    # Record the embedding cost from the prepare step
    cost_tracker.record_embedding_call(
        payload["embedding_metadata"]["model_id"],
        payload["embedding_metadata"]["token_count"],
    )

    result = router.route(
        task_description=payload["task_description"],
        query_embedding=query_embedding,
        workflow_id=payload["workflow_id"],
        chain_executor=chain_executor,
        cost_tracker=cost_tracker,
        embedding_metadata=payload["embedding_metadata"],
    )

    # Reduce cost metrics back to rank 0
    if fmi.available and fmi.world_size > 1:
        result["total_cost_reduced"] = fmi.reduce_cost(
            result.get("cost_usd", 0), root=0,
        )
        fmi.finalize()

    return result


def action_aggregate_results(payload):
    """Aggregate results — called by the AggregateResults Step Functions state."""
    coordinator = AgentCoordinator()
    return coordinator.aggregate_results(payload)


def action_model_parallel_stage(payload):
    """Model parallelism — run one ONNX stage and exchange tensors via FMI.

    Replaces Step Functions state passing for inter-stage tensor exchange,
    removing the 256KB payload limit that constrains large batch inference.

    Each Lambda receives its ONNX stage assignment (rank 0 = ViT, rank 1 =
    Inception) via the payload, runs its subgraph, then all-gathers outputs
    via FMI. Rank 0 runs the fusion stage once all outputs are collected.

    Payload:
        {
            "rank": 0,             # 0 = ViT, 1 = Inception
            "world_size": 2,       # number of parallel stages
            "onnx_stage": 0,       # ONNX subgraph index to run
            "onnx_s3_key": "...",  # S3 key of the ONNX partition
            "input_b64": "...",    # base64 float32 input tensor
            "input_shape": [...],  # tensor shape
            "fusion_s3_key": "...", # S3 key of the fusion subgraph (rank 0 only)
            "fmi_channel_type": "direct",  # TCPunch for low-latency tensor exchange
            "fmi_hint": "low_latency",
            "s3_bucket": "...",    # bucket for ONNX subgraph download
            "batch_size": 32,
        }

    Returns (on rank 0 after fusion):
        {
            "prediction": [...],   # model output (e.g., redshift predictions)
            "stage_latency_ms": { "vit": ..., "inception": ..., "fusion": ... },
            "fmi_latency_ms": ..., # allgather time
        }
    """
    import base64
    import time

    import numpy as np

    fmi = FMIBridge.from_payload(payload)
    if not fmi.available:
        raise RuntimeError(
            "FMI not available — model_parallel_stage requires Cylon FMI library"
        )

    rank = fmi.rank
    onnx_stage = int(payload.get("onnx_stage", rank))
    s3_bucket = payload.get("s3_bucket", "")
    onnx_s3_key = payload.get("onnx_s3_key", "")
    batch_size = int(payload.get("batch_size", 1))

    # Decode input tensor
    input_tensor = np.frombuffer(
        base64.b64decode(payload["input_b64"]), dtype=np.float32
    ).reshape(payload["input_shape"])

    # Download and run ONNX subgraph for this stage
    import onnxruntime as ort
    import boto3
    import io

    stage_start = time.monotonic()

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=s3_bucket, Key=onnx_s3_key)
    onnx_bytes = obj["Body"].read()

    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: input_tensor})[0]  # shape varies by stage

    stage_latency_ms = (time.monotonic() - stage_start) * 1000
    logger.info(
        "Stage %d complete: output shape=%s latency=%.1fms",
        onnx_stage, output.shape, stage_latency_ms,
    )

    # All-gather outputs from all stages via FMI direct channel
    allgather_start = time.monotonic()
    all_outputs = fmi.allgather_tensors(output)
    fmi_latency_ms = (time.monotonic() - allgather_start) * 1000

    logger.info(
        "FMI allgather complete: %d tensors, latency=%.1fms",
        len(all_outputs), fmi_latency_ms,
    )

    # Rank 0 runs the fusion stage with the concatenated outputs
    result = {
        "stage_latency_ms": {f"stage_{onnx_stage}": round(stage_latency_ms, 2)},
        "fmi_latency_ms": round(fmi_latency_ms, 2),
    }

    if rank == 0:
        fusion_s3_key = payload.get("fusion_s3_key", "")
        if not fusion_s3_key:
            raise ValueError("fusion_s3_key required on rank 0 for fusion stage")

        fusion_start = time.monotonic()

        obj = s3.get_object(Bucket=s3_bucket, Key=fusion_s3_key)
        fusion_bytes = obj["Body"].read()

        fusion_sess = ort.InferenceSession(
            fusion_bytes, providers=["CPUExecutionProvider"]
        )

        # Concatenate all stage outputs along the feature dimension
        concatenated = np.concatenate(all_outputs, axis=-1)
        fusion_input_name = fusion_sess.get_inputs()[0].name
        prediction = fusion_sess.run(None, {fusion_input_name: concatenated})[0]

        fusion_latency_ms = (time.monotonic() - fusion_start) * 1000
        logger.info(
            "Fusion complete: prediction shape=%s latency=%.1fms",
            prediction.shape, fusion_latency_ms,
        )

        result["prediction"] = prediction.tolist()
        result["stage_latency_ms"]["fusion"] = round(fusion_latency_ms, 2)

    fmi.finalize()
    return result


# Action dispatch table
_ACTIONS = {
    "prepare_tasks":        action_prepare_tasks,
    "route_task":           action_route_task,
    "aggregate_results":    action_aggregate_results,
    "model_parallel_stage": action_model_parallel_stage,
}


def main():
    action = os.environ.get("ACTION", "").strip()
    if not action:
        raise ValueError("ACTION environment variable not set")

    if action not in _ACTIONS:
        raise ValueError(f"Unknown action: {action}. Expected one of: {list(_ACTIONS.keys())}")

    logger.info("Dispatching action: %s", action)
    payload = _load_payload()
    result = _ACTIONS[action](payload)
    _write_result(result)
    logger.info("Action %s completed successfully", action)


if __name__ == "__main__":
    main()