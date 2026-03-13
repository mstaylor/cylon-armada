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
    """Route a single task — called by each Map iteration."""
    config = BedrockConfig.resolve(payload=payload.get("config", {}))
    context_manager = ContextManager(
        redis_host=os.environ.get("REDIS_HOST", "localhost"),
        redis_port=int(os.environ.get("REDIS_PORT", 6379)),
        region=config.region,
    )
    router = ContextRouter(context_manager, config=config)
    chain_executor = ChainExecutor(config=config)
    cost_tracker = BedrockCostTracker.create(region=config.region)

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

    return result


def action_aggregate_results(payload):
    """Aggregate results — called by the AggregateResults Step Functions state."""
    coordinator = AgentCoordinator()
    return coordinator.aggregate_results(payload)


# Action dispatch table
_ACTIONS = {
    "prepare_tasks": action_prepare_tasks,
    "route_task": action_route_task,
    "aggregate_results": action_aggregate_results,
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