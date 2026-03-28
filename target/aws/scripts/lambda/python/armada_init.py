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

Event payload:
    {
        "workflow_id": "<optional — generated if absent>",
        "tasks": ["task description 1", "task description 2", ...],
        "config": {                       # optional overrides
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

# Allow imports from the shared scripts directory when running inside Lambda.
# In the Docker image the scripts are installed at /opt/python or /var/task;
# locally they live in target/shared/scripts.
_SHARED_SCRIPTS = os.environ.get(
    "SHARED_SCRIPTS_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared", "scripts"),
)
if _SHARED_SCRIPTS not in sys.path:
    sys.path.insert(0, os.path.abspath(_SHARED_SCRIPTS))

from coordinator.agent_coordinator import AgentCoordinator  # noqa: E402
from cost.bedrock_pricing import BedrockConfig  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def handler(event, context):
    """Lambda entry point — prepare tasks for Map state.

    Analogous to cylon_init.lambda_handler: receives the top-level
    workflow payload, fans out into per-task payloads, returns the
    array in 'body' for Step Functions Map iteration.
    """
    logger.info("armada_init invoked: %d tasks", len(event.get("tasks", [])))

    config = BedrockConfig.resolve(payload=event.get("config", {}))
    coordinator = AgentCoordinator(config=config)

    result = coordinator.prepare_tasks(event)

    return {
        "statusCode": 200,
        "body": result["task_payloads"],
        "workflow_id": result["workflow_id"],
        "prepare_cost": result["prepare_cost"],
        "prepare_latency_ms": result["prepare_latency_ms"],
    }