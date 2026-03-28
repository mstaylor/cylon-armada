"""armada_aggregate — aggregate results from Map state.

Receives the full array of per-task results from armada_executor,
computes cost/reuse summary, and returns the final workflow output.

Input (from Step Functions state):
    {
        "workflow_id": "...",
        "task_results": [ <armada_executor result>, ... ],
        "prepare_cost": { ... }   # from armada_init
    }

Returns:
    {
        "workflow_id": "...",
        "results": [ ... ],
        "cost_summary": { ... },
        "reuse_stats": { ... },
        "latency": { ... }
    }
"""

import logging
import os
import sys

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
    """Lambda entry point — aggregate Map state results."""
    workflow_id = event.get("workflow_id", "")
    logger.info("armada_aggregate: workflow=%s tasks=%d", workflow_id, len(event.get("task_results", [])))

    config = BedrockConfig.resolve()
    coordinator = AgentCoordinator(config=config)
    return coordinator.aggregate_results(event)