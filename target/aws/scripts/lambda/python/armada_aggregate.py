"""armada_aggregate — aggregate results from Map state.

Receives the full array of per-task results from armada_executor,
computes cost/reuse summary, and returns the final workflow output.

Input (from Step Functions state):
    {
        "workflow_id": "...",
        "task_results": [ <armada_executor result>, ... ],
        "prepare_cost": { ... },         # from armada_init
        "s3_scripts_bucket": "<optional>",
        "s3_scripts_prefix": "scripts/"
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def handler(event, context):
    """Lambda entry point — aggregate Map state results."""
    # --- S3 script loading (optional) -----------------------------------
    # S3 coordinates were echoed by armada_init and forwarded by the ASL.
    s3_scripts_bucket = event.get("s3_scripts_bucket", "")
    s3_scripts_prefix = event.get("s3_scripts_prefix", "scripts/")

    if s3_scripts_bucket:
        import s3_loader
        ok = s3_loader.load_scripts(s3_scripts_bucket, s3_scripts_prefix)
        if not ok:
            logger.warning("armada_aggregate: S3 script load failed — falling back to baked-in scripts")

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
    workflow_id = event.get("workflow_id", "")
    logger.info("armada_aggregate: workflow=%s tasks=%d", workflow_id, len(event.get("task_results", [])))

    config = BedrockConfig.resolve()
    coordinator = AgentCoordinator(config=config)
    return coordinator.aggregate_results(event)