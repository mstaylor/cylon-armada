"""Agent Coordinator — orchestrates workflows via AWS Step Functions.

Two Lambda entry points triggered by the Step Functions state machine:
1. prepare_tasks — embeds all tasks, generates per-task payloads for Map state
2. aggregate_results — collects per-task results, computes cost summary

The coordinator can also run locally (in-process) for development
without deploying to Step Functions.
"""

import json
import logging
import time
import uuid
from typing import Optional

import boto3
import numpy as np

from context.embedding import EmbeddingService
from context.manager import ContextManager
from context.router import ContextRouter, SIMDBackend
from chain.executor import ChainExecutor
from cost.bedrock_pricing import BedrockConfig, BedrockCostTracker

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """Orchestrate multi-task workflows with Step Functions or locally."""

    def __init__(self, config: Optional[BedrockConfig] = None):
        if config is None:
            config = BedrockConfig.resolve()
        self.config = config

    def prepare_tasks(self, event: dict) -> dict:
        """Step Functions 'Prepare Tasks' state.

        Receives:
            workflow_id, tasks (list of strings), config overrides

        Returns:
            task_payloads — list of dicts for the Map state, each containing:
            {task_description, embedding (base64), workflow_id, config, rank, world_size}
        """
        workflow_id = event.get("workflow_id", str(uuid.uuid4()))
        tasks = event["tasks"]
        config = BedrockConfig.resolve(payload=event.get("config", {}))

        embedding_service = EmbeddingService(config=config)
        cost_tracker = BedrockCostTracker.create(
            region=config.region,
            config_path=event.get("pricing_config"),
        )

        start = time.perf_counter()
        task_payloads = []

        for rank, task_desc in enumerate(tasks):
            embedding, metadata = embedding_service.embed(task_desc)
            cost_tracker.record_embedding_call(
                metadata["model_id"], metadata["token_count"],
            )

            task_payloads.append({
                "task_description": task_desc,
                "embedding_b64": _ndarray_to_b64(embedding),
                "embedding_metadata": metadata,
                "workflow_id": workflow_id,
                "rank": rank,
                "world_size": len(tasks),
                "config": {
                    "llm_model_id": config.llm_model_id,
                    "embedding_model_id": config.embedding_model_id,
                    "embedding_dimensions": config.embedding_dimensions,
                    "similarity_threshold": config.similarity_threshold,
                    "region": config.region,
                },
            })

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Prepared %d tasks for workflow %s (%.1fms)",
            len(tasks), workflow_id, elapsed_ms,
        )

        return {
            "workflow_id": workflow_id,
            "task_payloads": task_payloads,
            "prepare_cost": cost_tracker.get_summary(),
            "prepare_latency_ms": round(elapsed_ms, 2),
        }

    def aggregate_results(self, event: dict) -> dict:
        """Step Functions 'Aggregate Results' state.

        Receives:
            workflow_id, task_results (array of per-task results from Map state),
            prepare_cost (from prepare step)

        Returns:
            {results, cost_summary, reuse_stats}
        """
        workflow_id = event.get("workflow_id", "")
        task_results = event.get("task_results", [])
        prepare_cost = event.get("prepare_cost", {})

        total_cost = 0.0
        cache_hits = 0
        llm_calls = 0
        total_latency = 0.0

        for result in task_results:
            total_cost += result.get("cost_usd", 0)
            total_latency += result.get("total_latency_ms", 0)
            if result.get("source") == "cache":
                cache_hits += 1
            else:
                llm_calls += 1

        reuse_rate = cache_hits / len(task_results) if task_results else 0
        embedding_cost = prepare_cost.get("cost_breakdown", {}).get("embedding", 0)

        return {
            "workflow_id": workflow_id,
            "results": task_results,
            "cost_summary": {
                "total_cost": round(total_cost + embedding_cost, 6),
                "embedding_cost": embedding_cost,
                "llm_cost": round(total_cost, 6),
                "prepare_cost": prepare_cost,
            },
            "reuse_stats": {
                "total_tasks": len(task_results),
                "cache_hits": cache_hits,
                "llm_calls": llm_calls,
                "reuse_rate": round(reuse_rate * 100, 2),
            },
            "latency": {
                "total_ms": round(total_latency, 2),
                "avg_per_task_ms": round(total_latency / len(task_results), 2) if task_results else 0,
            },
        }

    def run_local(
        self,
        tasks: list[str],
        workflow_id: Optional[str] = None,
        backend: SIMDBackend = SIMDBackend.NUMPY,
        dynamo_endpoint_url: Optional[str] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        baseline: bool = False,
    ) -> dict:
        """Run workflow locally (in-process) for development.

        If baseline=True, skips context reuse — all tasks get fresh LLM calls.
        """
        if workflow_id is None:
            workflow_id = str(uuid.uuid4())

        config = self.config
        embedding_service = EmbeddingService(config=config)
        chain_executor = ChainExecutor(config=config)
        context_manager = ContextManager(
            dynamo_endpoint_url=dynamo_endpoint_url,
            redis_host=redis_host,
            redis_port=redis_port,
            region=config.region,
        )
        cost_tracker = BedrockCostTracker.create(region=config.region)

        if baseline:
            # Baseline: skip context reuse, threshold=1.0 means nothing matches
            baseline_config = BedrockConfig(
                llm_model_id=config.llm_model_id,
                embedding_model_id=config.embedding_model_id,
                embedding_dimensions=config.embedding_dimensions,
                similarity_threshold=1.0,
                region=config.region,
            )
            router = ContextRouter(
                context_manager, config=baseline_config,
                backend=backend,
            )
        else:
            router = ContextRouter(
                context_manager, config=config,
                backend=backend,
            )

        results = []
        for task_desc in tasks:
            embedding, metadata = embedding_service.embed(task_desc)
            result = router.route(
                task_description=task_desc,
                query_embedding=embedding,
                workflow_id=workflow_id,
                chain_executor=chain_executor,
                cost_tracker=cost_tracker,
                embedding_metadata=metadata,
            )
            result["task_description"] = task_desc
            results.append(result)

        cache_hits = sum(1 for r in results if r["source"] == "cache")
        reuse_rate = cache_hits / len(results) if results else 0

        return {
            "workflow_id": workflow_id,
            "results": results,
            "cost_summary": cost_tracker.get_summary(),
            "reuse_stats": {
                "total_tasks": len(results),
                "cache_hits": cache_hits,
                "llm_calls": len(results) - cache_hits,
                "reuse_rate": round(reuse_rate * 100, 2),
            },
            "baseline": baseline,
            "backend": backend.value,
        }

    def start_step_functions_workflow(
        self,
        tasks: list[str],
        state_machine_arn: str,
        workflow_id: Optional[str] = None,
        config_overrides: Optional[dict] = None,
    ) -> dict:
        """Start a Step Functions workflow execution.

        Used by the Experiment Runner to trigger workflows on AWS.
        """
        if workflow_id is None:
            workflow_id = str(uuid.uuid4())

        sfn_client = boto3.client("stepfunctions", region_name=self.config.region)

        input_payload = {
            "workflow_id": workflow_id,
            "tasks": tasks,
            "config": config_overrides or {},
        }

        response = sfn_client.start_sync_execution(
            stateMachineArn=state_machine_arn,
            name=f"workflow-{workflow_id}",
            input=json.dumps(input_payload),
        )

        if response["status"] == "SUCCEEDED":
            return json.loads(response["output"])
        else:
            raise RuntimeError(
                f"Step Functions workflow failed: {response['status']} — "
                f"{response.get('error', '')} {response.get('cause', '')}"
            )


def _ndarray_to_b64(arr: np.ndarray) -> str:
    """Encode a numpy array as base64 for JSON-safe transport via Step Functions."""
    import base64
    return base64.b64encode(arr.tobytes()).decode("ascii")


def b64_to_ndarray(b64_str: str, dtype=np.float32) -> np.ndarray:
    """Decode a base64 string back to a numpy array."""
    import base64
    return np.frombuffer(base64.b64decode(b64_str), dtype=dtype)