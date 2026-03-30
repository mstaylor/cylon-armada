"""LlamaIndex baseline runner for cylon-armada comparison experiments.

Implements a standard retrieve-augment-generate (RAG) pipeline:
  - VectorStoreIndex holds prior task results (same role as cylon's context store)
  - Every task ALWAYS calls the LLM — no zero-cost cache hits
  - Same Bedrock embedding model and LLM model as cylon-armada

This baseline directly answers: "why not just use LlamaIndex?"

The comparison shows cylon-armada achieves lower cost and lower latency
because SIMD-accelerated similarity search returns cached responses at
zero LLM cost, while LlamaIndex must call the LLM for every task even
when a near-identical result already exists.

Output format matches AgentCoordinator.run_local() for transparent
integration with the existing results pipeline.
"""

import logging
import time
import uuid
from typing import Optional

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.bedrock import BedrockEmbedding

from cost.bedrock_pricing import BedrockConfig, BedrockCostTracker

logger = logging.getLogger(__name__)


def run_llamaindex_baseline(
    tasks: list[str],
    config: BedrockConfig,
    top_k: int = 3,
    workflow_id: Optional[str] = None,
) -> dict:
    """Run a LlamaIndex RAG pipeline over the task list.

    Processing order mirrors cylon-armada's run_local():
      - Tasks processed sequentially
      - Each result is indexed immediately (available for subsequent retrievals)
      - LLM is called for every task (no threshold-based short-circuit)

    Args:
        tasks:       Task descriptions, same list passed to run_local().
        config:      BedrockConfig (same as cylon-armada run).
        top_k:       Number of prior results to retrieve per task.
        workflow_id: Optional workflow identifier.

    Returns:
        Dict with keys: workflow_id, results, cost_summary, reuse_stats, backend.
        Shape matches AgentCoordinator.run_local() so run_experiment() handles
        it without modification.
    """
    if workflow_id is None:
        workflow_id = str(uuid.uuid4())

    cost_tracker = BedrockCostTracker.create(region=config.region)

    # LlamaIndex embedding — same Bedrock Titan model as cylon-armada
    embed_model = BedrockEmbedding(
        model_name=config.embedding_model_id,
        region_name=config.region,
        additional_kwargs={"dimensions": config.embedding_dimensions},
    )
    Settings.embed_model = embed_model
    Settings.llm = None  # LLM calls go through langchain for cost tracking

    # LLM via langchain for accurate token / cost accounting
    llm = ChatBedrock(
        model_id=config.llm_model_id,
        region_name=config.region,
        model_kwargs={"temperature": 0.0},
    )

    # Empty index — grows as tasks are processed
    index = VectorStoreIndex([])

    results = []

    for task_desc in tasks:
        t_start = time.perf_counter()

        # ------------------------------------------------------------------ #
        # Retrieval phase — LlamaIndex VectorStoreIndex similarity search
        # ------------------------------------------------------------------ #
        search_start = time.perf_counter()
        retrieved_context = ""
        best_score = 0.0

        if results:  # index is non-empty after the first task
            retriever = index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(task_desc)
            if nodes:
                best_score = float(nodes[0].score or 0.0)
                retrieved_context = "\n\n".join(
                    f"Prior task: {n.node.metadata.get('task', '')}\n"
                    f"Prior response: {n.node.text}"
                    for n in nodes
                )

        search_latency_ms = (time.perf_counter() - search_start) * 1000

        # ------------------------------------------------------------------ #
        # Generation phase — ALWAYS calls LLM (no zero-cost reuse)
        # This is the fundamental difference from cylon-armada.
        # ------------------------------------------------------------------ #
        llm_start = time.perf_counter()

        messages = []
        if retrieved_context:
            messages.append(SystemMessage(
                content=(
                    "Use the following related prior results as reference:\n\n"
                    + retrieved_context
                )
            ))
        messages.append(HumanMessage(content=task_desc))

        llm_response = llm.invoke(messages)
        llm_latency_ms = (time.perf_counter() - llm_start) * 1000

        usage = llm_response.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        llm_cost = cost_tracker.record_llm_call(
            config.llm_model_id, input_tokens, output_tokens
        )

        # ------------------------------------------------------------------ #
        # Index this result for future retrievals
        # ------------------------------------------------------------------ #
        node = TextNode(
            text=llm_response.content,
            metadata={"task": task_desc, "workflow_id": workflow_id},
        )
        index.insert(node)

        total_latency_ms = (time.perf_counter() - t_start) * 1000

        results.append({
            "task_description": task_desc,
            "response": llm_response.content,
            "source": "llm",        # always — LlamaIndex has no zero-cost path
            "cost_usd": llm_cost,
            "similarity": round(best_score, 4),
            "search_latency_ms": round(search_latency_ms, 2),
            "llm_latency_ms": round(llm_latency_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "backend": "llamaindex",
            "context_id": None,
        })

        logger.debug(
            "llamaindex task=%d/%d score=%.3f llm_ms=%.0f search_ms=%.0f cost=$%.5f",
            len(results), len(tasks),
            best_score, llm_latency_ms, search_latency_ms, llm_cost,
        )

    cost_summary = cost_tracker.get_summary()

    logger.info(
        "LlamaIndex baseline complete: %d tasks, total_cost=$%.4f, "
        "avg_llm_ms=%.0f, avg_search_ms=%.0f",
        len(results),
        cost_summary.get("total_cost", 0),
        sum(r["llm_latency_ms"] for r in results) / len(results) if results else 0,
        sum(r["search_latency_ms"] for r in results) / len(results) if results else 0,
    )

    return {
        "workflow_id": workflow_id,
        "results": results,
        "cost_summary": cost_summary,
        # reuse_rate=0 and cache_hits=0 by definition — LlamaIndex always calls LLM
        "reuse_stats": {
            "total_tasks": len(results),
            "cache_hits": 0,
            "llm_calls": len(results),
            "reuse_rate": 0.0,
        },
        "baseline": False,
        "backend": "llamaindex",
    }