"""Context Router — similarity-based routing for context reuse decisions.

Finds similar contexts using SIMD-accelerated cosine similarity and
decides whether to reuse a cached response or delegate to the
LangChain Executor for a new LLM call.

Supports three SIMD backends:
- Path A1: pycylon per-call (Python loop, one C++ call per embedding)
- Path A2: Cython batch (single C++ call for entire search)
- Path B:  WASM SIMD128 (Node.js — separate implementation)
"""

import logging
import time
from enum import Enum
from typing import Optional

import numpy as np

from context.manager import ContextManager
from chain.executor import ChainExecutor
from cost.bedrock_pricing import BedrockConfig, BedrockCostTracker

logger = logging.getLogger(__name__)


class SIMDBackend(Enum):
    """Available SIMD backends for similarity search."""
    PYCYLON = "pycylon"       # Path A1: per-call native C++ SIMD
    CYTHON_BATCH = "cython"   # Path A2: batch Cython extension
    NUMPY = "numpy"           # Fallback: pure numpy


def _cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Scalar cosine similarity via numpy (fallback)."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _load_pycylon():
    """Lazy-load pycylon SIMD bindings."""
    try:
        import pycylon
        return pycylon.simd.cosine_similarity_f32
    except (ImportError, AttributeError) as e:
        logger.warning("pycylon SIMD not available: %s", e)
        return None


def _load_cython_batch():
    """Lazy-load Cython batch search."""
    try:
        from simd.batch_search import batch_cosine_search
        return batch_cosine_search
    except ImportError as e:
        logger.warning("Cython batch search not available: %s", e)
        return None


class ContextRouter:
    """Route tasks through similarity search — reuse or new LLM call."""

    def __init__(
        self,
        context_manager: ContextManager,
        config: Optional[BedrockConfig] = None,
        backend: SIMDBackend = SIMDBackend.NUMPY,
        top_k: int = 5,
    ):
        if config is None:
            config = BedrockConfig.resolve()
        self.config = config
        self.context_manager = context_manager
        self.threshold = config.similarity_threshold
        self.top_k = top_k
        self.backend = backend

        # Lazy-loaded SIMD functions
        self._pycylon_cosine = None
        self._cython_batch = None

        if backend == SIMDBackend.PYCYLON:
            self._pycylon_cosine = _load_pycylon()
            if self._pycylon_cosine is None:
                logger.warning("Falling back to numpy for similarity")
                self.backend = SIMDBackend.NUMPY

        elif backend == SIMDBackend.CYTHON_BATCH:
            self._cython_batch = _load_cython_batch()
            if self._cython_batch is None:
                logger.warning("Falling back to numpy for similarity")
                self.backend = SIMDBackend.NUMPY

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity using the configured backend (A1 or fallback)."""
        if self.backend == SIMDBackend.PYCYLON and self._pycylon_cosine:
            return float(self._pycylon_cosine(a, b))
        return _cosine_similarity_numpy(a, b)

    def find_similar(
        self,
        query_embedding: np.ndarray,
        workflow_id: Optional[str] = None,
    ) -> list[dict]:
        """Return top-k contexts above threshold.

        Uses Path A2 (Cython batch) if available, otherwise
        Path A1 (per-call) or numpy fallback.

        Returns list of {context_id, similarity} sorted descending.
        """
        start = time.perf_counter()

        stored = self.context_manager.get_all_embeddings(workflow_id)
        if not stored:
            return []

        context_ids = [cid for cid, _ in stored]

        # Path A2: Cython batch search — single boundary crossing
        if self.backend == SIMDBackend.CYTHON_BATCH and self._cython_batch:
            embeddings_matrix = np.vstack([emb for _, emb in stored])
            batch_results = self._cython_batch(
                query_embedding, embeddings_matrix, self.threshold, self.top_k,
            )
            results = [
                {"context_id": context_ids[idx], "similarity": float(sim)}
                for idx, sim in batch_results
            ]

        # Path A1 / Fallback: per-embedding comparison
        else:
            similarities = []
            for cid, emb in stored:
                sim = self._cosine_similarity(query_embedding, emb)
                if sim >= self.threshold:
                    similarities.append({"context_id": cid, "similarity": sim})

            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            results = similarities[: self.top_k]

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "Similarity search: %d embeddings, %d above threshold %.2f (%.1fms, backend=%s)",
            len(stored), len(results), self.threshold, elapsed_ms, self.backend.value,
        )

        for r in results:
            r["search_latency_ms"] = round(elapsed_ms, 2)
            r["backend"] = self.backend.value
            r["embeddings_searched"] = len(stored)

        return results

    def should_reuse(
        self,
        query_embedding: np.ndarray,
        workflow_id: Optional[str] = None,
    ) -> tuple[bool, Optional[dict]]:
        """Decision function. Returns (reuse, best_match_or_none)."""
        results = self.find_similar(query_embedding, workflow_id)
        if results:
            return True, results[0]
        return False, None

    def route(
        self,
        task_description: str,
        query_embedding: np.ndarray,
        workflow_id: str,
        chain_executor: ChainExecutor,
        cost_tracker: BedrockCostTracker,
        embedding_metadata: dict,
    ) -> dict:
        """Full routing pipeline:
        1. Record embedding cost
        2. Check similarity → reuse or new call
        3. If new call: delegate to ChainExecutor, store context
        4. Record costs either way

        Returns:
            {response, source, cost_usd, similarity, search_latency_ms,
             llm_latency_ms, total_latency_ms, backend, context_id}
        """
        total_start = time.perf_counter()

        # Record embedding cost
        cost_tracker.record_embedding_call(
            embedding_metadata["model_id"],
            embedding_metadata["token_count"],
        )

        # Similarity search
        reuse, best_match = self.should_reuse(query_embedding, workflow_id)

        if reuse:
            ctx = self.context_manager.get_context(best_match["context_id"])
            if ctx is None:
                logger.warning(
                    "Context %s not found, falling back to new call",
                    best_match["context_id"],
                )
                reuse = False

        if reuse and ctx:
            # Cache hit — reuse existing response
            response_data = ctx.get("metadata", ctx)
            response_text = ctx.get("response", response_data.get("response", ""))

            # Record avoided cost using the original call's actual token counts
            input_tokens = ctx.get("cost_input_tokens", 0)
            output_tokens = ctx.get("cost_output_tokens", 0)
            if input_tokens == 0 and output_tokens == 0:
                logger.warning(
                    "Missing token counts for context %s, avoided cost will be 0",
                    best_match["context_id"],
                )
            cost_tracker.record_cache_hit(
                chain_executor.model_id, input_tokens, output_tokens,
            )

            self.context_manager.increment_reuse_count(
                best_match["context_id"], workflow_id,
            )

            total_ms = (time.perf_counter() - total_start) * 1000
            return {
                "response": response_text,
                "source": "cache",
                "cost_usd": 0.0,
                "similarity": best_match["similarity"],
                "search_latency_ms": best_match.get("search_latency_ms", 0),
                "llm_latency_ms": 0,
                "total_latency_ms": round(total_ms, 2),
                "backend": self.backend.value,
                "context_id": best_match["context_id"],
            }

        # Cache miss — new LLM call via ChainExecutor
        llm_result = chain_executor.execute(task_description)

        llm_cost = cost_tracker.record_llm_call(
            llm_result["model_id"],
            llm_result["input_tokens"],
            llm_result["output_tokens"],
        )

        # Store new context for future reuse
        context_id = self.context_manager.store_context(
            workflow_id=workflow_id,
            task_description=task_description,
            embedding=query_embedding,
            response=llm_result["response"],
            cost_metadata={
                "model_id": llm_result["model_id"],
                "input_tokens": llm_result["input_tokens"],
                "output_tokens": llm_result["output_tokens"],
                "cost_usd": llm_cost,
            },
        )

        total_ms = (time.perf_counter() - total_start) * 1000
        return {
            "response": llm_result["response"],
            "source": "llm",
            "cost_usd": llm_cost,
            "similarity": best_match["similarity"] if best_match else 0.0,
            "search_latency_ms": best_match.get("search_latency_ms", 0) if best_match else 0,
            "llm_latency_ms": llm_result["latency_ms"],
            "total_latency_ms": round(total_ms, 2),
            "backend": self.backend.value,
            "context_id": context_id,
        }