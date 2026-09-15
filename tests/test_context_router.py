"""Tests for ContextRouter — similarity search and routing logic."""

import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'shared', 'scripts'))

from context.router import ContextRouter, SIMDBackend
from cost.bedrock_pricing import BedrockConfig, BedrockCostTracker


def _stored_embeddings(cm, pairs):
    """Point the mocked manager at these (context_id, embedding) pairs.

    find_similar reads the store as one matrix rather than as per-row pairs, so
    both shapes are stocked: get_embedding_matrix is what the router calls, and
    get_all_embeddings is still the public contract other callers use.
    """
    ids = [cid for cid, _ in pairs]
    matrix = (np.vstack([emb for _, emb in pairs]) if pairs
              else np.empty((0, 0), dtype=np.float32))
    cm.get_all_embeddings.return_value = pairs
    cm.get_embedding_matrix.return_value = (ids, matrix)


@pytest.fixture
def mock_context_manager():
    cm = MagicMock()
    cm.get_all_embeddings = MagicMock(return_value=[])
    cm.get_embedding_matrix = MagicMock(
        return_value=([], np.empty((0, 0), dtype=np.float32)))
    cm.get_context = MagicMock(return_value=None)
    cm.store_context = MagicMock()
    cm.increment_reuse_count = MagicMock()
    return cm


@pytest.fixture
def config():
    return BedrockConfig(similarity_threshold=0.85)


@pytest.fixture
def router(mock_context_manager, config):
    return ContextRouter(
        mock_context_manager,
        config=config,
        backend=SIMDBackend.NUMPY,
    )


class TestContextRouter:
    def test_find_similar_empty_store(self, router):
        query = np.random.randn(256).astype(np.float32)
        results = router.find_similar(query, workflow_id="test-wf")
        assert results == []

    def test_find_similar_with_matches(self, router, mock_context_manager):
        # Store embeddings that are similar to the query
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        similar = np.array([0.99, 0.1, 0.0, 0.0], dtype=np.float32)
        dissimilar = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Normalize
        query /= np.linalg.norm(query)
        similar /= np.linalg.norm(similar)
        dissimilar /= np.linalg.norm(dissimilar)

        _stored_embeddings(mock_context_manager, [
            ("ctx-1", similar),
            ("ctx-2", dissimilar),
        ])

        results = router.find_similar(query, workflow_id="test-wf")

        # Similar should be above threshold, dissimilar below
        assert len(results) >= 1
        assert results[0]["context_id"] == "ctx-1"
        assert results[0]["similarity"] > 0.85

    def test_should_reuse_true(self, router, mock_context_manager):
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        stored = np.array([0.99, 0.05, 0.0], dtype=np.float32)
        query /= np.linalg.norm(query)
        stored /= np.linalg.norm(stored)

        _stored_embeddings(mock_context_manager, [("ctx-1", stored)])

        reuse, match = router.should_reuse(query, workflow_id="test-wf")
        assert reuse is True
        assert match is not None
        assert match["context_id"] == "ctx-1"

    def test_should_reuse_false(self, router, mock_context_manager):
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        stored = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        _stored_embeddings(mock_context_manager, [("ctx-1", stored)])

        reuse, match = router.should_reuse(query, workflow_id="test-wf")
        assert reuse is False

    def test_cosine_similarity_identical(self, router):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        sim = router._cosine_similarity(a, b)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_orthogonal(self, router):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        sim = router._cosine_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=1e-6)

    def test_backend_enum(self):
        assert SIMDBackend.NUMPY.value == "numpy"
        assert SIMDBackend.PYCYLON.value == "pycylon"
        assert SIMDBackend.CYTHON_BATCH.value == "cython"