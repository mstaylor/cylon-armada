"""Tests for ContextManager — configuration-driven backend selection."""

import os
import sys
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'shared', 'scripts'))


class TestContextManagerRedisBackend:
    """Tests using the 'redis' backend (numpy+Redis)."""

    @patch('context.manager.boto3')
    @patch('context.manager._create_redis_backend')
    def test_store_context(self, mock_create_redis, mock_boto3):
        from context.manager import ContextManager

        mock_dynamo = MagicMock()
        mock_boto3.resource.return_value.Table.return_value = mock_dynamo

        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis")
        embedding = np.random.randn(256).astype(np.float32)

        cm.store_context(
            workflow_id="wf-1",
            task_description="test task",
            embedding=embedding,
            response="test response",
            cost_metadata={"model_id": "test-model", "input_tokens": 100},
        )

        mock_dynamo.put_item.assert_called_once()
        mock_pipe.execute.assert_called_once()

    @patch('context.manager.boto3')
    @patch('context.manager._create_redis_backend')
    def test_get_context_from_redis(self, mock_create_redis, mock_boto3):
        from context.manager import ContextManager

        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({
            "response": "cached response",
            "metadata": {},
        })
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis")
        result = cm.get_context("ctx-123")

        assert result is not None
        assert result["response"] == "cached response"
        assert result["source"] == "cache"

    @patch('context.manager.boto3')
    @patch('context.manager._create_redis_backend')
    def test_get_all_embeddings(self, mock_create_redis, mock_boto3):
        from context.manager import ContextManager

        mock_redis = MagicMock()
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        mock_redis.smembers.return_value = {b"ctx-1", b"ctx-2"}
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [emb1.tobytes(), emb2.tobytes()]
        mock_redis.pipeline.return_value = mock_pipe
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis")
        results = cm.get_all_embeddings(workflow_id="wf-1")

        assert len(results) == 2
        for ctx_id, embedding in results:
            assert isinstance(ctx_id, str)
            assert isinstance(embedding, np.ndarray)
            assert embedding.dtype == np.float32

    @patch('context.manager.boto3')
    @patch('context.manager._create_redis_backend')
    def test_search_redis(self, mock_create_redis, mock_boto3):
        from context.manager import ContextManager

        mock_redis = MagicMock()
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        mock_redis.smembers.return_value = {b"ctx-1", b"ctx-2"}
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [emb1.tobytes(), emb2.tobytes()]
        mock_redis.pipeline.return_value = mock_pipe
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis")
        query = np.array([0.99, 0.1, 0.0], dtype=np.float32)
        results = cm.search(query, threshold=0.5, workflow_id="wf-1")

        assert len(results) >= 1
        # Best match should be the embedding closest to the query
        assert results[0]["similarity"] > 0.5
        # Results should be sorted by descending similarity
        if len(results) > 1:
            assert results[0]["similarity"] >= results[1]["similarity"]

    @patch('context.manager.boto3')
    @patch('context.manager._create_redis_backend')
    def test_increment_reuse_count(self, mock_create_redis, mock_boto3):
        from context.manager import ContextManager

        mock_dynamo = MagicMock()
        mock_boto3.resource.return_value.Table.return_value = mock_dynamo
        mock_create_redis.return_value = {"client": MagicMock()}

        cm = ContextManager(backend="redis")
        cm.increment_reuse_count("ctx-123", "wf-1")

        mock_dynamo.update_item.assert_called_once()


class TestContextManagerValidation:
    def test_invalid_backend_raises(self):
        from context.manager import ContextManager
        with pytest.raises(ValueError, match="Invalid context_backend"):
            ContextManager(backend="invalid")

    @patch('context.manager.boto3')
    @patch('context.manager._create_redis_backend')
    def test_from_config(self, mock_create_redis, mock_boto3):
        from context.manager import ContextManager
        from cost.bedrock_pricing import BedrockConfig

        mock_create_redis.return_value = {"client": MagicMock()}

        config = BedrockConfig(
            context_backend="redis",
            embedding_dimensions=512,
        )
        cm = ContextManager.from_config(config)
        assert cm._backend_name == "redis"
        assert cm._embedding_dim == 512