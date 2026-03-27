"""Tests for ContextManager — configuration-driven backend selection."""

import os
import sys
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'shared', 'scripts'))


class TestContextManagerRedisBackend:
    """Tests using the 'redis' backend (numpy+Redis).

    Default config: persist_to_redis=True, dynamo_table=None.
    Redis is the primary persistence layer; DynamoDB is disabled.
    """

    @patch('context.manager._create_redis_backend')
    def test_store_context_persists_to_redis(self, mock_create_redis):
        from context.manager import ContextManager

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

        # Embedding pipeline (setex + sadd + expire) should execute
        mock_pipe.execute.assert_called_once()
        # Metadata JSON should be written via setex
        mock_redis.setex.assert_called_once()
        setex_key = mock_redis.setex.call_args[0][0]
        assert setex_key.startswith("context:")

    @patch('context.manager._create_redis_backend')
    def test_store_context_no_dynamo_by_default(self, mock_create_redis):
        """DynamoDB should NOT be called when dynamo_table is None (default)."""
        from context.manager import ContextManager

        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis")
        assert cm._table is None

        cm.store_context(
            workflow_id="wf-1",
            task_description="test task",
            embedding=np.random.randn(256).astype(np.float32),
            response="resp",
            cost_metadata={},
        )
        # No DynamoDB put_item
        assert cm._dynamo is None

    @patch('context.manager._create_redis_backend')
    def test_store_context_with_dynamo(self, mock_create_redis):
        """DynamoDB put_item called when dynamo_table is provided."""
        from context.manager import ContextManager

        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        mock_create_redis.return_value = {"client": mock_redis}

        with patch('context.manager.boto3') as mock_boto3:
            mock_dynamo_table = MagicMock()
            mock_boto3.resource.return_value.Table.return_value = mock_dynamo_table

            cm = ContextManager(backend="redis", dynamo_table="my-table")
            cm.store_context(
                workflow_id="wf-1",
                task_description="test task",
                embedding=np.random.randn(256).astype(np.float32),
                response="resp",
                cost_metadata={},
            )

        mock_dynamo_table.put_item.assert_called_once()

    @patch('context.manager._create_redis_backend')
    def test_get_context_from_redis(self, mock_create_redis):
        from context.manager import ContextManager

        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({
            "response": "cached response",
            "metadata": {},
        }).encode()
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis")
        result = cm.get_context("ctx-123")

        assert result is not None
        assert result["response"] == "cached response"
        assert result["source"] == "redis"

    @patch('context.manager._create_redis_backend')
    def test_get_all_embeddings(self, mock_create_redis):
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

    @patch('context.manager._create_redis_backend')
    def test_search_redis(self, mock_create_redis):
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
        assert results[0]["similarity"] > 0.5
        if len(results) > 1:
            assert results[0]["similarity"] >= results[1]["similarity"]

    @patch('context.manager._create_redis_backend')
    def test_increment_reuse_count_updates_redis(self, mock_create_redis):
        from context.manager import ContextManager

        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({
            "response": "resp",
            "reuse_count": 2,
        }).encode()
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis")
        cm.increment_reuse_count("ctx-123", "wf-1")

        # Redis setex should be called with incremented count
        mock_redis.setex.assert_called_once()
        written = json.loads(mock_redis.setex.call_args[0][2])
        assert written["reuse_count"] == 3

    @patch('context.manager._create_redis_backend')
    def test_get_workflow_contexts_from_redis(self, mock_create_redis):
        from context.manager import ContextManager

        ctx_data = {"response": "resp", "workflow_id": "wf-1", "task_description": "t"}
        mock_redis = MagicMock()
        mock_redis.smembers.return_value = {b"ctx-1"}
        mock_redis.get.return_value = json.dumps(ctx_data).encode()
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis")
        results = cm.get_workflow_contexts("wf-1")

        assert len(results) == 1
        assert results[0]["context_id"] == "ctx-1"

    @patch('context.manager._create_redis_backend')
    def test_in_memory_fallback_when_no_persistence(self, mock_create_redis):
        """When persist_to_redis=False and no dynamo_table, contexts go to _memory."""
        from context.manager import ContextManager

        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis", persist_to_redis=False)
        embedding = np.random.randn(256).astype(np.float32)

        ctx_id = cm.store_context(
            workflow_id="wf-1",
            task_description="in-memory task",
            embedding=embedding,
            response="resp",
            cost_metadata={},
        )

        assert ctx_id in cm._memory
        assert cm._memory[ctx_id]["task_description"] == "in-memory task"
        # No metadata JSON written to Redis
        mock_redis.setex.assert_not_called()


class TestContextManagerValidation:
    def test_invalid_backend_raises(self):
        from context.manager import ContextManager
        with pytest.raises(ValueError, match="Invalid context_backend"):
            ContextManager(backend="invalid")

    @patch('context.manager._create_redis_backend')
    def test_from_config(self, mock_create_redis):
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