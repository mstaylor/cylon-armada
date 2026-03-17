"""Tests for ContextManager — DynamoDB + Redis context store."""

import os
import sys
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'shared', 'scripts'))


class TestContextManager:
    @patch('context.manager.redis.Redis')
    @patch('context.manager.boto3')
    def test_store_context(self, mock_boto3, mock_redis_cls):
        from context.manager import ContextManager

        mock_dynamo = MagicMock()
        mock_boto3.resource.return_value.Table.return_value = mock_dynamo

        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        mock_redis_cls.return_value = mock_redis

        cm = ContextManager(redis_host="localhost", redis_port=6379)
        embedding = np.random.randn(256).astype(np.float32)

        cm.store_context(
            workflow_id="wf-1",
            task_description="test task",
            embedding=embedding,
            response="test response",
            cost_metadata={"model_id": "test-model", "input_tokens": 100},
        )

        mock_dynamo.put_item.assert_called_once()

    @patch('context.manager.redis.Redis')
    @patch('context.manager.boto3')
    def test_get_context_from_redis(self, mock_boto3, mock_redis_cls):
        from context.manager import ContextManager

        mock_redis = MagicMock()
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # get returns JSON string for context, bytes for embedding
        def mock_get(key):
            if key.startswith("context:"):
                return json.dumps({"response": "cached response", "metadata": {}})
            elif key.startswith("embedding:"):
                return emb.tobytes()
            return None

        mock_redis.get.side_effect = mock_get
        mock_redis_cls.return_value = mock_redis

        cm = ContextManager(redis_host="localhost", redis_port=6379)
        result = cm.get_context("ctx-123")

        assert result is not None
        assert result["response"] == "cached response"
        assert result["source"] == "cache"

    @patch('context.manager.redis.Redis')
    @patch('context.manager.boto3')
    def test_get_all_embeddings_from_redis(self, mock_boto3, mock_redis_cls):
        from context.manager import ContextManager

        mock_redis = MagicMock()
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        mock_redis.smembers.return_value = {"ctx-1", "ctx-2"}

        # _load_embeddings_from_redis uses pipeline
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [emb1.tobytes(), emb2.tobytes()]
        mock_redis.pipeline.return_value = mock_pipe

        mock_redis_cls.return_value = mock_redis

        cm = ContextManager(redis_host="localhost", redis_port=6379)
        results = cm.get_all_embeddings(workflow_id="wf-1")

        assert len(results) == 2
        for ctx_id, embedding in results:
            assert isinstance(ctx_id, str)
            assert isinstance(embedding, np.ndarray)
            assert embedding.dtype == np.float32

    @patch('context.manager.redis.Redis')
    @patch('context.manager.boto3')
    def test_increment_reuse_count(self, mock_boto3, mock_redis_cls):
        from context.manager import ContextManager

        mock_dynamo = MagicMock()
        mock_boto3.resource.return_value.Table.return_value = mock_dynamo
        mock_redis_cls.return_value = MagicMock()

        cm = ContextManager(redis_host="localhost", redis_port=6379)
        cm.increment_reuse_count("ctx-123", "wf-1")

        mock_dynamo.update_item.assert_called_once()