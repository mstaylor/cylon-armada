"""Tests for ContextManager — DynamoDB + Redis context store."""

import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'shared', 'scripts'))


class TestContextManager:
    @patch('context.manager.redis.Redis')
    @patch('context.manager.boto3')
    def test_store_context(self, mock_boto3, mock_redis_cls):
        from context.manager import ContextManager

        mock_dynamo = MagicMock()
        mock_boto3.resource.return_value.Table.return_value = mock_dynamo

        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_redis
        mock_redis.__enter__ = MagicMock(return_value=mock_redis)
        mock_redis.__exit__ = MagicMock(return_value=False)
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

        # DynamoDB put_item should have been called
        mock_dynamo.put_item.assert_called_once()

    @patch('context.manager.redis.Redis')
    @patch('context.manager.boto3')
    def test_get_context_from_redis(self, mock_boto3, mock_redis_cls):
        from context.manager import ContextManager
        import json

        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({
            "response": "cached response",
            "model_id": "test-model",
        })
        mock_redis_cls.return_value = mock_redis

        cm = ContextManager(redis_host="localhost", redis_port=6379)
        result = cm.get_context("ctx-123")

        assert result is not None
        assert result["response"] == "cached response"

    @patch('context.manager.redis.Redis')
    @patch('context.manager.boto3')
    def test_get_all_embeddings(self, mock_boto3, mock_redis_cls):
        from context.manager import ContextManager

        mock_redis = MagicMock()
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_redis.smembers.return_value = {b"ctx-1", b"ctx-2"}
        mock_redis.get.return_value = emb.tobytes()
        mock_redis_cls.return_value = mock_redis

        cm = ContextManager(redis_host="localhost", redis_port=6379)
        results = cm.get_all_embeddings(workflow_id="wf-1")

        assert len(results) == 2
        for ctx_id, embedding in results:
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