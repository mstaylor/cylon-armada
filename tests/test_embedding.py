"""Tests for EmbeddingService."""

import os
import sys
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'shared', 'scripts'))

from cost.bedrock_pricing import BedrockConfig


class TestEmbeddingService:
    @patch('context.embedding.boto3')
    def test_embed_returns_correct_shape(self, mock_boto3):
        """Embedding should return array of configured dimensions."""
        from context.embedding import EmbeddingService

        dimensions = 256
        config = BedrockConfig(embedding_dimensions=dimensions)

        # Mock Bedrock response
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        fake_embedding = list(np.random.randn(dimensions).astype(float))
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "embedding": fake_embedding,
            "inputTextTokenCount": 15,
        }).encode()
        mock_client.invoke_model.return_value = {"body": mock_response}

        service = EmbeddingService(config=config)
        embedding, metadata = service.embed("test text")

        assert embedding.shape == (dimensions,)
        assert embedding.dtype == np.float32
        assert metadata["token_count"] == 15
        assert metadata["dimensions"] == dimensions

    @patch('context.embedding.boto3')
    def test_embed_returns_metadata(self, mock_boto3):
        """Metadata should include model_id, token_count, latency_ms, dimensions."""
        from context.embedding import EmbeddingService

        config = BedrockConfig(embedding_dimensions=1024)

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        fake_embedding = list(np.random.randn(1024).astype(float))
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "embedding": fake_embedding,
            "inputTextTokenCount": 42,
        }).encode()
        mock_client.invoke_model.return_value = {"body": mock_response}

        service = EmbeddingService(config=config)
        _, metadata = service.embed("test text")

        assert "model_id" in metadata
        assert "token_count" in metadata
        assert "latency_ms" in metadata
        assert "dimensions" in metadata
        assert metadata["token_count"] == 42
        assert metadata["latency_ms"] >= 0

    @patch('context.embedding.boto3')
    def test_embed_different_dimensions(self, mock_boto3):
        """Dimensions parameter should be passed to Bedrock."""
        from context.embedding import EmbeddingService

        for dim in [256, 512, 1024]:
            config = BedrockConfig(embedding_dimensions=dim)

            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            fake_embedding = list(np.random.randn(dim).astype(float))
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({
                "embedding": fake_embedding,
                "inputTextTokenCount": 10,
            }).encode()
            mock_client.invoke_model.return_value = {"body": mock_response}

            service = EmbeddingService(config=config)
            embedding, _ = service.embed("test")

            assert embedding.shape == (dim,)

            # Verify dimensions was passed in the request body
            call_args = mock_client.invoke_model.call_args
            body = json.loads(call_args[1]["body"] if "body" in call_args[1] else call_args[0][0])
            assert body["dimensions"] == dim