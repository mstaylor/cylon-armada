"""Tests for ChainExecutor — LangChain + Bedrock integration."""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'shared', 'scripts'))

from cost.bedrock_pricing import BedrockConfig


class TestChainExecutor:
    @patch('chain.executor.ChatBedrock')
    def test_execute_returns_structure(self, mock_chat_bedrock):
        from chain.executor import ChainExecutor

        # Mock LangChain response
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is a test response."
        mock_response.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
        }
        mock_llm.invoke.return_value = mock_response
        mock_chat_bedrock.return_value = mock_llm

        config = BedrockConfig()
        executor = ChainExecutor(config=config)
        result = executor.execute("Summarize serverless computing")

        assert "response" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert "latency_ms" in result
        assert "model_id" in result
        assert result["response"] == "This is a test response."
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    @patch('chain.executor.ChatBedrock')
    def test_execute_with_system_prompt(self, mock_chat_bedrock):
        from chain.executor import ChainExecutor

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "response"
        mock_response.usage_metadata = {"input_tokens": 50, "output_tokens": 25}
        mock_llm.invoke.return_value = mock_response
        mock_chat_bedrock.return_value = mock_llm

        executor = ChainExecutor(config=BedrockConfig())
        result = executor.execute("task", system_prompt="You are an astronomer.")

        assert result["response"] == "response"
        # Verify system prompt was included in the invocation
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        assert any("astronomer" in str(m) for m in messages)

    @patch('chain.executor.ChatBedrock')
    def test_execute_with_context(self, mock_chat_bedrock):
        from chain.executor import ChainExecutor

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "augmented response"
        mock_response.usage_metadata = {"input_tokens": 200, "output_tokens": 100}
        mock_llm.invoke.return_value = mock_response
        mock_chat_bedrock.return_value = mock_llm

        executor = ChainExecutor(config=BedrockConfig())
        similar_context = {
            "task_description": "related task",
            "response": "prior response",
            "similarity": 0.78,
        }
        result = executor.execute_with_context("new task", similar_context)

        assert result["response"] == "augmented response"
        assert result["input_tokens"] == 200

    @patch('chain.executor.ChatBedrock')
    def test_latency_is_positive(self, mock_chat_bedrock):
        from chain.executor import ChainExecutor

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "response"
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
        mock_llm.invoke.return_value = mock_response
        mock_chat_bedrock.return_value = mock_llm

        executor = ChainExecutor(config=BedrockConfig())
        result = executor.execute("test")

        assert result["latency_ms"] >= 0