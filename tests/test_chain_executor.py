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

def _stubbed_bedrock_llm(converse_response, calls_client=True):
    import boto3
    from botocore.stub import Stubber
    from langchain_core.messages import AIMessage

    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          aws_access_key_id="testing", aws_secret_access_key="testing")
    stubber = Stubber(client)
    stubber.add_response("converse", converse_response)
    stubber.activate()

    class StubbedLlm:
        def __init__(self):
            self.client = client

        def invoke(self, messages):
            if calls_client:
                client.converse(modelId="m",
                                messages=[{"role": "user", "content": [{"text": "q"}]}])
            return AIMessage(content="answer",
                             usage_metadata={"input_tokens": 3, "output_tokens": 4,
                                             "total_tokens": 7})

    return StubbedLlm()


_CONVERSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "answer"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 3, "outputTokens": 4, "totalTokens": 7},
    "metrics": {"latencyMs": 812},
}


@patch("chain.executor.ChatBedrock")
def test_bedrocks_own_latency_and_retries_are_reported_beside_the_wall_time(mock_chat_bedrock):
    from chain.executor import ChainExecutor

    mock_chat_bedrock.return_value = _stubbed_bedrock_llm(
        {**_CONVERSE, "ResponseMetadata": {"RetryAttempts": 2}})

    result = ChainExecutor(config=BedrockConfig()).execute("task")

    assert result["server_latency_ms"] == 812.0
    assert result["retry_attempts"] == 2
    assert result["latency_ms"] >= 0.0
    assert result["started_at"] > 0.0


@patch("chain.executor.ChatBedrock")
def test_a_call_without_server_metrics_reports_none_not_a_stale_value(mock_chat_bedrock):
    from chain.executor import ChainExecutor

    llm = _stubbed_bedrock_llm(_CONVERSE)
    mock_chat_bedrock.return_value = llm
    executor = ChainExecutor(config=BedrockConfig())
    executor.execute("task")
    llm_without_call = _stubbed_bedrock_llm(_CONVERSE, calls_client=False)
    executor.llm.invoke = llm_without_call.invoke

    result = executor.execute("task")

    assert result["server_latency_ms"] is None
    assert result["retry_attempts"] == 0


def test_invoke_model_latency_comes_from_its_response_header():
    from types import SimpleNamespace

    from chain.executor import server_latency_ms

    response = SimpleNamespace(headers={"x-amzn-bedrock-invocation-latency": "640"})
    assert server_latency_ms(response, {}) == 640.0
    assert server_latency_ms(SimpleNamespace(headers={}), {}) is None
