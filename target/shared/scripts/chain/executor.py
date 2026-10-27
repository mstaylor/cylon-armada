"""LangChain Executor — executes LLM chains via AWS Bedrock.

This is the langChain Executor Lambda component from the architecture diagram.
The Context Router delegates to this when a new LLM call is needed (no cache hit).
"""

import logging
import threading
import time
from typing import Optional

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

from cost.bedrock_pricing import BedrockConfig

logger = logging.getLogger(__name__)

BEDROCK_LATENCY_HEADER = "x-amzn-bedrock-invocation-latency"


def server_latency_ms(http_response, parsed):
    """Bedrock's own latency for one call: Converse reports it in
    metrics.latencyMs, InvokeModel in a response header. None if neither."""
    metrics = (parsed or {}).get("metrics") or {}
    if metrics.get("latencyMs") is not None:
        return float(metrics["latencyMs"])
    value = (getattr(http_response, "headers", None) or {}).get(BEDROCK_LATENCY_HEADER)
    return float(value) if value is not None else None


class ChainExecutor:
    """Execute LLM chains via Bedrock using LangChain."""

    def __init__(
        self,
        config: Optional[BedrockConfig] = None,
        endpoint_url: Optional[str] = None,
    ):
        if config is None:
            config = BedrockConfig.resolve()
        self.config = config

        kwargs = {
            "model_id": config.llm_model_id,
            "region_name": config.region,
            "model_kwargs": {"temperature": 0.0},
        }
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url

        self.llm = ChatBedrock(**kwargs)
        self._call = threading.local()
        client = getattr(self.llm, "client", None)
        events = getattr(getattr(client, "meta", None), "events", None)
        if events is not None:
            events.register("after-call.bedrock-runtime", self._capture_call)

    def _capture_call(self, http_response=None, parsed=None, **kwargs):
        self._call.server_latency_ms = server_latency_ms(http_response, parsed)
        self._call.retry_attempts = int(
            ((parsed or {}).get("ResponseMetadata") or {}).get("RetryAttempts", 0) or 0)

    @property
    def model_id(self) -> str:
        return self.config.llm_model_id

    def execute(
        self,
        task_description: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Execute a single LLM call for the given task.

        Returns:
            {response, input_tokens, output_tokens, latency_ms, model_id,
             server_latency_ms, retry_attempts, started_at}

        latency_ms is wall time in this process. server_latency_ms is what
        Bedrock reports for the same call, so their difference is time spent
        outside Bedrock, in the network, retries or this process.
        """
        self._call.server_latency_ms = None
        self._call.retry_attempts = 0
        started_at = time.time()
        start = time.perf_counter()

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=task_description))

        result = self.llm.invoke(messages)
        elapsed_ms = (time.perf_counter() - start) * 1000

        usage = result.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        logger.debug(
            "LLM call: %d input, %d output tokens (%.1fms)",
            input_tokens, output_tokens, elapsed_ms,
        )

        return {
            "response": result.content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(elapsed_ms, 2),
            "model_id": self.model_id,
            "server_latency_ms": self._call.server_latency_ms,
            "retry_attempts": self._call.retry_attempts,
            "started_at": started_at,
        }

    def execute_with_context(
        self,
        task_description: str,
        similar_context: dict,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Execute an LLM call augmented with a similar context for reference.

        Used when similarity is close but below the reuse threshold —
        the similar context provides useful reference without being
        directly reusable.
        """
        augmented_prompt = (
            f"A similar task was previously completed with this result:\n\n"
            f"Previous task: {similar_context.get('task_description', '')}\n"
            f"Previous response: {similar_context.get('response', '')}\n\n"
            f"Now complete this related task:\n{task_description}"
        )
        return self.execute(augmented_prompt, system_prompt=system_prompt)