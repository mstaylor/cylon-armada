"""Plan Cosmic AI executions with an LLM (Amazon Nova Lite through ChainExecutor)."""

from dataclasses import dataclass

from cosmic_campaign.plan_check import PLAN_FIELDS, PlanParseError, parse_plans


@dataclass(frozen=True)
class PlanOutcome:
    plans: list
    source: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    error: str | None


def system_prompt(settings, arm):
    return f"""You plan AWS Step Functions executions for the Cosmic AI inference campaign.
Reply with JSON only: one object per execution, or a JSON list of objects for a multi-run request.
Each object has exactly these keys: {", ".join(PLAN_FIELDS)}.
Rules:
- bucket and data_bucket are "{settings.bucket}" and "{settings.data_bucket}".
- object_type is "{settings.object_type}", S3_object_name is "{settings.object_name}", script is "{settings.scripts[arm]}".
- data_prefix is the partition size followed by MB, for example "100MB".
- world_size is the number of workers = ceil(data size in GB x 1024 / partition size in MB), an integer;
  file_limit is the same number written as a string.
- batch_size is an integer.
- result_path is "{settings.result_prefix}/{arm}/result-partition-<P>MB/<D>GB/<run>", where <P> is the
  partition size, <D> the data size without trailing zeros (1GB, 12.6GB), and <run> is warmup0 for the
  cold-start run or run<k> for measured run k.
- When the request says it belongs to the batch sweep, result_path is
  "{settings.result_prefix}/{arm}/result-partition-<P>MB/<D>GB/Batches/batch<B>/<run>", where <B> is the batch size."""


class LLMPlanner:
    source = "llm"

    def __init__(self, executor, settings, arm):
        self.executor, self.settings, self.arm = executor, settings, arm
        self.prompt = system_prompt(settings, arm)

    def plan(self, request):
        result = self.executor.execute(request.text, system_prompt=self.prompt)
        try:
            plans, error = parse_plans(result["response"]), None
        except PlanParseError as exc:
            plans, error = [], str(exc)
        return PlanOutcome(plans, self.source, result["latency_ms"], result["input_tokens"],
                           result["output_tokens"], error)


def worker_count(data_gb: float, partition_mb: int) -> int:
    """Number of workers for a data size in GB and a partition size in MB."""
    import math

    return math.ceil(data_gb * 1024 / partition_mb)


TOOL_INSTRUCTION = """
Use the worker_count tool to obtain world_size and file_limit; never compute them yourself."""


def _text(content):
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content


class ToolPlanner:
    """Plan with a chat model that can call worker_count (Nova Lite via ChatBedrockConverse)."""

    source = "llm_tool"

    def __init__(self, model, settings, arm, max_steps):
        from langchain_core.tools import tool

        self.tool = tool(worker_count)
        self.model = model.bind_tools([self.tool])
        self.prompt = system_prompt(settings, arm) + TOOL_INSTRUCTION
        self.max_steps = max_steps

    def plan(self, request):
        import time

        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        messages = [SystemMessage(content=self.prompt), HumanMessage(content=request.text)]
        start, input_tokens, output_tokens = time.perf_counter(), 0, 0
        for _ in range(self.max_steps):
            reply = self.model.invoke(messages)
            usage = reply.usage_metadata or {}
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
            messages.append(reply)
            if not reply.tool_calls:
                latency_ms = (time.perf_counter() - start) * 1000
                try:
                    return PlanOutcome(parse_plans(_text(reply.content)), self.source, latency_ms,
                                       input_tokens, output_tokens, None)
                except PlanParseError as exc:
                    return PlanOutcome([], self.source, latency_ms, input_tokens, output_tokens, str(exc))
            for call in reply.tool_calls:
                if call["name"] != self.tool.name:
                    return PlanOutcome([], self.source, (time.perf_counter() - start) * 1000, input_tokens,
                                       output_tokens, f"model called unknown tool {call['name']!r}")
                messages.append(ToolMessage(content=str(self.tool.invoke(call["args"])), tool_call_id=call["id"]))
        return PlanOutcome([], self.source, (time.perf_counter() - start) * 1000, input_tokens, output_tokens,
                           f"no plan after {self.max_steps} steps")
