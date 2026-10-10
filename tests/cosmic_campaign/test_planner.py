import json

from cosmic_campaign.grid import CampaignSettings, run_slots, Configuration
from cosmic_campaign.planner import LLMPlanner, system_prompt
from cosmic_campaign.requests import execution_requests

SETTINGS = CampaignSettings("b", "b", "Anomaly Detection", "folder", {"A": "/tmp/a.py"}, "p/exp2")


class _Executor:
    def __init__(self, response):
        self.response, self.calls = response, []

    def execute(self, task_description, system_prompt=None):
        self.calls.append((task_description, system_prompt))
        return {"response": self.response, "input_tokens": 100, "output_tokens": 50,
                "latency_ms": 900.0, "model_id": "nova-lite"}


def _request():
    return execution_requests(run_slots([Configuration("scaling", 100, 1, 512, 11)], 0, 1), "A", SETTINGS, 1)[0]


def test_the_system_prompt_states_every_campaign_convention_the_model_needs():
    prompt = system_prompt(SETTINGS, "A")
    for needle in ("ceil", "1024", "file_limit", "world_size", "result_path", "warmup0", "run<k>",
                   "data_prefix", "/tmp/a.py", "p/exp2/A", "JSON", "Batches/batch<B>", "batch sweep"):
        assert needle in prompt, needle


def test_a_good_response_becomes_a_plan_with_its_cost():
    request = _request()
    executor = _Executor(json.dumps(request.references[0]))
    outcome = LLMPlanner(executor, SETTINGS, "A").plan(request)
    assert outcome.plans == [request.references[0]] and outcome.source == "llm" and outcome.error is None
    assert (outcome.latency_ms, outcome.input_tokens, outcome.output_tokens) == (900.0, 100, 50)
    assert executor.calls[0][0] == request.text


def test_an_unparseable_response_is_an_error_outcome_not_an_exception():
    outcome = LLMPlanner(_Executor("no idea"), SETTINGS, "A").plan(_request())
    assert outcome.plans == [] and "no JSON" in outcome.error


from langchain_core.messages import AIMessage

from cosmic_campaign.planner import ToolPlanner, worker_count


class _ToolModel:
    """Stand-in chat model: first asks for the worker count, then returns the plan using the tool's answer."""

    def __init__(self, plan, steps=("tool", "plan")):
        self.plan, self.steps, self.seen = plan, list(steps), []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        step = self.steps.pop(0)
        if step == "tool":
            return AIMessage(content="", tool_calls=[{"name": "worker_count", "id": "c1",
                                                      "args": {"data_gb": 2, "partition_mb": 100}}],
                             usage_metadata={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60})
        tool_result = messages[-1].content
        return AIMessage(content=json.dumps({**self.plan, "world_size": int(tool_result),
                                             "file_limit": tool_result}),
                         usage_metadata={"input_tokens": 70, "output_tokens": 40, "total_tokens": 110})


def test_worker_count_is_the_campaign_formula():
    assert (worker_count(2, 100), worker_count(12.6, 25), worker_count(1, 100)) == (21, 517, 11)


def test_the_tool_planner_runs_the_tool_and_returns_a_correct_plan():
    request = execution_requests(run_slots([Configuration("scaling", 100, 2, 512, 21)], 0, 1), "A", SETTINGS, 1)[0]
    reference = request.references[0]
    model = _ToolModel({**reference, "world_size": 20, "file_limit": "20"})
    outcome = ToolPlanner(model, SETTINGS, "A", max_steps=4).plan(request)
    assert outcome.error is None and outcome.source == "llm_tool"
    assert outcome.plans == [reference]
    assert (outcome.input_tokens, outcome.output_tokens) == (120, 50)
    assert "worker_count" in model.seen[0][0].content


def test_a_call_to_an_unknown_tool_is_an_error_outcome():
    request = execution_requests(run_slots([Configuration("scaling", 100, 2, 512, 21)], 0, 1), "A", SETTINGS, 1)[0]

    class _Stray(_ToolModel):
        def invoke(self, messages):
            return AIMessage(content="", tool_calls=[{"name": "delete_bucket", "id": "c1", "args": {}}])

    outcome = ToolPlanner(_Stray(request.references[0]), SETTINGS, "A", max_steps=3).plan(request)
    assert outcome.plans == [] and "delete_bucket" in outcome.error


def test_a_model_that_never_stops_calling_tools_is_an_error_outcome():
    request = execution_requests(run_slots([Configuration("scaling", 100, 2, 512, 21)], 0, 1), "A", SETTINGS, 1)[0]
    model = _ToolModel(request.references[0], steps=("tool", "tool", "tool"))
    outcome = ToolPlanner(model, SETTINGS, "A", max_steps=3).plan(request)
    assert outcome.plans == [] and "steps" in outcome.error
