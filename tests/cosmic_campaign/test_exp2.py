import importlib.util
import os

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _exp2():
    path = os.path.join(ROOT, "target", "aws", "scripts", "experiment", "cosmic_campaign_exp2.py")
    spec = importlib.util.spec_from_file_location("cosmic_campaign_exp2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(request_id, plans, references, source="llm"):
    from cosmic_campaign.plan_check import mismatched_fields
    return {"request_id": request_id, "source": source, "planning_ms": 5.0, "input_tokens": 10,
            "output_tokens": 5, "error": None, "plans": plans, "references": references,
            "mismatches": [mismatched_fields(p, r) for p, r in zip(plans, references)]}


def test_a_wrong_plan_is_recorded_and_the_reference_is_executed_instead():
    m = _exp2()
    ref = {"result_path": "p/run1", "file_limit": "11"}
    wrong = {"result_path": "p/run1", "file_limit": "10"}
    to_run, records = m.merge_plans([_row("x1", [wrong], [ref])])
    assert to_run == [ref]
    assert records[0]["plan_correct"] is False and records[0]["mismatches"] == [["file_limit"]]


def test_metrics_separate_llm_calls_from_cache_hits_and_wrong_reuse():
    m = _exp2()
    ref = {"result_path": "p/run1", "file_limit": "11"}
    rows = [_row("x1", [ref], [ref]),
            _row("x2", [ref], [ref], source="cache_structural"),
            _row("x3", [{"result_path": "p/run2", "file_limit": "11"}], [ref], source="cache_exact")]
    metrics = m.plan_metrics(rows)
    assert (metrics["llm_calls"], metrics["cache_hits_structural"], metrics["cache_hits_exact"]) == (1, 1, 1)
    assert (metrics["wrong_plans"], metrics["wrong_reuse"]) == (1, 1)


def test_experiment2_requests_leave_out_the_baselines(tmp_path):
    import json
    m = _exp2()
    out = tmp_path / "r.json"
    m.cmd_requests(["--out", str(out)])
    paths = [r["references"][0]["result_path"] for r in json.load(open(out))["requests"]]
    assert paths and not any("baseline" in p for p in paths)
    assert any("/Batches/" in p for p in paths)


def test_metrics_count_tool_planner_calls_separately():
    m = _exp2()
    ref = {"result_path": "p/run1", "file_limit": "11"}
    metrics = m.plan_metrics([_row("x1", [ref], [ref]), _row("x2", [ref], [ref], source="llm_tool")])
    assert (metrics["llm_calls"], metrics["llm_tool_calls"]) == (1, 1)


def test_metrics_report_planning_wall_clock_allgather_and_embedding_cost():
    m = _exp2()
    ref = {"result_path": "p/run1", "file_limit": "11"}
    rows = [{**_row("x1", [ref], [ref]), "embed_ms": 40.0, "embed_tokens": 7},
            {**_row("x2", [ref], [ref], source="cache_structural"), "embed_ms": 20.0, "embed_tokens": 5}]
    agents = [{"rank": 0, "wall_ms": 900.0, "share_ms": [3.0, 5.0]},
              {"rank": 1, "wall_ms": 1200.0, "share_ms": [4.0]}]
    metrics = m.plan_metrics(rows, agents)
    assert metrics["planning_wall_ms"] == 1200.0
    assert metrics["share_ms_mean"] == 4.0
    assert (metrics["embed_ms_mean"], metrics["embed_tokens"]) == (30.0, 12)


def test_a_correct_plan_runs_with_only_the_plan_fields_in_the_reference_types():
    from cosmic_campaign.plan_check import PLAN_FIELDS
    m = _exp2()
    ref = {f: f"v-{f}" for f in PLAN_FIELDS}
    ref.update(file_limit="11", world_size=11, batch_size=512)
    plan = {**ref, "world_size": "11", "batch_size": "512"}
    [executed], _ = m.merge_plans([_row("x1", [plan], [ref])])
    assert executed == ref and type(executed["world_size"]) is int


def test_executions_follow_the_campaign_order_not_the_agent_order():
    m = _exp2()
    campaign = [{"result_path": f"p/run{k}"} for k in range(5)]
    by_agent = [campaign[k] for k in (0, 2, 4, 1, 3)]
    assert m.campaign_order(campaign, by_agent) == campaign


def test_campaign_order_rejects_inputs_that_match_no_planned_execution():
    m = _exp2()
    with pytest.raises(SystemExit, match="match no planned"):
        m.campaign_order([{"result_path": "p/run1"}], [{"result_path": "p/elsewhere"}])


def test_merge_requires_every_request_exactly_once():
    m = _exp2()
    ref = {"result_path": "p/run1", "file_limit": "11"}
    with pytest.raises(SystemExit, match="x2"):
        m.check_complete([_row("x1", [ref], [ref])], ["x1", "x2"])
    with pytest.raises(SystemExit, match="x1"):
        m.check_complete([_row("x1", [ref], [ref])] * 2, ["x1"])
    m.check_complete([_row("x1", [ref], [ref])], ["x1"])


def test_sweep_requests_are_refused(tmp_path):
    m = _exp2()
    with pytest.raises(SystemExit, match="sweep"):
        m.cmd_requests(["--kind", "sweep", "--out", str(tmp_path / "r.json")])
