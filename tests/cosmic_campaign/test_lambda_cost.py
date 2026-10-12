"""Cosmic AI Lambda cost: billed duration per execution window, priced and summarized."""

import json
import statistics

import pytest

from results.cosmic_lambda_cost import (
    MissingBilledUsageError,
    collect_usage,
    execution_cost,
    query_billed_usage,
    render_cost_chart,
    summarize_costs,
)

EXECUTOR = "cosmic-executor"


class _FakeLogs:
    def __init__(self, results_by_window, pending_polls=0, status="Complete"):
        self.results_by_window = results_by_window
        self.pending_polls = pending_polls
        self.status = status
        self.queries = []

    def start_query(self, logGroupNames, startTime, endTime, queryString):
        self.queries.append({"groups": logGroupNames, "start": startTime, "end": endTime, "query": queryString})
        return {"queryId": str(len(self.queries) - 1)}

    def get_query_results(self, queryId):
        if self.pending_polls:
            self.pending_polls -= 1
            return {"status": "Running", "results": []}
        q = self.queries[int(queryId)]
        rows = self.results_by_window.get(q["start"], [])
        return {"status": self.status, "results": [
            [{"field": "@log", "value": f"123:/aws/lambda/{fn}"}, {"field": "n", "value": str(n)},
             {"field": "billed_ms", "value": str(ms)}, {"field": "mem", "value": str(mb * 1000000)}]
            for fn, n, ms, mb in rows]}


def _row(name, arm, workers, start_epoch, duration=40.0, phase="measured", index=1, partition=25, status="SUCCEEDED",
         experiment="exp2"):
    from datetime import datetime, timezone
    start = datetime.fromtimestamp(start_epoch, tz=timezone.utc).isoformat()
    stop = datetime.fromtimestamp(start_epoch + duration, tz=timezone.utc).isoformat()
    return {"experiment": experiment, "arm": arm, "name": name, "status": status, "phase": phase,
            "workers": workers, "slot": ["scaling", partition, 1.0, 512, workers, phase, index],
            "start": start, "stop": stop, "duration_s": duration}


def test_billed_usage_is_read_per_function_for_the_execution_window_plus_slack():
    logs = _FakeLogs({1000: [(EXECUTOR, 4, 40000, 10240), ("summarize", 1, 3000, 128)]})
    usage = query_billed_usage(logs, ["/aws/lambda/x"], 1000, 1040, slack_s=2, poll_s=0)
    assert logs.queries[0]["start"] == 1000 and logs.queries[0]["end"] == 1042
    assert usage[EXECUTOR] == {"invocations": 4, "billed_ms": 40000.0, "memory_mb": 10240.0}
    assert usage["summarize"]["memory_mb"] == 128.0


def test_billed_usage_waits_for_the_query_to_finish():
    logs = _FakeLogs({1000: [(EXECUTOR, 1, 1000, 1024)]}, pending_polls=2)
    assert query_billed_usage(logs, ["g"], 1000, 1001, slack_s=0, poll_s=0)[EXECUTOR]["invocations"] == 1


def test_a_failed_query_is_an_error_not_an_empty_usage():
    logs = _FakeLogs({}, status="Failed")
    with pytest.raises(MissingBilledUsageError):
        query_billed_usage(logs, ["g"], 1000, 1001, slack_s=0, poll_s=0)


def test_cost_prices_gb_seconds_and_requests_across_functions():
    usage = {EXECUTOR: {"invocations": 4, "billed_ms": 40000.0, "memory_mb": 10240.0},
             "summarize": {"invocations": 1, "billed_ms": 2000.0, "memory_mb": 128.0}}
    cost = execution_cost(usage, gb_second_usd=0.5, request_usd=0.01)
    gb_seconds = 40.0 * 10.0 + 2.0 * 0.125
    assert cost["gb_seconds"] == pytest.approx(gb_seconds)
    assert cost["usd"] == pytest.approx(gb_seconds * 0.5 + 5 * 0.01)


def test_usage_is_collected_once_per_successful_execution_and_cached_ones_are_not_requeried():
    rows = [_row("a1", "A", 4, 1000), _row("b1", "B", 4, 2000), _row("f1", "B", 4, 3000, status="FAILED")]
    cache = {"a1": {EXECUTOR: {"invocations": 4, "billed_ms": 1.0, "memory_mb": 1.0}}}
    logs = _FakeLogs({2000: [(EXECUTOR, 4, 8000, 1024)]})
    usage = collect_usage(rows, logs, ["g"], cache, slack_s=0, poll_s=0)
    assert [q["start"] for q in logs.queries] == [2000]
    assert set(usage) == {"a1", "b1"}


def test_summary_averages_measured_runs_and_drops_windows_that_caught_another_execution():
    rows = [_row("a1", "A", 4, 1000, index=1), _row("a2", "A", 4, 2000, index=2),
            _row("b1", "B", 4, 3000, index=1), _row("b2", "B", 4, 4000, index=2)]
    usage = {"a1": {EXECUTOR: {"invocations": 4, "billed_ms": 1000.0, "memory_mb": 1024.0}},
             "a2": {EXECUTOR: {"invocations": 4, "billed_ms": 3000.0, "memory_mb": 1024.0}},
             "b1": {EXECUTOR: {"invocations": 4, "billed_ms": 5000.0, "memory_mb": 1024.0}},
             "b2": {EXECUTOR: {"invocations": 6, "billed_ms": 9000.0, "memory_mb": 1024.0}}}
    executions, summary = summarize_costs(rows, usage, EXECUTOR, gb_second_usd=1.0, request_usd=0.0)
    assert {e["name"]: e["attributed"] for e in executions} == {"a1": True, "a2": True, "b1": True, "b2": False}
    by_arm = {s["arm"]: s for s in summary}
    assert by_arm["A"]["n"] == 2
    assert by_arm["A"]["usd_mean"] == pytest.approx(2.0)
    assert by_arm["A"]["usd_std"] == pytest.approx(statistics.stdev([1.0, 3.0]))
    assert (by_arm["B"]["n"], by_arm["B"]["usd_std"]) == (1, None)


def test_rows_from_a_labelled_manifest_are_kept_apart_from_the_same_arm():
    rows = [_row("b1", "B", 4, 1000), {**_row("r1", "B", 4, 2000), "variant": "run1"}]
    usage = {n: {EXECUTOR: {"invocations": 4, "billed_ms": 1000.0, "memory_mb": 1024.0}} for n in ("b1", "r1")}
    _, summary = summarize_costs(rows, usage, EXECUTOR, gb_second_usd=1.0, request_usd=0.0)
    assert sorted((s["arm"], s["variant"]) for s in summary) == [("B", ""), ("B", "run1")]


def test_cost_ratio_is_averaged_over_run_groups_with_the_sample_std_across_them():
    from results.cosmic_lambda_cost import cost_ratio_by_run_group
    rows, usage = [], {}
    for arm, costs in (("A", (1.0, 2.0, 2.0)), ("B", (3.0, 3.0, 5.0))):
        for run, ms in enumerate(costs, start=1):
            name = f"{arm}{run}"
            rows.append(_row(name, arm, 4, 1000 * run + (0 if arm == "A" else 500), index=run))
            usage[name] = {EXECUTOR: {"invocations": 4, "billed_ms": ms * 1000, "memory_mb": 1024.0}}
    rows.append({**_row("old1", "B", 4, 9000, index=1), "variant": "run1"})
    usage["old1"] = {EXECUTOR: {"invocations": 4, "billed_ms": 99000.0, "memory_mb": 1024.0}}
    executions, _ = summarize_costs(rows, usage, EXECUTOR, gb_second_usd=1.0, request_usd=0.0)
    ratio = cost_ratio_by_run_group(executions)[0]
    per_run = [3.0 / 1.0, 3.0 / 2.0, 5.0 / 2.0]
    assert ratio["run_pairs"] == 3
    assert ratio["cost_ratio_mean"] == pytest.approx(statistics.mean(per_run))
    assert ratio["cost_ratio_std"] == pytest.approx(statistics.stdev(per_run))


def test_cost_chart_is_written(tmp_path):
    rows, usage = [], {}
    for arm, ms in (("A", 1000.0), ("B", 3000.0)):
        for workers, start in ((11, 1000), (41, 2000)):
            name = f"{arm}{workers}"
            rows.append(_row(name, arm, workers, start + (0 if arm == "A" else 500)))
            usage[name] = {EXECUTOR: {"invocations": workers, "billed_ms": ms * workers, "memory_mb": 1024.0}}
    rows.append({**_row("r11", "B", 11, 5000), "variant": "run1"})
    usage["r11"] = {EXECUTOR: {"invocations": 11, "billed_ms": 9000.0, "memory_mb": 1024.0}}
    executions, summary = summarize_costs(rows, usage, EXECUTOR, gb_second_usd=1.0, request_usd=0.0)
    names = render_cost_chart(executions, summary, str(tmp_path), "png", 60)
    assert set(names) == {"exp2_lambda_cost_25MB.png", "exp2_lambda_cost_ratio.png"}
    assert all((tmp_path / n).exists() for n in names)