"""Cosmic AI results computed the way the published AI-for-Astronomy scripts computed them."""

import statistics
from datetime import datetime, timedelta, timezone

STARTED = datetime(2026, 10, 8, 21, 3, 20)

import pandas as pd
import pytest

from results.cosmic_paper_metrics import (
    MissingMapStateError,
    fetch_map_seconds,
    iqr_mean,
    map_seconds_from_log_rows,
    paper_cost_estimate,
    per_run_metrics,
    summarize_runs,
)

T0 = datetime(2026, 10, 15, 9, 0, 0, tzinfo=timezone.utc)


def _event(kind, seconds):
    return {"type": kind, "timestamp": T0 + timedelta(seconds=seconds)}


class _FakeStepFunctions:
    def __init__(self, events, page_size):
        self.events = events
        self.page_size = page_size
        self.calls = 0

    def get_execution_history(self, executionArn, reverseOrder=False, maxResults=None, nextToken=None):
        self.calls += 1
        ordered = list(reversed(self.events)) if reverseOrder else list(self.events)
        start = int(nextToken or 0)
        page = ordered[start:start + self.page_size]
        response = {"events": page}
        if start + self.page_size < len(ordered):
            response["nextToken"] = str(start + self.page_size)
        return response


def _history(map_start, map_stop, iterations):
    events = [_event("ExecutionStarted", 0), _event("TaskStateEntered", 0.1), _event("TaskStateExited", 1.0),
              _event("MapStateEntered", map_start), _event("MapStateStarted", map_start + 0.01)]
    events += [_event("MapIterationSucceeded", map_start + 1) for _ in range(iterations)]
    events += [_event("MapStateSucceeded", map_stop - 0.01), _event("MapStateExited", map_stop),
               _event("TaskStateEntered", map_stop + 0.1), _event("ExecutionSucceeded", map_stop + 5)]
    return events


def test_map_seconds_is_map_state_entered_to_exited():
    sfn = _FakeStepFunctions(_history(1.5, 29.5, 20), page_size=4)
    assert fetch_map_seconds(sfn, "arn") == pytest.approx(28.0)


def test_map_seconds_reads_only_the_ends_of_a_long_history():
    sfn = _FakeStepFunctions(_history(1.5, 29.5, 5000), page_size=10)
    fetch_map_seconds(sfn, "arn")
    assert sfn.calls <= 4


def test_execution_without_a_map_state_is_rejected():
    events = [_event("ExecutionStarted", 0), _event("ExecutionSucceeded", 5)]
    with pytest.raises(MissingMapStateError):
        fetch_map_seconds(_FakeStepFunctions(events, page_size=10), "arn")


def _log_row(log, kind, timestamp):
    return [{"field": "@log", "value": f"123:/aws/lambda/{log}"}, {"field": "@type", "value": kind},
            {"field": "@timestamp", "value": timestamp}]


def test_map_seconds_from_logs_runs_from_initialize_report_to_summarize_start():
    rows = [_log_row("init", "START", "2026-10-08 21:03:20.000"), _log_row("init", "REPORT", "2026-10-08 21:03:21.500"),
            _log_row("summarize", "START", "2026-10-08 21:03:49.500"),
            _log_row("summarize", "REPORT", "2026-10-08 21:03:50.000")]
    assert map_seconds_from_log_rows(rows, "init", "summarize", STARTED) == pytest.approx(28.0)


def test_map_seconds_from_logs_ignores_the_previous_execution_in_the_window():
    rows = [_log_row("summarize", "START", "2026-10-08 21:03:19.000"),
            _log_row("init", "REPORT", "2026-10-08 21:03:21.500"),
            _log_row("summarize", "START", "2026-10-08 21:03:49.500")]
    assert map_seconds_from_log_rows(rows, "init", "summarize", STARTED) == pytest.approx(28.0)


def test_map_seconds_from_logs_ignores_the_next_execution_in_the_window():
    rows = [_log_row("init", "REPORT", "2026-10-08 21:03:21.500"),
            _log_row("summarize", "START", "2026-10-08 21:03:49.500"),
            _log_row("init", "REPORT", "2026-10-08 21:03:55.000")]
    assert map_seconds_from_log_rows(rows, "init", "summarize", STARTED) == pytest.approx(28.0)


def test_map_seconds_from_logs_needs_both_ends():
    with pytest.raises(MissingMapStateError):
        map_seconds_from_log_rows([_log_row("init", "REPORT", "2026-10-08 21:03:21.500")], "init", "summarize",
                                  STARTED)


def test_iqr_mean_drops_values_beyond_one_and_a_half_iqr():
    assert iqr_mean([5.0, 5.2, 5.4, 5.6, 40.0]) == pytest.approx(5.3)


def test_iqr_mean_keeps_everything_when_nothing_is_an_outlier():
    assert iqr_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def _workers(execution, cpu_times, bits_each=1000.0, memory_each=10.0):
    return pd.DataFrame({"execution": execution, "total_cpu_time (seconds)": cpu_times,
                         "throughput_bps": [bits_each / t for t in cpu_times],
                         "total_cpu_memory (MB)": memory_each})


def test_per_run_metrics_follow_the_published_aggregation():
    runs = per_run_metrics(pd.concat([_workers("e1", [5.0, 5.2, 5.4, 5.6, 40.0]), _workers("e2", [2.0, 4.0])]))
    first = runs.set_index("execution").loc["e1"]
    assert first.cpu_time_s == pytest.approx(5.3)
    assert first.total_bits == pytest.approx(5000.0)
    assert first.memory_mb == pytest.approx(50.0)
    assert first.throughput_sum_bps == pytest.approx(sum(1000.0 / t for t in [5.0, 5.2, 5.4, 5.6, 40.0]))
    assert first.workers == 5


def test_summary_reports_mean_sample_std_and_run_count():
    runs = pd.DataFrame({"system": "A", "partition_mb": 25, "data_gb": 1.0, "batch_size": 512,
                         "map_s": [10.0, 12.0, 14.0]})
    row = summarize_runs(runs, ["map_s"]).iloc[0]
    assert row.map_s_mean == pytest.approx(12.0)
    assert row.map_s_std == pytest.approx(statistics.stdev([10.0, 12.0, 14.0]))
    assert row.n == 3


def test_paper_cost_is_requests_times_duration_times_memory_times_price():
    assert paper_cost_estimate(517, 5.70, 2.8, 0.0000166667) == pytest.approx(517 * 5.70 * 2.8 * 0.0000166667)
