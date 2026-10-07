"""Campaign driver: planning, naming, cost, sequential execution and resume."""

import importlib.util
import json
import os

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _driver():
    path = os.path.join(ROOT, "target", "aws", "scripts", "experiment", "cosmic_campaign_run.py")
    spec = importlib.util.spec_from_file_location("cosmic_campaign_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(*extra):
    return _driver().build_parser().parse_args(["--dry-run", *extra])


def test_default_campaign_is_one_cold_start_run_then_the_published_run_counts():
    """The published campaign ran 3 runs per scaling configuration and 4 per batch
    size. Each configuration first gets one cold-start run, reported rather than
    discarded; the 1 and 2 worker baselines follow the scaling count."""
    m = _driver()
    planned = m.plan_campaign(_args())
    configs = {p.slot.configuration for p in planned}
    assert len(configs) == 28 + 5 + 2
    for c in configs:
        phases = [p.slot.phase for p in planned if p.slot.configuration == c]
        measured = {"scaling": 3, "batch": 4, "baseline": 3}[c.series]
        assert phases == ["warmup"] + ["measured"] * measured, c
    assert len(planned) == 28 * 4 + 5 * 5 + 2 * 4
    assert {p.sfn_input["script"] for p in planned} == {"/tmp/Anomaly Detection/Inference/inference.py"}


def test_run_counts_remain_overridable():
    m = _driver()
    planned = m.plan_campaign(_args("--series", "batch", "--warmup-runs", "0", "--batch-measured-runs", "2"))
    assert len(planned) == 5 * 2
    assert {p.slot.phase for p in planned} == {"measured"}


def test_series_and_max_workers_filter_the_campaign_for_a_ramp():
    m = _driver()
    planned = m.plan_campaign(_args("--series", "scaling", "--max-workers", "41"))
    assert max(p.slot.configuration.workers for p in planned) <= 41
    assert {p.slot.configuration.series for p in planned} == {"scaling"}


def test_execution_names_are_valid_and_unique():
    m = _driver()
    planned = m.plan_campaign(_args())
    names = [p.name for p in planned]
    assert len(names) == len(set(names))
    for name in names:
        assert len(name) <= 80
        assert all(ch.isalnum() or ch in "-_" for ch in name), name
    assert any("12.6GB" in n.replace("12p6GB", "12.6GB") for n in names)


def test_cost_estimate_multiplies_workers_by_seconds_and_memory():
    m = _driver()
    planned = m.plan_campaign(_args("--series", "baseline"))
    cost = m.estimate_cost(planned, worker_seconds=10, memory_gb=10, gb_second_usd=0.00001)
    assert cost["executions"] == 8
    assert cost["worker_invocations"] == 4 * (1 + 2)
    assert cost["gb_seconds"] == 12 * 10 * 10
    assert cost["usd"] == pytest.approx(1200 * 0.00001)


class _FakeSfn:
    def __init__(self, outcomes=None, running=0):
        self.started = []
        self.outcomes = outcomes or {}
        self.running = running

    def list_executions(self, stateMachineArn, statusFilter, maxResults):
        assert statusFilter == "RUNNING"
        return {"executions": [{"name": "other"}] * self.running}

    def start_execution(self, stateMachineArn, name, input):
        self.started.append(name)
        return {"executionArn": f"arn:exec:{name}", "startDate": 0}

    def describe_execution(self, executionArn):
        name = executionArn.split(":")[-1]
        return {"status": self.outcomes.get(name, "SUCCEEDED"),
                "startDate": __import__("datetime").datetime(2026, 10, 6, 12, 0, 0),
                "stopDate": __import__("datetime").datetime(2026, 10, 6, 12, 0, 30)}


def _planned(m, *extra):
    return m.plan_campaign(_args("--series", "baseline", *extra))


def test_runs_every_slot_in_order_and_records_it(tmp_path):
    m = _driver()
    planned = _planned(m)
    sfn = _FakeSfn()
    manifest = tmp_path / "m.jsonl"
    rows = m.run_campaign(planned, sfn, "arn:sm", str(manifest), poll_s=0, timeout_s=10, sleep=lambda s: None)
    assert sfn.started == [p.name for p in planned]
    assert [r["status"] for r in rows] == ["SUCCEEDED"] * len(planned)
    assert len(manifest.read_text().splitlines()) == len(planned)
    assert rows[0]["duration_s"] == 30.0


def test_resume_skips_succeeded_slots_and_retries_failed_ones(tmp_path):
    m = _driver()
    first = _planned(m)
    manifest = tmp_path / "m.jsonl"
    failing = {first[1].name: "FAILED"}
    m.run_campaign(first, _FakeSfn(failing), "arn:sm", str(manifest), 0, 10, sleep=lambda s: None)

    second = _planned(m)
    sfn = _FakeSfn()
    m.run_campaign(second, sfn, "arn:sm", str(manifest), 0, 10, sleep=lambda s: None)
    assert len(sfn.started) == 1
    assert m.slot_key(second[1].slot) in {m.slot_key(p.slot) for p in second if p.name in sfn.started}
    assert sfn.started[0] != first[1].name


def test_refuses_to_start_while_the_state_machine_is_busy(tmp_path):
    m = _driver()
    with pytest.raises(m.ConcurrentExecutionError, match="RUNNING"):
        m.run_campaign(_planned(m), _FakeSfn(running=1), "arn:sm", str(tmp_path / "m.jsonl"),
                       0, 10, sleep=lambda s: None)


def test_a_failed_execution_is_recorded_and_the_campaign_continues(tmp_path):
    m = _driver()
    planned = _planned(m)
    sfn = _FakeSfn({planned[0].name: "FAILED"})
    rows = m.run_campaign(planned, sfn, "arn:sm", str(tmp_path / "m.jsonl"), 0, 10, sleep=lambda s: None)
    assert rows[0]["status"] == "FAILED"
    assert len(sfn.started) == len(planned)


def test_a_shared_manifest_keeps_arms_and_experiments_apart(tmp_path):
    """Resume is keyed by experiment and arm as well as slot: an Arm A campaign in the
    manifest must not make a later Arm B or exp2 campaign look already done."""
    m = _driver()
    manifest = str(tmp_path / "m.jsonl")
    m.run_campaign(_planned(m), _FakeSfn(), "arn:sm", manifest, 0, 10, sleep=lambda s: None)

    for extra in (("--arm", "B"), ("--experiment", "exp2")):
        sfn = _FakeSfn()
        planned = _planned(m, *extra)
        m.run_campaign(planned, sfn, "arn:sm", manifest, 0, 10, sleep=lambda s: None)
        assert sfn.started == [p.name for p in planned], extra


def test_manifest_rows_record_experiment_and_arm(tmp_path):
    m = _driver()
    rows = m.run_campaign(_planned(m, "--arm", "B"), _FakeSfn(), "arn:sm", str(tmp_path / "m.jsonl"),
                          0, 10, sleep=lambda s: None)
    assert {(r["experiment"], r["arm"]) for r in rows} == {("exp1", "B")}


class _LaggingSfn(_FakeSfn):
    """list_executions keeps reporting each finished execution as RUNNING once more."""

    def __init__(self):
        super().__init__()
        self.stale = []

    def list_executions(self, stateMachineArn, statusFilter, maxResults):
        names, self.stale = self.stale, []
        return {"executions": [{"name": n} for n in names]}

    def describe_execution(self, executionArn):
        self.stale.append(executionArn.split(":")[-1])
        return super().describe_execution(executionArn)


def test_a_lagging_running_list_does_not_stop_the_campaign(tmp_path):
    m = _driver()
    planned = _planned(m)
    sfn = _LaggingSfn()
    rows = m.run_campaign(planned, sfn, "arn:sm", str(tmp_path / "m.jsonl"), 0, 10, sleep=lambda s: None)
    assert [r["status"] for r in rows] == ["SUCCEEDED"] * len(planned)


def test_a_foreign_execution_that_stays_running_still_stops_the_campaign(tmp_path):
    m = _driver()
    sfn = _FakeSfn(running=1)
    slept = []
    with pytest.raises(m.ConcurrentExecutionError):
        m.run_campaign(_planned(m), sfn, "arn:sm", str(tmp_path / "m.jsonl"), 5, 10, sleep=slept.append)
    assert slept and not sfn.started


class _StuckSfn(_FakeSfn):
    """The first execution never finishes; it stays in the RUNNING list."""

    def list_executions(self, stateMachineArn, statusFilter, maxResults):
        return {"executions": [{"name": n} for n in self.started[:1]]}

    def describe_execution(self, executionArn):
        return {"status": "RUNNING", "startDate": None, "stopDate": None}


def test_an_execution_the_driver_gave_up_on_still_blocks_the_next_start(tmp_path):
    m = _driver()
    sfn = _StuckSfn()
    with pytest.raises(m.ConcurrentExecutionError):
        m.run_campaign(_planned(m), sfn, "arn:sm", str(tmp_path / "m.jsonl"), 0, -1, sleep=lambda s: None)
    assert len(sfn.started) == 1


class _SummaryFailSfn(_FakeSfn):
    def describe_execution(self, executionArn):
        return {**super().describe_execution(executionArn),
                "output": json.dumps({"statusCode": 404, "body": "No files found"})}


def test_a_succeeded_execution_whose_summarizer_failed_is_not_recorded_as_succeeded(tmp_path):
    m = _driver()
    rows = m.run_campaign(_planned(m), _SummaryFailSfn(), "arn:sm", str(tmp_path / "m.jsonl"), 0, 10,
                          sleep=lambda s: None)
    assert {r["status"] for r in rows} == {"SUMMARY_FAILED"}
    assert m.completed_slots(str(tmp_path / "m.jsonl")) == set()
