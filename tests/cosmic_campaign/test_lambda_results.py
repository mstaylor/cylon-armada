"""Cosmic AI Lambda results: validation, warm up exclusion and per-cell statistics."""

import json
import statistics

import pytest

from results.cosmic_lambda_results import (
    IncompleteExecutionError,
    load_execution,
    summarize,
)


class _FakeS3:
    def __init__(self, objects):
        self.objects = objects

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        body = json.dumps(self.objects[(Bucket, Key)]).encode()
        return {"Body": __import__("io").BytesIO(body)}


def _arm_a_records(n, total_s=3.0):
    return [{"num_samples": 511.0, "num_batches": 1, "batch_size": 512,
             "total_s": total_s + r, "inference_s": 1.0 + r} for r in range(n)]


def test_load_execution_restores_integer_counts_without_needing_rank_ids():
    s3 = _FakeS3({("b", "p/combined_data.json"): _arm_a_records(2),
                  ("b", "p/aggregate_metrics.json"): {"aggregate_s": 0.5, "ranks_aggregated": 2}})
    loaded = load_execution(s3, "b", "p", workers=2)
    assert all(isinstance(r["num_samples"], int) for r in loaded["records"])
    assert loaded["metrics"]["aggregate_s"] == 0.5


@pytest.mark.parametrize("records,ranks", [(1, 2), (2, 1)])
def test_load_execution_rejects_a_short_rank_set(records, ranks):
    s3 = _FakeS3({("b", "p/combined_data.json"): _arm_a_records(records),
                  ("b", "p/aggregate_metrics.json"): {"aggregate_s": 0.5, "ranks_aggregated": ranks}})
    with pytest.raises(IncompleteExecutionError, match="p"):
        load_execution(s3, "b", "p", workers=2)


def _row(phase, index, duration, status="SUCCEEDED", workers=2, arm="A", experiment="exp1"):
    return {"experiment": experiment, "arm": arm,
            "slot": ["baseline", 100, 0.1953, 512, workers, phase, index], "phase": phase,
            "status": status, "workers": workers, "duration_s": duration,
            "result_path": f"p/{phase}{index}"}


def _loader(total=3.0):
    return lambda row: {"records": _arm_a_records(row["workers"], total),
                        "metrics": {"aggregate_s": 0.5, "ranks_aggregated": row["workers"]}}


def test_measured_statistics_exclude_the_cold_start_run_and_use_sample_std():
    rows = [_row("warmup", 0, 99.0)] + [_row("measured", k, d) for k, d in [(1, 10), (2, 12), (3, 14), (4, 16)]]
    cells = {c["phase"]: c for c in summarize(rows, _loader(), measured_runs=4)}
    assert cells["cold_start"]["duration_s_mean"] == 99.0
    cell = cells["measured"]
    assert cell["n"] == 4 and cell["complete"] is True
    assert cell["duration_s_mean"] == 13.0
    assert cell["duration_s_std"] == pytest.approx(statistics.stdev([10, 12, 14, 16]))
    assert cell["total_s_max_mean"] == 4.0


def test_execution_runs_lists_each_validated_run_with_its_run_group():
    from results.cosmic_lambda_results import execution_runs
    rows = [_row("warmup", 0, 99.0), _row("measured", 1, 10), _row("measured", 2, 12, status="FAILED"),
            _row("measured", 3, 14)]
    runs = execution_runs(rows, _loader())
    assert [(r["phase"], r["run"], r["duration_s"]) for r in runs] == [
        ("cold_start", 0, 99.0), ("measured", 1, 10), ("measured", 3, 14)]
    assert runs[1]["workers"] == 2 and runs[1]["total_s_max"] == 4.0


def test_execution_runs_record_the_account_and_the_original_a_rerun_replaced():
    from results.cosmic_lambda_results import execution_runs
    rerun = {**_row("measured", 2, 11), "arn": "arn:aws:states:us-east-1:881908115028:execution:wf:rerun5",
             "rerun_of": "exp1-r2"}
    original = {**_row("measured", 1, 10), "arn": "arn:aws:states:us-east-1:448324707516:execution:wf:exp1-r1"}
    runs = execution_runs([original, rerun], _loader())
    assert [(r["account"], r["rerun_of"]) for r in runs] == [("448324707516", None), ("881908115028", "exp1-r2")]


def test_each_row_is_read_from_the_store_of_the_account_that_ran_it():
    from results.cosmic_lambda_results import store_for
    stores = {"881908115028": ("cosmicai-client", "cosmicai")}
    default = ("cylon-client", "cosmicai-data-cylon")
    rerun = {"arn": "arn:aws:states:us-east-1:881908115028:execution:wf:rerun5"}
    original = {"arn": "arn:aws:states:us-east-1:448324707516:execution:wf:exp1"}
    assert store_for(rerun, stores, default) == ("cosmicai-client", "cosmicai")
    assert store_for(original, stores, default) == default


def test_failed_and_incomplete_runs_lower_n_and_flag_the_cell():
    rows = [_row("measured", 1, 10), _row("measured", 2, 12, status="FAILED"),
            _row("measured", 3, 14), _row("measured", 4, 16)]

    def loader(row):
        if row["slot"][-1] == 4:
            raise IncompleteExecutionError("p/measured4 short")
        return _loader()(row)

    [cell] = summarize(rows, loader, measured_runs=4)
    assert cell["n"] == 2 and cell["complete"] is False
    assert cell["duration_s_mean"] == 12.0


def test_a_single_measured_run_has_no_std():
    [cell] = summarize([_row("measured", 1, 10)], _loader(), measured_runs=4)
    assert cell["n"] == 1 and cell["duration_s_std"] is None and cell["complete"] is False


def test_each_arm_and_experiment_gets_its_own_cell():
    rows = [_row("measured", 1, 10, arm="A"), _row("measured", 1, 20, arm="B"),
            _row("measured", 1, 30, experiment="exp2")]
    cells = {(c["experiment"], c["arm"]): c for c in summarize(rows, _loader(), measured_runs=1)}
    assert set(cells) == {("exp1", "A"), ("exp1", "B"), ("exp2", "A")}
    assert cells[("exp1", "B")]["duration_s_mean"] == 20


def _series_row(series, phase, index, duration, workers=2):
    return {"experiment": "exp1", "arm": "A", "slot": [series, 100, 1, 512, workers, phase, index],
            "phase": phase, "status": "SUCCEEDED", "workers": workers, "duration_s": duration,
            "result_path": f"p/{series}/{phase}{index}"}


def test_completeness_follows_each_series_published_run_count():
    rows = ([_series_row("scaling", "measured", k, 10) for k in (1, 2, 3)]
            + [_series_row("batch", "measured", k, 10) for k in (1, 2, 3)])
    cells = {c["series"]: c for c in summarize(rows, _loader(), {"scaling": 3, "batch": 4, "baseline": 3})
             if c["phase"] == "measured"}
    assert cells["scaling"]["complete"] is True
    assert cells["batch"]["complete"] is False and cells["batch"]["n"] == 3


def test_the_cold_start_run_is_reported_in_its_own_row_not_discarded():
    rows = [_series_row("scaling", "warmup", 0, 50)] + [_series_row("scaling", "measured", k, 10) for k in (1, 2, 3)]
    cells = {c["phase"]: c for c in summarize(rows, _loader(), {"scaling": 3})}
    assert cells["cold_start"]["n"] == 1 and cells["cold_start"]["duration_s_mean"] == 50
    assert cells["measured"]["n"] == 3 and cells["measured"]["duration_s_mean"] == 10


def test_cold_worker_fraction_counts_ranks_whose_init_exceeds_the_threshold():
    def loader(row):
        init = [40.0, 2.0] if row["phase"] == "warmup" else [2.0, 2.5]
        records = [{**r, "init_s": i} for r, i in zip(_arm_a_records(2), init)]
        return {"records": records, "metrics": {"aggregate_s": 0.5, "ranks_aggregated": 2}}

    rows = [_series_row("scaling", "warmup", 0, 50), _series_row("scaling", "measured", 1, 10)]
    cells = {c["phase"]: c for c in summarize(rows, loader, {"scaling": 1}, cold_init_threshold_s=10)}
    assert cells["cold_start"]["cold_worker_fraction_mean"] == 0.5
    assert cells["measured"]["cold_worker_fraction_mean"] == 0.0


class _DatedS3(_FakeS3):
    def __init__(self, objects, modified):
        super().__init__(objects)
        self.modified = modified

    def get_object(self, Bucket, Key):
        return {**super().get_object(Bucket, Key), "LastModified": self.modified}


def _execution_objects(n=2):
    return {("b", "p/combined_data.json"): _arm_a_records(n),
            ("b", "p/aggregate_metrics.json"): {"aggregate_s": 0.5, "ranks_aggregated": n}}


def test_results_written_before_the_execution_started_are_rejected_as_stale():
    from datetime import datetime, timezone

    s3 = _DatedS3(_execution_objects(), datetime(2026, 10, 6, 11, 0, tzinfo=timezone.utc))
    with pytest.raises(IncompleteExecutionError, match="stale"):
        load_execution(s3, "b", "p", workers=2, started_at="2026-10-06T12:00:00+00:00")
    fresh = _DatedS3(_execution_objects(), datetime(2026, 10, 6, 12, 1, tzinfo=timezone.utc))
    assert load_execution(fresh, "b", "p", workers=2, started_at="2026-10-06T12:00:00+00:00")


def test_missing_results_files_are_an_incomplete_execution_not_a_crash():
    with pytest.raises(IncompleteExecutionError, match="missing"):
        load_execution(_FakeS3({}), "b", "p", workers=2)


def test_collect_s_is_comparable_between_arm_a_and_arm_b():
    """Arm A collects by per-rank S3 writes then the summarizer's reads; Arm B by the
    first gather, the trailing gather and rank 0's write. aggregate_s alone covers only
    the last step in Arm B, so comparing it would favour B."""
    records = [{**r, "publish_s": p} for r, p in zip(_arm_a_records(2), (0.2, 0.3))]

    def arm(metrics):
        return lambda row: {"records": records, "metrics": {"ranks_aggregated": 2, **metrics}}

    rows_a = [_row("measured", 1, 10, arm="A")]
    rows_b = [_row("measured", 1, 10, arm="B")]
    [a] = summarize(rows_a, arm({"aggregate_s": 2.0}), measured_runs=1)
    [b] = summarize(rows_b, arm({"aggregate_s": 0.1, "trailing_gather_s": 0.4}), measured_runs=1)
    assert a["collect_s_mean"] == pytest.approx(0.3 + 2.0)
    assert b["collect_s_mean"] == pytest.approx(0.3 + 0.4 + 0.1)


def test_cold_worker_fraction_prefers_the_containers_own_cold_start_flag():
    """init_s misses cold containers once the image is cached; the executor's flag does not."""
    def loader(row):
        records = [{**r, "init_s": 2.0, "cold_start": c} for r, c in zip(_arm_a_records(2), (1, 0))]
        return {"records": records, "metrics": {"aggregate_s": 0.5, "ranks_aggregated": 2}}

    [cell] = summarize([_row("measured", 1, 10)], loader, measured_runs=1, cold_init_threshold_s=10)
    assert cell["cold_worker_fraction_mean"] == 0.5


def test_a_cold_start_that_succeeded_on_a_retry_is_complete():
    rows = [_row("warmup", 0, 300.0, status="FAILED"), _row("warmup", 0, 40.0)]
    cold = [r for r in summarize(rows, _loader(), measured_runs=3) if r["phase"] == "cold_start"][0]
    assert (cold["n"], cold["complete"]) == (1, True)
    assert cold["duration_s_mean"] == 40.0


def test_a_measured_run_that_succeeded_on_a_retry_counts_once():
    rows = [_row("measured", 1, 300.0, status="FAILED")] + [_row("measured", k, 10.0) for k in (1, 2, 3)]
    measured = [r for r in summarize(rows, _loader(), measured_runs=3) if r["phase"] == "measured"][0]
    assert (measured["n"], measured["complete"]) == (3, True)


def _samples_loader(counts):
    return lambda row: {"records": [{"data_path": f"100MB/{i}.pt", "num_samples": float(c)}
                                    for i, c in enumerate(counts[row["experiment"], row["arm"]])],
                        "metrics": {"ranks_aggregated": len(counts[row["experiment"], row["arm"]])}}


def test_partition_samples_lists_every_partition_of_every_successful_execution():
    from results.cosmic_lambda_results import partition_samples
    rows = [_row("measured", 1, 10.0), _row("measured", 2, 10.0, status="FAILED")]
    samples = partition_samples(rows, _samples_loader({("exp1", "A"): [5113, 5000]}))
    assert [(s["run"], s["data_path"], s["num_samples"]) for s in samples] == [
        ("measured1", "100MB/0.pt", 5113), ("measured1", "100MB/1.pt", 5000)]


def test_sample_coverage_flags_a_partition_whose_count_differs_from_the_reference():
    from results.cosmic_lambda_results import compare_samples, partition_samples
    rows = [_row("measured", 1, 10.0), _row("measured", 1, 10.0, arm="B", experiment="exp2"),
            _row("measured", 1, 10.0, arm="A", experiment="exp2")]
    counts = {("exp1", "A"): [5113, 5000], ("exp2", "B"): [5113, 4999], ("exp2", "A"): [5113, 5000]}
    report = compare_samples(partition_samples(rows, _samples_loader(counts)), reference=("exp1", "A"))
    assert [(m["experiment"], m["arm"], m["data_path"], m["num_samples"], m["reference_samples"])
            for m in report["mismatches"]] == [("exp2", "B", "100MB/1.pt", 4999, 5000)]
    assert report["compared"] == 4


def test_sample_coverage_flags_a_partition_the_reference_never_processed():
    from results.cosmic_lambda_results import compare_samples, partition_samples
    rows = [_row("measured", 1, 10.0), _row("measured", 1, 10.0, arm="B", experiment="exp2")]
    counts = {("exp1", "A"): [5113], ("exp2", "B"): [5113, 5000]}
    report = compare_samples(partition_samples(rows, _samples_loader(counts)), reference=("exp1", "A"))
    assert [(m["data_path"], m["reference_samples"]) for m in report["mismatches"]] == [("100MB/1.pt", None)]


def test_abort_markers_under_the_result_path_are_not_counted_as_ranks():
    records = [{**r, "data_path": f"25MB/{i}.pt"} for i, r in enumerate(_arm_a_records(2))]
    markers = [{"source": "campaign_driver", "execution": "x", "status": "FAILED"}]
    s3 = _FakeS3({("b", "p/combined_data.json"): records + markers,
                  ("b", "p/aggregate_metrics.json"): {"aggregate_s": 0.5, "ranks_aggregated": 3}})
    loaded = load_execution(s3, "b", "p", workers=2)
    assert len(loaded["records"]) == 2
