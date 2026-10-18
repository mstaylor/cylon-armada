"""Cosmic AI campaign results written in the AI-for-Astronomy aws/results and aws/figures layout."""

import pandas as pd
import pytest

from results.cosmic_repo_format import format_duration, render_repo_figures, repo_tables


def _runs():
    rows = []
    for partition, data, workers in ((25, 1.0, 41), (25, 2.0, 82), (100, 1.0, 11)):
        for run in (1, 2):
            rows.append({"system": "A", "series": "scaling", "partition_mb": partition, "data_gb": data,
                         "batch_size": 512, "run_id": run, "workers": workers, "map_s": 10.0 * run,
                         "duration_s": 20.0 * run, "cpu_time_s": 5.0 + run, "memory_mb": 100.0,
                         "throughput_sum_bps": 1e9 * run, "total_bits": 8e9, "throughput_map_bps": 8e8 / run})
    for batch in (32, 512):
        for run in (1, 2, 3, 4):
            rows.append({"system": "A", "series": "batch", "partition_mb": 100, "data_gb": 1.0, "batch_size": batch,
                         "run_id": run, "workers": 11, "map_s": 30.0, "duration_s": 40.0, "cpu_time_s": 20.0,
                         "memory_mb": 50.0, "throughput_sum_bps": 2e8, "total_bits": 6e9,
                         "throughput_map_bps": 2e8})
    return pd.DataFrame(rows)


def test_durations_are_written_as_minutes_and_seconds():
    assert format_duration(26.642) == "00:26.642"
    assert format_duration(62.5) == "01:02.500"


def test_result_stats_has_the_published_columns_and_a_mean_row_per_configuration():
    stats = repo_tables(_runs())["result_stats.csv"]
    assert list(stats.columns) == ["partition (MB)", "data (GB)", "run", "total_cpu_time (seconds)",
                                   "total_cpu_memory (MB)", "throughput_bps", "total_bits"]
    group = stats[(stats["partition (MB)"] == 25) & (stats["data (GB)"] == 1.0)]
    assert group.run.tolist() == ["1", "2", "mean"]
    assert group.iloc[2]["total_cpu_time (seconds)"] == pytest.approx(6.5)
    assert group.iloc[2]["throughput_bps"] == pytest.approx(1.5e9)


def test_adjusted_stats_take_throughput_from_the_map_state_time():
    adjusted = repo_tables(_runs())["result_stats_adjusted.csv"]
    first = adjusted[(adjusted["partition (MB)"] == 25) & (adjusted["data (GB)"] == 1.0) & (adjusted.run == "1")]
    row = first.iloc[0]
    assert row.throughput_bps == pytest.approx(8e8)
    assert (row.num_worlds, row["inference_duration (s)"], row["total_duration (s)"]) == (41, 10.0, 20.0)


def test_state_machine_logs_mark_the_batch_sweep():
    logs = repo_tables(_runs())["state_machine_logs.csv"]
    assert list(logs.columns) == ["partition (MB)", "data (GB)", "batch_size", "batch_varying", "run",
                                  "num_worlds", "total_duration (s)", "inference_duration (s)"]
    assert logs[~logs.batch_varying].num_worlds.tolist() == [41, 41, 82, 82, 11, 11]
    assert logs.batch_varying.sum() == 8 and (~logs.batch_varying).sum() == 6


def test_batch_tables_hold_one_mean_row_per_batch_size():
    tables = repo_tables(_runs())
    assert (tables["batch_varying_results.csv"].run == "mean").sum() == 2
    assert (tables["batch_varying_adjusted.csv"].run == "mean").sum() == 2


def test_total_execution_time_lists_every_scaling_run():
    total = repo_tables(_runs())["total_execution_time.csv"]
    assert list(total.columns) == ["partition(MB)", "data(GB)", "run", "num_worlds", "duration"]
    assert len(total) == 6 and total.duration.iloc[0] == "00:20.000"
    assert total.num_worlds.dtype == float


def test_the_four_published_figures_are_written(tmp_path):
    names = render_repo_figures(repo_tables(_runs()), str(tmp_path), "Subexperiment A")
    assert sorted(names) == ["batch_varying_inference_time.jpg", "batch_varying_throughput.jpg",
                             "inference_time.jpg", "throughput.jpg"]
    assert all((tmp_path / n).exists() for n in names)
