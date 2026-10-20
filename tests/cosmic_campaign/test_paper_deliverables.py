"""Cosmic AI Experiment 2 figures and tables in the format of the published Cosmic AI paper."""

import os

import pandas as pd
import pytest

from results.cosmic_paper_deliverables import (METRICS, batch_table, headline_table, mean_std_text,
                                               scaling_table, shared_limits, table6, write_deliverables)
from results.cosmic_repo_format import repo_tables

SYSTEMS = {"published": "Published", "A": "Subexperiment A", "B": "Subexperiment B"}
REAL_METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "target", "shared", "scripts", "experiment",
                                "results", "cosmic_exp2", "exp2_fix", "paper_metrics")


def _summary_row(system, series, partition, data, batch, base, workers):
    row = {"system": system, "series": series, "partition_mb": partition, "data_gb": data, "batch_size": batch,
           "workers_mean": workers, "workers_std": 0.0, "n": 3 if series == "scaling" else 4}
    for i, metric in enumerate(METRICS):
        scale = 1e9 if metric.key == "throughput_map_bps" else 1.0
        row[f"{metric.key}_mean"] = (base + i) * scale
        row[f"{metric.key}_std"] = 0.1 * (i + 1) * scale
    return row


def _summary():
    rows = []
    for system, base in (("published", 30.0), ("A", 20.0), ("B", 10.0)):
        for partition, workers in ((25, 41), (100, 11)):
            for data in (1.0, 12.6):
                rows.append(_summary_row(system, "scaling", partition, data, 512, base + partition / 25 + data,
                                         workers * data))
    for system, base in (("A", 40.0), ("B", 35.0)):
        for batch in (32, 512):
            rows.append(_summary_row(system, "batch", 100, 1.0, batch, base - batch / 64, 11))
    return pd.DataFrame(rows)


def _published_batch():
    rows = []
    for batch in (32, 512):
        for run in (1, 2, 3):
            rows.append({"batch_size": batch, "run": str(run), "total_cpu_time (seconds)": 8.0 + run,
                         "total_cpu_memory (MB)": 1.0, "throughput_bps": 2e8 * run, "total_bits": 8e9,
                         "num_worlds": 41.0, "total_duration (s)": 40.0 + run, "inference_duration (s)": 30.0 + run})
        rows.append({"batch_size": batch, "run": "mean", "total_cpu_time (seconds)": 10.0,
                     "total_cpu_memory (MB)": 1.0, "throughput_bps": 4e8, "total_bits": 8e9, "num_worlds": 41.0,
                     "total_duration (s)": 42.0, "inference_duration (s)": 32.0})
    return pd.DataFrame(rows)


def _cost():
    rows = []
    for partition, requests in ((25, 517), (100, 130)):
        rows.append({"partition_mb": partition, "requests": requests, "cpu_time_s": 5.7, "peak_memory_gb": 2.8,
                     "paper_estimate_usd": 0.16, "system": "published", "billed_usd": None, "billed_usd_std": None,
                     "n": None})
        for system, billed in (("A", 0.8), ("B", 1.2)):
            rows.append({"partition_mb": partition, "requests": requests, "cpu_time_s": 6.0, "peak_memory_gb": 3.0,
                         "paper_estimate_usd": 0.2, "system": system, "billed_usd": billed, "billed_usd_std": 0.01,
                         "n": 3.0})
    return pd.DataFrame(rows)


def test_mean_std_text_keeps_the_requested_decimals():
    assert mean_std_text(28.606, 5.8846, 2) == "28.61 ± 5.88"
    assert mean_std_text(3.8933, 0.7846, 3) == "3.893 ± 0.785"
    assert mean_std_text(float("nan"), float("nan"), 2) == ""


def test_scaling_table_has_one_row_per_configuration_with_summary_values():
    summary = _summary()
    table = scaling_table(summary, SYSTEMS)
    assert list(zip(table["Partition (MB)"], table["Data (GB)"])) == [(25, 1.0), (25, 12.6), (100, 1.0), (100, 12.6)]
    row = table.iloc[1]
    source = summary[(summary.system == "A") & (summary.series == "scaling") & (summary.partition_mb == 25)
                     & (summary.data_gb == 12.6)].iloc[0]
    assert row["End to end time (s) Subexperiment A mean"] == pytest.approx(source.duration_s_mean)
    assert row["Map state time (s) Subexperiment A std"] == pytest.approx(source.map_s_std)
    assert row["Adjusted throughput (billion bps) Subexperiment A mean"] == pytest.approx(
        source.throughput_map_bps_mean / 1e9)
    assert row["Workers"] == pytest.approx(source.workers_mean)


def test_scaling_table_speedup_is_a_end_to_end_over_b_end_to_end():
    table = scaling_table(_summary(), SYSTEMS)
    expected = (table["End to end time (s) Subexperiment A mean"]
                / table["End to end time (s) Subexperiment B mean"])
    assert table["End to end speedup of Subexperiment B over A (x)"].tolist() == pytest.approx(expected.tolist())
    assert (table["End to end speedup of Subexperiment B over A (x)"] > 1).all()


def test_scaling_table_rejects_systems_that_disagree_on_worker_count():
    summary = _summary()
    summary.loc[(summary.system == "B") & (summary.series == "scaling"), "workers_mean"] += 1
    with pytest.raises(ValueError, match="worker"):
        scaling_table(summary, SYSTEMS)


def test_headline_table_lists_every_metric_for_each_partition_at_the_full_dataset():
    table = headline_table(_summary(), SYSTEMS)
    assert table["Partition (MB)"].tolist() == [25] * 4 + [100] * 4
    assert table["Metric"].tolist()[:4] == [m.label for m in METRICS]
    e2e = table[(table["Partition (MB)"] == 100) & (table["Metric"] == "End to end time")].iloc[0]
    assert e2e["End to end speedup of Subexperiment B over A (x)"] == pytest.approx(
        e2e["Subexperiment A mean"] / e2e["Subexperiment B mean"])
    other = table[table["Metric"] != "End to end time"]
    assert other["End to end speedup of Subexperiment B over A (x)"].isna().all()


def test_batch_table_takes_published_values_from_its_run_rows_and_the_rest_from_the_summary():
    table = batch_table(_summary(), {"published": (_published_batch(), 25, 1.0)}, SYSTEMS)
    assert table["Batch size"].tolist() == [32, 512]
    first = table.iloc[0]
    assert (first["Published partition (MB)"], first["Subexperiment A partition (MB)"]) == (25, 100)
    assert first["CPU time per worker (s) Published mean"] == pytest.approx(10.0)
    assert first["CPU time per worker (s) Published std"] == pytest.approx(1.0)
    assert first["Published workers"] == 41 and first["Published runs"] == 3
    assert first["Subexperiment A workers"] == 11 and first["Subexperiment A runs"] == 4
    assert first["CPU time per worker (s) Subexperiment A mean"] == pytest.approx(41.5)


def test_table6_mirrors_the_paper_rows_and_adds_the_billed_cost():
    table = table6(_cost(), SYSTEMS)
    assert table["Partition (MB)"].tolist() == [25, 100, 100]
    assert table["Requests"].tolist() == [517, 130, 10240]
    assert table["Data"].tolist() == ["12.6 GB", "12.6 GB", "1 TB"]
    first = table.iloc[0]
    assert first["Published cost estimate ($)"] == pytest.approx(0.16)
    assert pd.isna(first["Published billed cost ($)"])
    assert first["Subexperiment B billed over estimate (x)"] == pytest.approx(1.2 / 0.2)
    terabyte = table.iloc[2]
    assert terabyte["Published cost estimate ($)"] == pytest.approx(27.48)
    assert pd.isna(terabyte["Subexperiment A cost estimate ($)"])


def test_shared_limits_cover_every_system():
    low, high = shared_limits([[10.0, 20.0], [5.0, 50.0]], 0.95, 1.1)
    assert low <= 5.0 * 0.95 and high >= 50.0 * 1.1


def _metrics_dir(tmp_path):
    tmp_path.mkdir(parents=True)
    summary = _summary()
    summary.to_csv(tmp_path / "paper_metrics_summary.csv", index=False)
    _cost().to_csv(tmp_path / "paper_cost_table.csv", index=False)
    for system, frame in summary.groupby("system"):
        runs = []
        for _, s in frame.iterrows():
            for run in (1, 2, 3):
                runs.append({"system": system, "series": s.series, "partition_mb": s.partition_mb,
                             "data_gb": s.data_gb, "batch_size": s.batch_size, "run_id": run,
                             "workers": s.workers_mean, "map_s": s.map_s_mean, "duration_s": s.duration_s_mean,
                             "cpu_time_s": s.cpu_time_s_mean, "memory_mb": 1.0,
                             "throughput_sum_bps": s.throughput_map_bps_mean, "total_bits": 8e9,
                             "throughput_map_bps": s.throughput_map_bps_mean})
        results = tmp_path / "cosmic_ai_format" / system / "results"
        results.mkdir(parents=True)
        for name, table in repo_tables(pd.DataFrame(runs)).items():
            table.to_csv(results / name, index=False)
    _published_batch().to_csv(tmp_path / "cosmic_ai_format" / "published" / "results" / "batch_varying_adjusted.csv",
                              index=False)
    return tmp_path


def test_write_deliverables_writes_the_workbook_csvs_and_figures(tmp_path):
    output = tmp_path / "out"
    written = write_deliverables(str(_metrics_dir(tmp_path / "in")), str(output), SYSTEMS, {"published": (25, 1.0)})
    workbook = pd.ExcelFile(output / "tables" / "cosmic_ai_paper_tables.xlsx")
    assert workbook.sheet_names == ["Table 6 cost", "Scaling", "Headline 12.6 GB", "Batch size sweep", "Notes"]
    for name in ("table6_cost.csv", "scaling_results.csv", "headline_12_6_gb.csv", "batch_size_sweep.csv"):
        assert (output / "tables" / name).exists()
    pngs = [p for p in written if p.endswith(".png")]
    assert len(pngs) == 3 * 4 + 3 * 2 + 4
    assert all(os.path.getsize(p) > 0 for p in pngs)
    scaling = pd.read_excel(workbook, "Scaling")
    assert scaling["End to end time (s) Subexperiment B mean"].tolist() == pytest.approx(
        pd.read_csv(output / "tables" / "scaling_results.csv")["End to end time (s) Subexperiment B mean"].tolist())


@pytest.mark.skipif(not os.path.isdir(REAL_METRICS_DIR), reason="campaign results not present")
def test_scaling_table_agrees_with_the_cosmic_ai_format_mean_rows():
    summary = pd.read_csv(os.path.join(REAL_METRICS_DIR, "paper_metrics_summary.csv"))
    table = scaling_table(summary, SYSTEMS).set_index(["Partition (MB)", "Data (GB)"])
    columns = {"total_duration (s)": "End to end time (s)", "inference_duration (s)": "Map state time (s)",
               "total_cpu_time (seconds)": "CPU time per worker (s)"}
    for system, label in SYSTEMS.items():
        repo = pd.read_csv(os.path.join(REAL_METRICS_DIR, "cosmic_ai_format", system, "results",
                                        "result_stats_adjusted.csv"))
        means = repo[repo.run == "mean"].set_index(["partition (MB)", "data (GB)"])
        assert len(means) == len(table) == 28
        for repo_column, label_prefix in columns.items():
            ours = table[f"{label_prefix} {label} mean"].round(2)
            assert (ours - means[repo_column].reindex(ours.index)).abs().max() <= 0.0101
        throughput = table[f"Adjusted throughput (billion bps) {label} mean"] * 1e9
        assert ((throughput - means.throughput_bps.reindex(throughput.index)).abs() <= 0.01).all()