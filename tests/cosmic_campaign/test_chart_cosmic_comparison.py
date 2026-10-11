import json

import pandas as pd
import pytest

from results.chart_cosmic_comparison import (
    compare_arms,
    compare_experiments,
    planning_table,
    render_all,
)

SUMMARY_COLUMNS = ["experiment", "arm", "series", "partition_mb", "data_gb", "batch_size", "workers", "phase", "n",
                   "complete", "duration_s_mean", "duration_s_std", "inference_s_max_mean", "collect_s_mean"]


def _row(experiment, arm, series, partition, data, workers, mean, phase="measured", batch=512):
    return (experiment, arm, series, partition, data, batch, workers, phase, 3, True, mean, 1.0, 5.0, 0.1)


def _summary():
    rows = [
        _row("exp1", "A", "baseline", 100, 0.0977, 1, 20.0),
        _row("exp1", "B", "baseline", 100, 0.0977, 1, 24.0),
        _row("exp1", "A", "scaling", 100, 1.0, 11, 32.0),
        _row("exp2", "A", "scaling", 100, 1.0, 11, 33.0),
        _row("exp2", "B", "scaling", 100, 1.0, 11, 36.0),
        _row("exp2", "B", "scaling", 100, 1.0, 11, 90.0, phase="cold_start"),
        _row("exp1", "A", "scaling", 25, 12.6, 517, 53.0),
        _row("exp2", "A", "scaling", 25, 12.6, 517, 54.0),
        _row("exp2", "B", "scaling", 25, 12.6, 517, 45.0),
        _row("exp2", "A", "scaling", 25, 1.0, 41, 28.0),
        _row("exp1", "A", "batch", 100, 1.0, 11, 40.0, batch=32),
        _row("exp2", "A", "batch", 100, 1.0, 11, 41.0, batch=32),
        _row("exp2", "B", "batch", 100, 1.0, 11, 43.0, batch=32),
    ]
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _published():
    return pd.DataFrame({"partition(MB)": [100, 25], "data(GB)": [1.0, 12.6], "run": [1, 1],
                         "num_worlds": [11.0, 517.0], "duration": ["00:32.640", "00:59.789"]})


def test_arms_are_compared_per_configuration_from_the_chosen_experiment():
    joined = compare_arms(_summary(), experiment="exp2")
    row = joined[(joined.series == "scaling") & (joined.workers == 517)].iloc[0]
    assert (row.a_mean_s, row.b_mean_s) == (54.0, 45.0)
    assert row.speedup == pytest.approx(54.0 / 45.0)
    assert (row.a_experiment, row.b_experiment) == ("exp2", "exp2")


def test_baselines_missing_from_the_chosen_experiment_come_from_another_one():
    joined = compare_arms(_summary(), experiment="exp2")
    row = joined[joined.series == "baseline"].iloc[0]
    assert (row.workers, row.a_mean_s, row.b_mean_s) == (1, 20.0, 24.0)
    assert (row.a_experiment, row.b_experiment) == ("exp1", "exp1")


def test_arm_comparison_uses_measured_runs_and_only_configurations_both_arms_ran():
    joined = compare_arms(_summary(), experiment="exp2")
    assert 41 not in set(joined.workers)
    assert joined[(joined.series == "scaling") & (joined.workers == 11)].iloc[0].b_mean_s == 36.0
    assert list(joined.workers) == sorted(joined.workers)


def test_experiments_are_compared_per_configuration_for_one_arm():
    joined = compare_experiments(_summary(), arm="A")
    row = joined[(joined.series == "scaling") & (joined.workers == 517)].iloc[0]
    assert (row.exp1_mean_s, row.exp2_mean_s) == (53.0, 54.0)
    assert row.difference_s == pytest.approx(1.0)
    assert set(joined.series) == {"scaling", "batch"}


def _write_metrics(root, arm, mode, **values):
    path = root / f"plans_{arm}_{mode}"
    path.mkdir()
    metrics = {"requests": 100, "llm_calls": 0, "llm_tool_calls": 50, "cache_hits_exact": 0,
               "cache_hits_structural": 50, "wrong_plans": 2, "wrong_reuse": 1, "planning_ms_mean": 1000.0,
               "planning_wall_ms": 60000.0, "input_tokens": 1000, "output_tokens": 100}
    metrics.update(values)
    (path / "plan_metrics.json").write_text(json.dumps(metrics))


def test_planning_table_reads_every_arm_and_cache_mode_and_derives_rates(tmp_path):
    _write_metrics(tmp_path, "B", "structural")
    _write_metrics(tmp_path, "B", "off", llm_tool_calls=100, cache_hits_structural=0, wrong_plans=3, wrong_reuse=0)
    table = planning_table(str(tmp_path))
    assert set(zip(table.arm, table.cache_mode)) == {("B", "structural"), ("B", "off")}
    structural = table[table.cache_mode == "structural"].iloc[0]
    assert structural.llm_calls_total == 50
    assert structural.cache_hit_rate == pytest.approx(0.5)
    assert structural.plan_correct_rate == pytest.approx(0.98)
    assert structural.wrong_reuse_rate == pytest.approx(1 / 50)


def test_every_chart_and_table_is_written(tmp_path):
    plans = tmp_path / "plans"
    plans.mkdir()
    for mode in ("structural", "exact", "off"):
        _write_metrics(plans, "A", mode)
        _write_metrics(plans, "B", mode)
    paths = render_all(_summary(), _published(), str(plans), str(tmp_path / "out"), "png", 72)
    assert len(paths) >= 4 and all((tmp_path / "out" / p).exists() for p in paths)

def test_rendering_works_before_both_arms_share_a_scaling_configuration(tmp_path):
    summary = _summary()
    summary = summary[~((summary.arm == "B") & (summary.series == "scaling"))]
    paths = render_all(summary, _published(), str(tmp_path), str(tmp_path / "out"), "png", 72)
    assert all((tmp_path / "out" / p).exists() for p in paths)


def _pair_latency():
    rows = [("accept_timeout_1s", t, s, 1000.0 if t % 2 else 2.0) for t in range(4) for s in (0, 1)]
    rows += [("accept_timeout_50ms", t, s, 50.0) for t in range(4) for s in (0, 1)]
    return pd.DataFrame(rows, columns=["variant", "trial", "side", "pair_ms"])


def test_pair_latency_summary_excludes_failed_pairings_and_reports_the_speedup():
    from results.chart_cosmic_comparison import pair_latency_summary
    frame = pd.concat([_pair_latency(), pd.DataFrame([("accept_timeout_50ms", 9, 0, -1.0)],
                                                     columns=["variant", "trial", "side", "pair_ms"])])
    summary = pair_latency_summary(frame).set_index("variant")
    assert summary.loc["accept_timeout_50ms", "n"] == 8
    assert summary.loc["accept_timeout_1s", "mean_ms"] == pytest.approx(501.0)
    assert summary.loc["accept_timeout_50ms", "speedup_vs_1s"] == pytest.approx(501.0 / 50.0)


def test_pair_latency_chart_is_written(tmp_path):
    from results.chart_cosmic_comparison import chart_pair_latency
    name = chart_pair_latency(_pair_latency(), str(tmp_path), "png", 72)
    assert (tmp_path / name).exists()
