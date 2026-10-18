import pandas as pd

from results.chart_cosmic_campaign import compare_with_published, render_all, published_seconds

SUMMARY_COLUMNS = ["experiment", "arm", "series", "partition_mb", "data_gb", "batch_size", "workers", "phase", "n",
                   "complete", "duration_s_mean", "duration_s_std", "inference_s_max_mean", "collect_s_mean"]


def _summary():
    rows = [("exp1", "A", "scaling", p, d, 512, w, phase, 3, True, 30.0 + w / 10, 1.0, 5.0, 0.05 * w)
            for p, d, w in ((25, 1.0, 41), (25, 2.0, 82), (100, 1.0, 11)) for phase in ("cold_start", "measured")]
    rows += [("exp1", "A", "batch", 100, 1.0, b, 11, "measured", 4, True, 100.0 - b / 8, 2.0, 15.0, 1.0)
             for b in (32, 512)]
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _published():
    return pd.DataFrame({"partition(MB)": [25, 25, 25, 100], "data(GB)": [1.0, 1.0, 2.0, 1.0],
                         "run": [1, 2, 1, 1], "num_worlds": [41.0, 41.0, 82.0, 11.0],
                         "duration": ["00:26.000", "00:28.000", "01:02.500", "00:32.640"]})


def test_published_durations_in_minutes_and_seconds_become_seconds():
    assert list(published_seconds(pd.Series(["00:26.642", "01:02.500"]))) == [26.642, 62.5]


def test_measured_scaling_rows_are_joined_to_the_published_mean_and_ratio():
    joined = compare_with_published(_summary(), _published())
    assert len(joined) == 3
    row = joined[(joined.partition_mb == 25) & (joined.data_gb == 1.0)].iloc[0]
    assert (row.published_mean_s, row.published_n) == (27.0, 2)
    assert round(row.ratio, 4) == round((30.0 + 4.1) / 27.0, 4)


def test_every_chart_is_written(tmp_path):
    paths = render_all(_summary(), _published(), str(tmp_path), "png", 72)
    assert set(paths) == {"exp1_vs_published.png", "exp1_time_breakdown.png", "exp1_collect_scaling.png",
                          "exp1_batch_sweep.png"}
    assert all((tmp_path / p).exists() for p in paths)


def test_a_line_chart_draws_every_series_with_its_error_bars_on_one_plot(tmp_path):
    from results.chart_cosmic_campaign import line_chart
    import matplotlib.pyplot as plt
    series = [{"label": "Arm A", "color": "orange", "marker": "o", "linestyle": "-", "x": [41, 82], "y": [20.0, 23.0],
               "yerr": [1.0, 2.0]},
              {"label": "Arm B", "color": "green", "marker": "s", "linestyle": "--", "x": [41, 82], "y": [21.0, 20.0],
               "yerr": [0.5, None]}]
    captured = {}
    original = plt.close
    plt.close = lambda fig=None: captured.setdefault("fig", fig)
    try:
        name = line_chart(series, "Workers", "Execution time (s)", "t", str(tmp_path), "lines", "png", 60)
    finally:
        plt.close = original
    ax = captured["fig"].axes[0]
    assert len(captured["fig"].axes) == 1 and (tmp_path / name).exists()
    assert ax.get_xscale() == "log"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Arm A", "Arm B"]
    caps = [c for c in ax.containers if hasattr(c, "has_yerr") and c.has_yerr]
    assert len(caps) == 2

def test_the_paper_version_published_file_gives_seconds_per_run_without_the_mean_rows():
    from results.chart_cosmic_campaign import normalize_published
    paper = pd.DataFrame({"partition (MB)": [25, 25, 25], "data (GB)": [1.0, 1.0, 1.0], "run": ["1", "2", "mean"],
                          "total_duration (s)": [20.0, 22.0, 21.0], "inference_duration (s)": [15.0, 16.0, 15.5]})
    rows = normalize_published(paper)
    assert list(rows.columns) == ["partition_mb", "data_gb", "run", "seconds"]
    assert rows.seconds.tolist() == [20.0, 22.0]


def test_the_console_published_file_gives_seconds_per_run():
    from results.chart_cosmic_campaign import normalize_published
    rows = normalize_published(_published())
    assert rows.seconds.tolist() == [26.0, 28.0, 62.5, 32.64]
    assert rows.partition_mb.tolist() == [25, 25, 25, 100]


def test_partition_panels_draw_one_panel_per_partition_with_every_system(tmp_path):
    from results.chart_cosmic_campaign import partition_panels
    import matplotlib.pyplot as plt
    rows = pd.DataFrame({"partition_mb": [25, 25, 100], "data_gb": [1.0, 2.0, 1.0], "workers": [41, 82, 11],
                         "a": [20.0, 22.0, 30.0], "a_std": [1.0, 1.0, 2.0], "b": [19.0, 20.0, 31.0],
                         "b_std": [0.5, None, 1.0]})
    lines = [("A", "a", "a_std"), ("B", "b", "b_std")]
    captured = {}
    original = plt.close
    plt.close = lambda fig=None: captured.setdefault("fig", fig)
    try:
        name = partition_panels(rows, lines, "Execution time (s)", "t", str(tmp_path), "panels", "png", 60)
    finally:
        plt.close = original
    axes = [ax for ax in captured["fig"].axes if ax.has_data()]
    assert (tmp_path / name).exists() and len(axes) == 2
    assert [ax.get_title() for ax in axes] == ["25 MB partitions", "100 MB partitions"]
    assert [t.get_text() for t in axes[0].get_xticklabels()] == ["1\n(41)", "2\n(82)"]
    assert all(len(ax.get_lines()) >= 2 for ax in axes)


def test_partition_panels_share_one_y_scale(tmp_path):
    from results.chart_cosmic_campaign import partition_panels
    import matplotlib.pyplot as plt
    rows = pd.DataFrame({"partition_mb": [25, 100], "data_gb": [1.0, 1.0], "workers": [41, 11],
                         "a": [20.0, 60.0], "a_std": [None, 1.0]})
    captured = {}
    original = plt.close
    plt.close = lambda fig=None: captured.setdefault("fig", fig)
    try:
        partition_panels(rows, [("A", "a", "a_std")], "s", "t", str(tmp_path), "shared", "png", 60)
    finally:
        plt.close = original
    first, second = [ax for ax in captured["fig"].axes if ax.has_data()]
    assert first.get_ylim() == second.get_ylim()
