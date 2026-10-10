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
    assert len(paths) == 4 and all((tmp_path / p).exists() for p in paths)