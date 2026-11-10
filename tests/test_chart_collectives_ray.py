import csv
import os
import sys

import pytest

pytest.importorskip("matplotlib")
from matplotlib.container import ErrorbarContainer

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from results import chart_collectives_ray as chart  # noqa: E402


def _row(channel, n, p50, inner="", unsupported=False, collective="allgather", msg_size=4096,
         payload="float32"):
    return {"channel": channel, "payload": payload, "collective": collective,
            "msg_size": str(msg_size), "N": str(n), "latency_p50_ms": str(p50),
            "latency_p50_ms_std": "0.1", "latency_inner_p50_ms": str(inner),
            "latency_inner_p50_ms_std": "0.05" if inner != "" else "0.0",
            "unsupported": str(unsupported)}


def _all_series(ns=(1, 2, 4)):
    rows = []
    for n in ns:
        rows += [_row("fmi-direct-redis", n, 1.0 * n), _row("ray-gloo", n, 3.0 * n, inner=1.5 * n),
                 _row("ray-objectstore", n, 4.0 * n)]
    return rows


@pytest.fixture
def captured(monkeypatch):
    figures = []

    def keep(fig, output_dir, name, chart_format, chart_dpi):
        figures.append((name, fig))
        return name

    monkeypatch.setattr(chart, "_save", keep)
    return figures


def test_one_line_per_series_with_error_bars_and_log_axes(captured, tmp_path):
    chart.chart_latency_vs_n(_all_series(), "allgather", 4096, "float32", str(tmp_path), "png", 72)

    ax = captured[0][1].axes[0]
    containers = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert len(containers) == 4
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    assert not ax.texts
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "Cylon (direct Redis communicator)", "Ray Gloo, from the driver",
        "Ray Gloo, inside the actor", "Ray object store"]


def test_unsupported_cells_are_left_out_not_drawn_at_zero(captured, tmp_path):
    rows = [_row("fmi-direct-redis", n, 1.0, collective="gather", msg_size=64) for n in (2, 4)]
    rows += [_row("ray-gloo", n, 0.0, unsupported=True, collective="gather", msg_size=64)
             for n in (2, 4)]

    chart.chart_latency_vs_n(rows, "gather", 64, "float32", str(tmp_path), "png", 72)

    ax = captured[0][1].axes[0]
    assert len([c for c in ax.containers if isinstance(c, ErrorbarContainer)]) == 1


def test_a_collective_with_no_ray_rows_still_draws_cylon_alone(captured, tmp_path):
    rows = [_row("fmi-direct-redis", n, 1.0, collective="scatter") for n in (1, 2)]

    assert chart.chart_latency_vs_n(rows, "scatter", 4096, "float32", str(tmp_path), "png", 72)
    labels = [t.get_text() for t in captured[0][1].axes[0].get_legend().get_texts()]
    assert labels == ["Cylon (direct Redis communicator)"]


def test_no_measured_rows_draws_nothing(captured, tmp_path):
    assert chart.chart_latency_vs_n([], "allgather", 4096, "float32", str(tmp_path), "png", 72) is None
    assert captured == []


def test_latency_vs_msg_size_plots_each_series_across_sizes(captured, tmp_path):
    rows = []
    for size in (8, 512, 4096):
        rows += [_row("fmi-direct-redis", 8, 1.0, msg_size=size),
                 _row("ray-gloo", 8, 2.0, inner=1.0, msg_size=size)]

    chart.chart_latency_vs_msg_size(rows, "allgather", 8, "float32", str(tmp_path), "png", 72)

    ax = captured[0][1].axes[0]
    containers = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert len(containers) == 3
    assert list(containers[0].lines[0].get_xdata()) == [8, 512, 4096]


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_load_rows_walks_channel_payload_and_world_size_dirs(tmp_path):
    for channel in ("fmi-direct-redis", "ray-gloo"):
        for n in (1, 2):
            _write(os.path.join(tmp_path, channel, "kv_pairs", f"ws{n}",
                                "exp_b_collectives_results.csv"),
                   [_row(channel, n, 1.0, payload="kv_pairs")])

    rows = chart.load_rows(str(tmp_path))

    assert sorted((r["channel"], r["N"]) for r in rows) == [
        ("fmi-direct-redis", "1"), ("fmi-direct-redis", "2"), ("ray-gloo", "1"), ("ray-gloo", "2")]


def test_generate_writes_charts_per_payload(tmp_path):
    for channel in ("fmi-direct-redis", "ray-gloo"):
        for n in (8, 64):
            _write(os.path.join(tmp_path, "data", channel, "float32", f"ws{n}",
                                "exp_b_collectives_results.csv"),
                   [_row(channel, n, 1.0, inner=0.5 if channel == "ray-gloo" else "")])

    paths = chart.generate(os.path.join(tmp_path, "data"), os.path.join(tmp_path, "charts"))

    names = sorted(os.path.basename(p) for p in paths)
    assert "exp_b_ray_float32_allgather_latency_vs_n.png" in names
    assert "exp_b_ray_float32_allgather_latency_vs_msg_size_n8.png" in names
    assert all(os.path.exists(p) for p in paths)

def test_generate_draws_latency_vs_n_at_each_requested_message_size(tmp_path):
    for channel in ("fmi-direct-redis", "ray-gloo"):
        for n in (8, 64):
            _write(os.path.join(tmp_path, "data", channel, "kv_pairs", f"ws{n}",
                                "exp_b_collectives_results.csv"),
                   [_row(channel, n, 1.0, payload="kv_pairs", msg_size=size)
                    for size in (4096, 1048576)])

    paths = chart.generate(os.path.join(tmp_path, "data"), os.path.join(tmp_path, "charts"),
                           vs_n_sizes=(4096, 1048576))

    names = {os.path.basename(p) for p in paths}
    assert {"exp_b_ray_kv_pairs_allgather_latency_vs_n_4096B.png",
            "exp_b_ray_kv_pairs_allgather_latency_vs_n_1048576B.png"} <= names
