"""Experiment B Cylon against Ray collective charts.

Line charts in the quals style (box frame, no gridlines, major ticks only on
log axes, error bars with capsize 5, legend in a bordered box below), one line
per series: Cylon over the direct Redis communicator, Ray Gloo timed from the
driver, Ray Gloo timed inside the actor, and the Ray object store.

Reads `<root>/<channel>/<payload>/ws<N>/`, aggregating each directory's per-run
CSVs first.

Usage:
    python -m results.chart_collectives_ray --root <dir> --output <dir>/charts \
        [--format png] [--dpi 300] [--world-sizes 8 64]
"""

import argparse
import csv
import glob
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .chart_collectives import FONT_SIZE, LEGEND_SIZE, TITLE_SIZE, _measured, _num, _save
from .pipeline import aggregate_collectives_runs

logger = logging.getLogger("chart_collectives_ray")

SERIES = (
    {"channel": "fmi-direct-redis", "metric": "latency_p50_ms",
     "label": "Cylon (direct Redis communicator)", "color": "#2ca02c", "linestyle": "-",
     "marker": "o"},
    {"channel": "ray-gloo", "metric": "latency_p50_ms", "label": "Ray Gloo, from the driver",
     "color": "#9467bd", "linestyle": "-", "marker": "P"},
    {"channel": "ray-gloo", "metric": "latency_inner_p50_ms", "label": "Ray Gloo, inside the actor",
     "color": "#9467bd", "linestyle": "--", "marker": "x"},
    {"channel": "ray-objectstore", "metric": "latency_p50_ms", "label": "Ray object store",
     "color": "#ff7f0e", "linestyle": "-", "marker": "^"},
)
COLLECTIVES = ("allgather", "broadcast", "gather", "allreduce", "reduce", "barrier")
SIZE_SWEPT = ("allgather", "broadcast", "gather")
DEFAULT_SIZE_WORLD_SIZES = (8, 64)


def load_rows(root):
    rows = []
    for ws_dir in sorted(glob.glob(os.path.join(root, "*", "*", "ws*"))):
        aggregate_collectives_runs(ws_dir)
        path = os.path.join(ws_dir, "exp_b_collectives_results.csv")
        if not os.path.exists(path):
            logger.warning("no results CSV in %s, skipping", ws_dir)
            continue
        with open(path) as f:
            rows.extend(csv.DictReader(f))
    return rows


def _points(rows, series, x_key):
    points = []
    for row in _measured(rows):
        if row["channel"] != series["channel"] or row.get(series["metric"], "") in ("", None):
            continue
        points.append((int(row[x_key]), _num(row, series["metric"]),
                       _num(row, f"{series['metric']}_std")))
    return sorted(points)


def _line_chart(rows, x_key, xlabel, title, name, output_dir, chart_format, chart_dpi):
    drawn = [(series, _points(rows, series, x_key)) for series in SERIES]
    drawn = [(series, points) for series, points in drawn if points]
    if not drawn:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = sorted({x for _, points in drawn for x, _, _ in points})
    for series, points in drawn:
        ax.errorbar([p[0] for p in points], [p[1] for p in points],
                    yerr=[p[2] for p in points], color=series["color"], ecolor=series["color"],
                    linestyle=series["linestyle"], marker=series["marker"], lw=2, capsize=5,
                    label=series["label"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel("P50 latency (ms)", fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.32),
              frameon=True)
    return _save(fig, output_dir, name, chart_format, chart_dpi)


def chart_latency_vs_n(rows, collective, msg_size, payload, output_dir, chart_format, chart_dpi,
                       name_suffix=""):
    """P50 latency against N for one collective at one message size."""
    cell = [r for r in rows if r["collective"] == collective and r.get("payload") == payload
            and int(r["msg_size"]) == msg_size]
    return _line_chart(cell, "N", "World size N (Fargate tasks)",
                       f"Experiment B {collective} latency vs N ({payload}, {msg_size} B)",
                       f"exp_b_ray_{payload}_{collective}_latency_vs_n{name_suffix}", output_dir,
                       chart_format, chart_dpi)


def chart_latency_vs_msg_size(rows, collective, world_size, payload, output_dir, chart_format,
                              chart_dpi):
    """P50 latency against message size for one collective at one N."""
    cell = [r for r in rows if r["collective"] == collective and r.get("payload") == payload
            and int(r["N"]) == world_size]
    return _line_chart(cell, "msg_size", "Message size (bytes)",
                       f"Experiment B {collective} latency vs message size ({payload}, N={world_size})",
                       f"exp_b_ray_{payload}_{collective}_latency_vs_msg_size_n{world_size}",
                       output_dir, chart_format, chart_dpi)


def _representative_size(rows, collective, payload):
    sizes = [int(r["msg_size"]) for r in _measured(rows)
             if r["collective"] == collective and r.get("payload") == payload]
    return max(sizes) if sizes else None


def generate(root, output_dir, chart_format="png", chart_dpi=300,
             size_world_sizes=DEFAULT_SIZE_WORLD_SIZES, vs_n_sizes=()):
    """Charts for every payload and collective. `vs_n_sizes` adds a latency
    against N chart per message size for the size-swept collectives, named
    with a `_<size>B` suffix."""
    rows = load_rows(root)
    paths = []
    for payload in sorted({r.get("payload", "float32") for r in rows}):
        for collective in COLLECTIVES:
            size = _representative_size(rows, collective, payload)
            if size is None:
                continue
            paths.append(chart_latency_vs_n(rows, collective, size, payload, output_dir,
                                            chart_format, chart_dpi))
            if collective in SIZE_SWEPT:
                for world_size in size_world_sizes:
                    paths.append(chart_latency_vs_msg_size(rows, collective, world_size, payload,
                                                           output_dir, chart_format, chart_dpi))
                for msg_size in vs_n_sizes:
                    paths.append(chart_latency_vs_n(rows, collective, msg_size, payload, output_dir,
                                                    chart_format, chart_dpi,
                                                    name_suffix=f"_{msg_size}B"))
    return [p for p in paths if p]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", default="png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--world-sizes", type=int, nargs="+", default=list(DEFAULT_SIZE_WORLD_SIZES))
    parser.add_argument("--vs-n-sizes", type=int, nargs="+", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    generate(args.root, args.output, args.format, args.dpi, args.world_sizes, args.vs_n_sizes)


if __name__ == "__main__":
    main()