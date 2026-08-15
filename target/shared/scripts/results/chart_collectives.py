# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiment B collective-benchmark charts, in the quals scaling-notebook style.

Four figures matching Join_Weak_Scaling_quals2.ipynb conventions (box frame, no
gridlines, major ticks only on log axes, black-edged bars, white background, top
title, legend in a bordered box below, error bars with capsize=5):

1. latency_vs_N        - P50 latency vs N (log-log), one line per collective; O(log N).
2. throughput_vs_msgsize - throughput vs message size per collective (runtime-hue-ready).
3. speedup_vs_http     - collective-vs-HTTP latency reduction per collective (H1), bars.
4. barrier_vs_N        - barrier latency vs N (log2 N growth).

Consumes `exp_b_collectives_results.csv` (the Cylon runner, aggregated with std) and,
for the speedup figure, an HTTP-baseline CSV (rows carrying `baseline == "http"`).
"""

import argparse
import csv
import glob
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger("chart_collectives")

# Match chart_zerocopy.py / the quals notebook.
FONT_SIZE = 12
TITLE_SIZE = 14
LEGEND_SIZE = 10
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
})
BAR_EDGE = "black"
BAR_LW = 0.8

# One stable color per collective.
COLLECTIVE_COLORS = {
    "scatter": "#2ca02c",
    "scatterv": "#17becf",
    "gather": "#1f77b4",
    "allgather": "#9467bd",
    "reduce": "#ff7f0e",
    "allreduce": "#d62728",
    "broadcast": "#8c564b",
    "barrier": "#7f7f7f",
}
COLLECTIVE_ORDER = ["scatter", "scatterv", "gather", "allgather",
                    "reduce", "allreduce", "broadcast", "barrier"]


def _load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _num(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _true(row, key):
    return str(row.get(key, "")).strip().lower() == "true"


def _save(fig, output_dir, name, chart_format, chart_dpi):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.{chart_format}")
    fig.savefig(path, dpi=chart_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def _measured(rows):
    """Rows that carry a real measurement (drop unsupported / N/A cells)."""
    return [r for r in rows if not _true(r, "unsupported")]


def _rep_size(rows, collective):
    """This collective's representative message size — its largest measured size."""
    sizes = [int(r["msg_size"]) for r in rows if r["collective"] == collective]
    return max(sizes) if sizes else 0


def _collectives_present(rows):
    present = {r["collective"] for r in rows}
    return [c for c in COLLECTIVE_ORDER if c in present]


def _legend_below(ax, ncol=4):
    ax.legend(fontsize=LEGEND_SIZE, ncol=ncol, loc="lower center",
              bbox_to_anchor=(0.5, -0.32), frameon=True)


# --------------------------------------------------------------------------- 1

def chart_latency_vs_n(rows, output_dir, chart_format, chart_dpi):
    """P50 latency vs N, log-log, one line per collective (at its representative size)."""
    rows = _measured(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for coll in _collectives_present(rows):
        size = _rep_size(rows, coll)
        pts = sorted(
            ((int(r["N"]), _num(r, "latency_p50_ms"), _num(r, "latency_p50_ms_std"))
             for r in rows if r["collective"] == coll and int(r["msg_size"]) == size),
            key=lambda t: t[0],
        )
        if not pts:
            continue
        ns = [p[0] for p in pts]
        lat = [p[1] for p in pts]
        err = [p[2] for p in pts]
        ax.errorbar(ns, lat, yerr=err if any(err) else None, marker="o",
                    color=COLLECTIVE_COLORS.get(coll, "#333333"), lw=2, capsize=5,
                    label=coll)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.minorticks_off()  # major ticks only, matching the quals notebook
    ax.set_xlabel("World size N (log2)", fontsize=FONT_SIZE)
    ax.set_ylabel("P50 latency (ms, log scale)", fontsize=FONT_SIZE)
    ax.set_title("Collective latency vs N  (O(log N) on the direct channel)", fontsize=TITLE_SIZE)
    _legend_below(ax)
    return _save(fig, output_dir, "collectives_latency_vs_N", chart_format, chart_dpi)


# --------------------------------------------------------------------------- 2

def chart_throughput_vs_msgsize(rows, output_dir, chart_format, chart_dpi, runtime="pycylon"):
    """Throughput vs message size per size-swept collective, at the largest N.

    `runtime` labels the series so a future Python-vs-WASM panel can overlay a second
    runtime on the same axes (the hue slot the design calls for).
    """
    rows = _measured(rows)
    size_swept = [c for c in _collectives_present(rows) if c not in ("reduce", "allreduce", "barrier")]
    if not rows or not size_swept:
        return None
    n_max = max(int(r["N"]) for r in rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    for coll in size_swept:
        pts = sorted(
            ((int(r["msg_size"]), _num(r, "throughput_MBps"), _num(r, "throughput_MBps_std"))
             for r in rows if r["collective"] == coll and int(r["N"]) == n_max),
            key=lambda t: t[0],
        )
        if not pts:
            continue
        xs = [p[0] for p in pts]
        tp = [p[1] for p in pts]
        err = [p[2] for p in pts]
        ax.errorbar(xs, tp, yerr=err if any(err) else None, marker="o",
                    color=COLLECTIVE_COLORS.get(coll, "#333333"), lw=2, capsize=5,
                    label=f"{coll} ({runtime})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xlabel("Message size (bytes, log2)", fontsize=FONT_SIZE)
    ax.set_ylabel("Throughput (MB/s, log scale)", fontsize=FONT_SIZE)
    ax.set_title(f"Collective throughput vs message size  (N={n_max})", fontsize=TITLE_SIZE)
    _legend_below(ax)
    return _save(fig, output_dir, "collectives_throughput_vs_msgsize", chart_format, chart_dpi)


# --------------------------------------------------------------------------- 3

def chart_speedup_vs_http(cylon_rows, http_rows, output_dir, chart_format, chart_dpi):
    """Per-collective latency reduction vs the HTTP baseline (the H1 headline)."""
    cylon_rows = _measured(cylon_rows)
    if not http_rows:
        logger.info("no HTTP baseline rows — skipping collectives_speedup_vs_http")
        return None
    n_max = max(int(r["N"]) for r in cylon_rows)

    def _lat(rows, coll, size, n):
        for r in rows:
            if r["collective"] == coll and int(r["msg_size"]) == size and int(r["N"]) == n:
                return _num(r, "latency_p50_ms")
        return None

    labels, speedups, colors = [], [], []
    for coll in _collectives_present(cylon_rows):
        size = _rep_size(cylon_rows, coll)
        c = _lat(cylon_rows, coll, size, n_max)
        h = _lat(http_rows, coll, size, n_max)
        if not c or not h or c <= 0:
            continue
        labels.append(coll)
        speedups.append(h / c)  # HTTP latency / Cylon latency = reduction factor
        colors.append(COLLECTIVE_COLORS.get(coll, "#333333"))
    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, speedups, color=colors, alpha=0.9, edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_ylabel("Latency reduction vs HTTP  (x, log scale)", fontsize=FONT_SIZE)
    ax.set_title(f"Collective vs HTTP latency reduction (H1; >=10x is >=90%)  N={n_max}",
                 fontsize=TITLE_SIZE)
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, output_dir, "collectives_speedup_vs_http", chart_format, chart_dpi)


# --------------------------------------------------------------------------- 4

def chart_barrier_vs_n(rows, output_dir, chart_format, chart_dpi):
    """Barrier latency vs N — the pure synchronization overhead (~log2 N)."""
    rows = _measured(rows)
    pts = sorted(
        ((int(r["N"]), _num(r, "latency_p50_ms"), _num(r, "latency_p50_ms_std"))
         for r in rows if r["collective"] == "barrier"),
        key=lambda t: t[0],
    )
    if not pts:
        return None
    ns = [str(p[0]) for p in pts]
    lat = [p[1] for p in pts]
    err = [p[2] for p in pts]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ns, lat, yerr=err if any(err) else None, color=COLLECTIVE_COLORS["barrier"],
           alpha=0.9, edgecolor=BAR_EDGE, linewidth=BAR_LW, capsize=5)
    ax.set_ylabel("Barrier P50 latency (ms)", fontsize=FONT_SIZE)
    ax.set_xlabel("World size N", fontsize=FONT_SIZE)
    ax.set_title("Barrier synchronization latency vs N", fontsize=TITLE_SIZE)
    return _save(fig, output_dir, "collectives_barrier_vs_N", chart_format, chart_dpi)


def _load_http_baseline(results_dir):
    """HTTP-baseline rows from any baseline CSV in the dir (rows tagged baseline=http)."""
    http = []
    for path in glob.glob(os.path.join(results_dir, "*baseline*results.csv")):
        for r in _load_rows(path):
            if str(r.get("baseline", "")).lower() == "http":
                http.append(r)
    return http


def generate_collectives_charts(results_dir, output_dir=None, chart_format="svg", chart_dpi=300):
    """Generate the four Experiment B collective charts from a results dir."""
    results_csv = os.path.join(results_dir, "exp_b_collectives_results.csv")
    if not os.path.exists(results_csv):
        logger.error("No exp_b_collectives_results.csv in %s", results_dir)
        return []
    output_dir = output_dir or os.path.join(results_dir, "charts")
    rows = _load_rows(results_csv)
    http_rows = _load_http_baseline(results_dir)

    written = []
    for maker in (
        lambda: chart_latency_vs_n(rows, output_dir, chart_format, chart_dpi),
        lambda: chart_throughput_vs_msgsize(rows, output_dir, chart_format, chart_dpi),
        lambda: chart_speedup_vs_http(rows, http_rows, output_dir, chart_format, chart_dpi),
        lambda: chart_barrier_vs_n(rows, output_dir, chart_format, chart_dpi),
    ):
        path = maker()
        if path:
            written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description="Experiment B collective charts")
    parser.add_argument("results_dir", help="dir with exp_b_collectives_results.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--chart-format", default="png", choices=["svg", "png"])
    parser.add_argument("--chart-dpi", type=int, default=300)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    written = generate_collectives_charts(args.results_dir, args.output_dir,
                                          args.chart_format, args.chart_dpi)
    logger.info("wrote %d chart(s)", len(written))


if __name__ == "__main__":
    main()