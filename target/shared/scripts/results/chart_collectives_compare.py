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

"""Rivanna (UCC) vs Fargate (direct-redis) comparison charts for Experiment B.

Same quals-notebook style as chart_collectives.py (box frame, no gridlines, major
ticks only on log axes, black-edged bars, white background, top title, legend in a
bordered box below, error bars with capsize=5), extended with a platform dimension:
color still encodes the collective (matching the single-platform charts), while
linestyle/marker encodes the platform. The barrier chart has only one collective, so
platform gets the color slot there via grouped bars.

Consumes per-world-size result directories (each holding `run{n}_exp_b_collectives_
results.csv` or the already-aggregated `exp_b_collectives_results.csv`) for two
platforms and combines them into one comparison dataset.
"""

import argparse
import csv
import glob
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .chart_collectives import (
    BAR_EDGE,
    BAR_LW,
    COLLECTIVE_COLORS,
    FONT_SIZE,
    LEGEND_SIZE,
    TITLE_SIZE,
    _collectives_present,
    _measured,
    _num,
    _rep_size,
    _save,
)
from .notebook_generator import generate_collectives_compare_notebook
from .pipeline import aggregate_collectives_runs

logger = logging.getLogger("chart_collectives_compare")

PLATFORM_STYLE = {
    "rivanna": {"color": "#1f77b4", "linestyle": "-", "marker": "o", "label": "Rivanna (UCC)"},
    "fargate": {"color": "#d62728", "linestyle": "--", "marker": "s", "label": "Fargate (direct-redis)"},
}
PLATFORM_ORDER = ["rivanna", "fargate"]


def _legend_below(ax, ncol=4, y=-0.38):
    ax.legend(fontsize=LEGEND_SIZE - 1, ncol=ncol, loc="lower center",
              bbox_to_anchor=(0.5, y), frameon=True)


def load_platform_rows(platform, world_size_dirs):
    """Aggregate each ws dir's per-run CSVs, tag every row with `platform`, combine."""
    rows = []
    for ws_dir in sorted(world_size_dirs):
        aggregate_collectives_runs(ws_dir)
        results_csv = os.path.join(ws_dir, "exp_b_collectives_results.csv")
        if not os.path.exists(results_csv):
            logger.warning("no results CSV in %s — skipping", ws_dir)
            continue
        with open(results_csv) as f:
            for r in csv.DictReader(f):
                r["platform"] = platform
                rows.append(r)
    return rows


def write_combined_csv(rows, output_dir):
    if not rows:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "exp_b_collectives_rivanna_vs_fargate.csv")
    fieldnames = ["platform"] + [k for k in rows[0].keys() if k != "platform"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info("wrote %s (%d rows)", path, len(rows))
    return path


def chart_latency_vs_n_compare(rows, output_dir, chart_format, chart_dpi):
    """P50 latency vs N, log-log, one line per (collective, platform)."""
    rows = _measured(rows)
    fig, ax = plt.subplots(figsize=(11, 7))
    plotted = False
    for coll in _collectives_present(rows):
        color = COLLECTIVE_COLORS.get(coll, "#333333")
        for plat in PLATFORM_ORDER:
            plat_rows = [r for r in rows if r["platform"] == plat]
            size = _rep_size(plat_rows, coll)
            pts = sorted(
                ((int(r["N"]), _num(r, "latency_p50_ms"), _num(r, "latency_p50_ms_std"))
                 for r in plat_rows if r["collective"] == coll and int(r["msg_size"]) == size),
                key=lambda t: t[0],
            )
            if not pts:
                continue
            ns = [p[0] for p in pts]
            lat = [p[1] for p in pts]
            err = [p[2] for p in pts]
            style = PLATFORM_STYLE[plat]
            ax.errorbar(ns, lat, yerr=err if any(err) else None,
                        marker=style["marker"], linestyle=style["linestyle"],
                        color=color, lw=2, capsize=5,
                        label=f"{coll} — {style['label']}")
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xlabel("World size N (log2)", fontsize=FONT_SIZE)
    ax.set_ylabel("P50 latency (ms, log scale)", fontsize=FONT_SIZE)
    ax.set_title("Collective latency vs N — Rivanna (UCC) vs Fargate (direct-redis)",
                 fontsize=TITLE_SIZE)
    _legend_below(ax, ncol=4)
    return _save(fig, output_dir, "collectives_latency_vs_N_rivanna_vs_fargate", chart_format, chart_dpi)


def chart_throughput_vs_msgsize_compare(rows, output_dir, chart_format, chart_dpi):
    """Throughput vs message size per size-swept collective, at the largest common N."""
    rows = _measured(rows)
    size_swept = [c for c in _collectives_present(rows) if c not in ("reduce", "allreduce", "barrier")]
    if not rows or not size_swept:
        return None
    n_common = min(
        max(int(r["N"]) for r in rows if r["platform"] == plat)
        for plat in PLATFORM_ORDER if any(r["platform"] == plat for r in rows)
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    plotted = False
    for coll in size_swept:
        color = COLLECTIVE_COLORS.get(coll, "#333333")
        for plat in PLATFORM_ORDER:
            pts = sorted(
                ((int(r["msg_size"]), _num(r, "throughput_MBps"), _num(r, "throughput_MBps_std"))
                 for r in rows if r["collective"] == coll and r["platform"] == plat
                 and int(r["N"]) == n_common),
                key=lambda t: t[0],
            )
            if not pts:
                continue
            xs = [p[0] for p in pts]
            tp = [p[1] for p in pts]
            err = [p[2] for p in pts]
            style = PLATFORM_STYLE[plat]
            ax.errorbar(xs, tp, yerr=err if any(err) else None,
                        marker=style["marker"], linestyle=style["linestyle"],
                        color=color, lw=2, capsize=5,
                        label=f"{coll} — {style['label']}")
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xlabel("Message size (bytes, log2)", fontsize=FONT_SIZE)
    ax.set_ylabel("Throughput (MB/s, log scale)", fontsize=FONT_SIZE)
    ax.set_title(f"Collective throughput vs message size — Rivanna vs Fargate  (N={n_common})",
                 fontsize=TITLE_SIZE)
    _legend_below(ax, ncol=4)
    return _save(fig, output_dir, "collectives_throughput_vs_msgsize_rivanna_vs_fargate",
                 chart_format, chart_dpi)


def chart_barrier_vs_n_compare(rows, output_dir, chart_format, chart_dpi):
    """Barrier P50 latency vs N, grouped bars (Rivanna vs Fargate) per N."""
    rows = _measured(rows)
    ns = sorted({int(r["N"]) for r in rows if r["collective"] == "barrier"})
    if not ns:
        return None

    x = range(len(ns))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, plat in enumerate(PLATFORM_ORDER):
        style = PLATFORM_STYLE[plat]
        lat, err = [], []
        for n in ns:
            cell = [r for r in rows if r["collective"] == "barrier" and r["platform"] == plat
                    and int(r["N"]) == n]
            lat.append(_num(cell[0], "latency_p50_ms") if cell else 0.0)
            err.append(_num(cell[0], "latency_p50_ms_std") if cell else 0.0)
        offset = (i - 0.5) * width
        xs = [xi + offset for xi in x]
        ax.bar(xs, lat, width=width, yerr=err if any(err) else None,
               color=style["color"], alpha=0.9, edgecolor=BAR_EDGE, linewidth=BAR_LW,
               capsize=5, label=style["label"])
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("World size N", fontsize=FONT_SIZE)
    ax.set_ylabel("Barrier P50 latency (ms)", fontsize=FONT_SIZE)
    ax.set_title("Barrier synchronization latency vs N — Rivanna vs Fargate", fontsize=TITLE_SIZE)
    _legend_below(ax, ncol=2, y=-0.28)
    return _save(fig, output_dir, "collectives_barrier_vs_N_rivanna_vs_fargate", chart_format, chart_dpi)


def generate_compare_charts(rows, output_dir, chart_format="png", chart_dpi=300):
    written = []
    for maker in (
        lambda: chart_latency_vs_n_compare(rows, output_dir, chart_format, chart_dpi),
        lambda: chart_throughput_vs_msgsize_compare(rows, output_dir, chart_format, chart_dpi),
        lambda: chart_barrier_vs_n_compare(rows, output_dir, chart_format, chart_dpi),
    ):
        path = maker()
        if path:
            written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description="Rivanna vs Fargate Experiment B comparison charts")
    parser.add_argument("--rivanna-dir", required=True, help="dir containing rivanna ws*/ subdirs")
    parser.add_argument("--fargate-dir", required=True, help="dir containing fargate ws*/ subdirs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chart-format", default="png", choices=["svg", "png"])
    parser.add_argument("--chart-dpi", type=int, default=300)
    parser.add_argument("--notebook-name", default="exp_b_collectives_rivanna_vs_fargate")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rivanna_ws = glob.glob(os.path.join(args.rivanna_dir, "ws*"))
    fargate_ws = glob.glob(os.path.join(args.fargate_dir, "ws*"))
    rows = load_platform_rows("rivanna", rivanna_ws) + load_platform_rows("fargate", fargate_ws)

    combined_csv = write_combined_csv(rows, args.output_dir)
    charts_dir = os.path.join(args.output_dir, "charts")
    written = generate_compare_charts(rows, charts_dir, args.chart_format, args.chart_dpi)
    logger.info("wrote %d chart(s)", len(written))

    if combined_csv:
        notebook_path = os.path.join(args.output_dir, f"{args.notebook_name}.ipynb")
        generate_collectives_compare_notebook(
            results_csv=os.path.basename(combined_csv),
            output_path=notebook_path,
            output_chart_dir="charts",
        )


if __name__ == "__main__":
    main()