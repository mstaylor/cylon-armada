"""Experiment E scaling charts from exp_e_scaling.write_summary's CSV.

Per scaling mode: the end-to-end comparable time against N, one line per arm;
each arm's framework setup (the last rank to arrive), which is kept out of the
comparable time and so has to be shown on its own; and the time to result,
the two added. Mean with sample-std error bars across the measured runs, in
the quals notebook style.

Usage:
    python -m results.chart_exp_e_scaling --summary <dir>/exp_e_scaling_summary.csv \
        --output <dir>/charts [--format png] [--dpi 300]
"""

import argparse
import csv
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

FONT_SIZE = 12
TITLE_SIZE = 14
TICK_SIZE = 10
LEGEND_SIZE = 10

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
})
BAR_EDGE = "black"
BAR_LW = 0.8

ARM_ORDER = ("armada", "armada-overlap", "armada-async", "ray-collective",
             "ray-collective-overlap", "ray-cylon", "ray-native", "langchain", "isolated")
ARM_STYLE = {
    "armada": {"label": "Armada (Cylon collectives)", "color": "#2ca02c", "marker": "o"},
    "armada-overlap": {"label": "Armada, overlapped epochs", "color": "#2ca02c", "marker": "o",
                       "linestyle": "--", "hatch": "//"},
    "armada-async": {"label": "Armada, publish on completion", "color": "#2ca02c", "marker": "o",
                     "linestyle": ":", "hatch": "xx"},
    "ray-collective": {"label": "Ray collectives (Gloo)", "color": "#9467bd", "marker": "P"},
    "ray-collective-overlap": {"label": "Ray collectives (Gloo), overlapped epochs",
                               "color": "#9467bd", "marker": "P", "linestyle": "--",
                               "hatch": "//"},
    "ray-cylon": {"label": "Ray + Cylon collectives", "color": "#1f77b4", "marker": "s"},
    "ray-native": {"label": "Ray native (object store)", "color": "#ff7f0e", "marker": "^"},
    "langchain": {"label": "LangChain (Redis store)", "color": "#d62728", "marker": "D"},
    "isolated": {"label": "Isolated (no sharing)", "color": "#7f7f7f", "marker": "v"},
}
DEFAULT_STYLE = {"color": "#333333", "marker": "o"}
COMPARABLE_LABEL = {
    "ray-native": "run + teardown barrier",
    "armada-async": "run + teardown barrier",
}


def _float(value):
    return float(value) if value not in (None, "") else None


def load_summary(summary_csv):
    with open(summary_csv) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["world_size"] = int(row["world_size"])
        for key in list(row):
            if key.endswith("_mean") or key.endswith("_std"):
                row[key] = _float(row[key])
    return rows


def _arms(rows):
    present = {row["arm"] for row in rows}
    return [arm for arm in ARM_ORDER if arm in present] + sorted(present - set(ARM_ORDER))


def _series(rows, arm, metric):
    points = sorted((row["world_size"], row[f"{metric}_mean"], row[f"{metric}_std"])
                    for row in rows if row["arm"] == arm and row[f"{metric}_mean"] is not None)
    return ([p[0] for p in points], [p[1] for p in points], [p[2] or 0.0 for p in points])


def _save(fig, output_dir, name, chart_format, chart_dpi):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.{chart_format}")
    fig.savefig(path, dpi=chart_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def _legend_below(ax):
    ax.legend(fontsize=LEGEND_SIZE, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.32), frameon=True)


def _log2_world_size_axis(ax, world_sizes):
    ax.set_xscale("log", base=2)
    ax.minorticks_off()
    ax.set_xticks(world_sizes)
    ax.set_xticklabels([str(n) for n in world_sizes], fontsize=TICK_SIZE)
    ax.set_xlabel("World size N (Fargate tasks)", fontsize=FONT_SIZE)


def _style(arm):
    return ARM_STYLE.get(arm, {**DEFAULT_STYLE, "label": arm})


def _line_chart(rows, metric, ylabel, title, name, output_dir, chart_format, chart_dpi,
                comparable_labels=False):
    """One line per arm against N, linear time axis from zero."""
    fig, ax = plt.subplots(figsize=(10, 6))
    world_sizes = sorted({row["world_size"] for row in rows})
    for arm in _arms(rows):
        xs, ys, errs = _series(rows, arm, metric)
        if not xs:
            continue
        style = _style(arm)
        label = style["label"]
        if comparable_labels and arm in COMPARABLE_LABEL:
            label = f"{label}, {COMPARABLE_LABEL[arm]}"
        ax.errorbar(xs, ys, yerr=errs if any(errs) else None, marker=style["marker"],
                    linestyle=style.get("linestyle", "-"), color=style["color"],
                    ecolor=style["color"], lw=2, capsize=5, label=label)
    _log2_world_size_axis(ax, world_sizes)
    ax.set_ylim(bottom=0)
    ax.minorticks_off()
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    _legend_below(ax)
    return _save(fig, output_dir, name, chart_format, chart_dpi)


def chart_comparable_time(rows, scaling, output_dir, chart_format, chart_dpi):
    """End-to-end comparable time against N, one line per arm."""
    return _line_chart(rows, "comparable_s", "End-to-end time, slowest rank (s)",
                       f"Experiment E end-to-end time vs N ({scaling} scaling)",
                       f"exp_e_{scaling}_comparable_time", output_dir, chart_format,
                       chart_dpi, comparable_labels=True)


def chart_time_to_result(rows, scaling, output_dir, chart_format, chart_dpi):
    """Framework setup plus end-to-end time against N, one line per arm."""
    return _line_chart(rows, "time_to_result_s", "Framework setup + end-to-end time (s)",
                       f"Experiment E time to result vs N ({scaling} scaling)",
                       f"exp_e_{scaling}_time_to_result", output_dir, chart_format,
                       chart_dpi, comparable_labels=True)


def chart_framework_setup(rows, scaling, output_dir, chart_format, chart_dpi):
    """Each arm's framework setup against N, measured on the last rank to arrive."""
    arms = [arm for arm in _arms(rows) if any(_series(rows, arm, "framework_setup_s")[1])]
    if not arms:
        return None
    world_sizes = sorted({row["world_size"] for row in rows if row["arm"] in arms})
    width = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, arm in enumerate(arms):
        by_n = {x: (y, e) for x, y, e in zip(*_series(rows, arm, "framework_setup_s"))}
        positions = [j + (i - (len(arms) - 1) / 2) * width
                     for j, n in enumerate(world_sizes) if n in by_n]
        heights = [by_n[n][0] for n in world_sizes if n in by_n]
        style = _style(arm)
        ax.bar(positions, heights, width,
               color=style["color"], hatch=style.get("hatch"), alpha=0.9,
               edgecolor=BAR_EDGE, linewidth=BAR_LW, label=style["label"])
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xticks(range(len(world_sizes)))
    ax.set_xticklabels([str(n) for n in world_sizes], fontsize=TICK_SIZE)
    ax.set_xlabel("World size N (Fargate tasks)", fontsize=FONT_SIZE)
    ax.set_ylabel("Framework setup, last rank to arrive (s)", fontsize=FONT_SIZE)
    ax.set_title(f"Experiment E framework setup vs N ({scaling} scaling)",
                 fontsize=TITLE_SIZE)
    _legend_below(ax)
    return _save(fig, output_dir, f"exp_e_{scaling}_framework_setup", chart_format, chart_dpi)


def generate_exp_e_scaling_charts(summary_csv, output_dir, chart_format="png", chart_dpi=300):
    rows = load_summary(summary_csv)
    paths = []
    for scaling in sorted({row["scaling"] for row in rows}):
        subset = [row for row in rows if row["scaling"] == scaling]
        paths.append(chart_comparable_time(subset, scaling, output_dir, chart_format, chart_dpi))
        setup = chart_framework_setup(subset, scaling, output_dir, chart_format, chart_dpi)
        if setup:
            paths.append(setup)
        paths.append(chart_time_to_result(subset, scaling, output_dir, chart_format, chart_dpi))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", default="png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    generate_exp_e_scaling_charts(args.summary, args.output, args.format, args.dpi)


if __name__ == "__main__":
    main()
