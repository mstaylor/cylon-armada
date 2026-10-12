"""Charts for the Cosmic AI campaign (Experiment 1): reproduction against the published timings,
the per-stage time breakdown, result-collection scaling and the batch sweep.

Reads the summary CSV written by results.cosmic_lambda_results and the published
AI-for-Astronomy aws/results/total_execution_time.csv. Style follows chart_zerocopy.py
and the quals scaling notebook.

Usage:
    python -m results.chart_cosmic_campaign --summary exp1_summary.csv \
        --published total_execution_time.csv --output docs/cosmic_campaign/exp1 [--format png] [--dpi 300]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FONT_SIZE = 12
TITLE_SIZE = 14
LEGEND_SIZE = 10
BAR_EDGE = "black"
BAR_LW = 0.8
CAPSIZE = 5
STAGE_COLORS = {"inference": "#1f77b4", "collect": "#d62728", "other": "#bdbdbd"}
PARTITION_COLORS = {25: "#d62728", 50: "#ff7f0e", 75: "#1f77b4", 100: "#2ca02c"}
PARTITION_MARKERS = {25: "o", 50: "s", 75: "^", 100: "D"}

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "axes.grid": False})


def published_seconds(durations):
    parts = durations.str.split(":", expand=True).astype(float)
    return parts[0] * 60 + parts[1]


def compare_with_published(summary, published):
    pub = published.assign(seconds=published_seconds(published["duration"]))
    pub = (pub.groupby(["partition(MB)", "data(GB)"])["seconds"].agg(["mean", "std", "count"]).reset_index()
           .rename(columns={"partition(MB)": "partition_mb", "data(GB)": "data_gb", "mean": "published_mean_s",
                            "std": "published_std_s", "count": "published_n"}))
    ours = summary[(summary.series == "scaling") & (summary.phase == "measured")]
    joined = ours.merge(pub, on=["partition_mb", "data_gb"]).sort_values(["partition_mb", "data_gb"])
    return joined.assign(ratio=joined.duration_s_mean / joined.published_mean_s).reset_index(drop=True)


def _legend_below(ax_or_fig, handles=None, labels=None, ncol=2, anchor=(0.5, -0.32)):
    kwargs = {"loc": "lower center", "bbox_to_anchor": anchor, "ncol": ncol, "frameon": True,
              "fontsize": LEGEND_SIZE}
    if handles is not None:
        kwargs.update(handles=handles, labels=labels)
    ax_or_fig.legend(**kwargs)


SERIES_STYLES = {"published": ("Published", "#7f7f7f", "o"), "A": ("Arm A (S3)", "orange", "D"),
                 "B": ("Arm B (direct communication)", "green", "s")}


def line_chart(series, tick_labels, xlabel, ylabel, title, output_dir, name, chart_format, dpi):
    """One plot, evenly spaced categories, one line per series with the run-group std as error bars.

    `series` is a list of (label, color, marker, means, stds); a std of None draws no error bar.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = [str(t) for t in tick_labels]
    for label, color, marker, means, stds in series:
        ax.plot(x, means, marker=marker, color=color, label=label, linewidth=2)
        if stds is not None:
            ax.errorbar(x, means, yerr=[0 if pd.isna(s) else s for s in stds], fmt="none", color=color,
                        capsize=CAPSIZE)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    _legend_below(ax, ncol=len(series) if len(series) <= 3 else 2, anchor=(0.5, -0.3))
    return _save(fig, output_dir, name, chart_format, dpi)


def data_size_ticks(rows):
    return [f"{d:g}\n({w})" for d, w in zip(rows.data_gb, rows.workers)]


def _save(fig, output_dir, name, chart_format, dpi):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{name}.{chart_format}"
    fig.savefig(os.path.join(output_dir, filename), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filename


def chart_reproduction(joined, output_dir, chart_format, dpi):
    files = []
    for partition in sorted(joined.partition_mb.unique()):
        rows = joined[joined.partition_mb == partition].sort_values("data_gb")
        published, ours = SERIES_STYLES["published"], SERIES_STYLES["A"]
        files.append(line_chart(
            [(published[0], published[1], published[2], rows.published_mean_s, rows.published_std_s),
             ("This reproduction", ours[1], ours[2], rows.duration_s_mean, rows.duration_s_std)],
            data_size_ticks(rows), "Dataset size in GB (workers)", "Execution time (s)",
            f"Reproduction against the published runs, {partition} MB partitions",
            output_dir, f"exp1_vs_published_{partition}MB", chart_format, dpi))
    return files


def chart_breakdown(summary, output_dir, chart_format, dpi):
    rows = summary[(summary.series == "scaling") & (summary.phase == "measured")]
    rows = rows[rows.partition_mb == rows.partition_mb.min()].sort_values("workers")
    inference, collect = rows.inference_s_max_mean, rows.collect_s_mean
    other = (rows.duration_s_mean - inference - collect).clip(lower=0)
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = 0
    for label, values, color in (("Inference (slowest worker)", inference, STAGE_COLORS["inference"]),
                                 ("Result collection", collect, STAGE_COLORS["collect"]),
                                 ("Other (start up, scheduling)", other, STAGE_COLORS["other"])):
        ax.bar(list(x), values, bottom=bottom, color=color, edgecolor=BAR_EDGE, linewidth=BAR_LW, label=label)
        bottom = bottom + values.values
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(w) for w in rows.workers])
    ax.set_xlabel("Workers", fontsize=FONT_SIZE)
    ax.set_ylabel("Time (s)", fontsize=FONT_SIZE)
    ax.set_title(f"Where the time goes, {int(rows.partition_mb.iloc[0])} MB partitions (Arm A)",
                 fontsize=TITLE_SIZE)
    _legend_below(ax, ncol=3)
    return _save(fig, output_dir, "exp1_time_breakdown", chart_format, dpi)


def chart_collect_scaling(summary, output_dir, chart_format, dpi):
    rows = summary[(summary.series == "scaling") & (summary.phase == "measured")]
    fig, ax = plt.subplots(figsize=(10, 6))
    for partition in sorted(rows.partition_mb.unique()):
        series = rows[rows.partition_mb == partition].sort_values("workers")
        ax.plot(series.workers, series.collect_s_mean, marker="o", color=PARTITION_COLORS.get(partition),
                markeredgecolor=BAR_EDGE, label=f"{partition} MB partitions")
    ax.set_xlabel("Workers", fontsize=FONT_SIZE)
    ax.set_ylabel("Result collection time (s)", fontsize=FONT_SIZE)
    ax.set_title("Result collection grows linearly with workers (Arm A, S3)", fontsize=TITLE_SIZE)
    _legend_below(ax, ncol=4)
    return _save(fig, output_dir, "exp1_collect_scaling", chart_format, dpi)


def chart_batch_sweep(summary, output_dir, chart_format, dpi):
    rows = summary[(summary.series == "batch") & (summary.phase == "measured")].sort_values("batch_size")
    first = rows.iloc[0] if len(rows) else None
    where = f", {int(first.partition_mb)} MB partitions, {first.data_gb:g} GB" if first is not None else ""
    ours = SERIES_STYLES["A"]
    return line_chart([("Arm A (S3)", ours[1], ours[2], rows.duration_s_mean, rows.duration_s_std)],
                      list(rows.batch_size), "Batch size", "Execution time (s)", f"Batch size sweep{where}",
                      output_dir, "exp1_batch_sweep", chart_format, dpi)


def render_all(summary, published, output_dir, chart_format, dpi):
    joined = compare_with_published(summary, published)
    return [*chart_reproduction(joined, output_dir, chart_format, dpi),
            chart_breakdown(summary, output_dir, chart_format, dpi),
            chart_collect_scaling(summary, output_dir, chart_format, dpi),
            chart_batch_sweep(summary, output_dir, chart_format, dpi)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--published", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--format", default="png")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    summary, published = pd.read_csv(args.summary), pd.read_csv(args.published)
    os.makedirs(args.output, exist_ok=True)
    compare_with_published(summary, published).to_csv(os.path.join(args.output, "exp1_vs_published.csv"),
                                                      index=False)
    for name in render_all(summary, published, args.output, args.format, args.dpi):
        print(os.path.join(args.output, name))


if __name__ == "__main__":
    main()