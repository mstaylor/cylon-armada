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
OURS, PUBLISHED = "#2ca02c", "#7f7f7f"
STAGE_COLORS = {"inference": "#1f77b4", "collect": "#d62728", "other": "#bdbdbd"}
PARTITION_COLORS = {25: "#d62728", 50: "#ff7f0e", 75: "#1f77b4", 100: "#2ca02c"}

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


def _save(fig, output_dir, name, chart_format, dpi):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{name}.{chart_format}"
    fig.savefig(os.path.join(output_dir, filename), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filename


def chart_reproduction(joined, output_dir, chart_format, dpi):
    partitions = sorted(joined.partition_mb.unique())
    fig, axes = plt.subplots(1, len(partitions), figsize=(4 * len(partitions), 4.5), sharey=True, squeeze=False)
    width = 0.38
    for ax, partition in zip(axes[0], partitions):
        rows = joined[joined.partition_mb == partition]
        x = range(len(rows))
        ax.bar([i - width / 2 for i in x], rows.duration_s_mean, width, yerr=rows.duration_s_std.fillna(0),
               capsize=CAPSIZE, color=OURS, edgecolor=BAR_EDGE, linewidth=BAR_LW, label="This reproduction")
        ax.bar([i + width / 2 for i in x], rows.published_mean_s, width, yerr=rows.published_std_s.fillna(0),
               capsize=CAPSIZE, color=PUBLISHED, edgecolor=BAR_EDGE, linewidth=BAR_LW, label="Published")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{d:g}" for d in rows.data_gb])
        ax.set_xlabel("Data size (GB)", fontsize=FONT_SIZE)
        ax.set_title(f"{partition} MB partitions", fontsize=FONT_SIZE)
    axes[0][0].set_ylabel("Execution time (s)", fontsize=FONT_SIZE)
    fig.suptitle("Cosmic AI on Lambda: reproduction vs published", fontsize=TITLE_SIZE)
    handles, labels = axes[0][0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, anchor=(0.5, -0.12))
    return _save(fig, output_dir, "exp1_vs_published", chart_format, dpi)


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
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(rows))
    ax.bar(list(x), rows.duration_s_mean, yerr=rows.duration_s_std.fillna(0), capsize=CAPSIZE, color=OURS,
           edgecolor=BAR_EDGE, linewidth=BAR_LW, label="Execution time (mean of measured runs)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(b) for b in rows.batch_size])
    ax.set_xlabel("Batch size", fontsize=FONT_SIZE)
    ax.set_ylabel("Execution time (s)", fontsize=FONT_SIZE)
    first = rows.iloc[0] if len(rows) else None
    where = f", {int(first.partition_mb)} MB partitions, {first.data_gb:g} GB" if first is not None else ""
    ax.set_title(f"Batch size sweep{where}", fontsize=TITLE_SIZE)
    _legend_below(ax, ncol=1)
    return _save(fig, output_dir, "exp1_batch_sweep", chart_format, dpi)


def render_all(summary, published, output_dir, chart_format, dpi):
    joined = compare_with_published(summary, published)
    return [chart_reproduction(joined, output_dir, chart_format, dpi),
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