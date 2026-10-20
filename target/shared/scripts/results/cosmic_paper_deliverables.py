"""Cosmic AI Experiment 2 figures and tables in the format of the published Cosmic AI paper.

Reads the paper_metrics folder written by results.cosmic_paper_metrics and results.cosmic_repo_format
and writes, for co-authors, the figures of AI-for-Astronomy aws/plots.ipynb and of the paper's Figs. 8
and 9 (code/Anomaly Detection/Plots/CosmicAIPlots.ipynb) at commit 566e722 for each system, 2x2
comparison figures of the systems, and the paper's tables as one workbook plus CSVs.

    python -m results.cosmic_paper_deliverables --metrics-dir <paper_metrics> --output <dir> \
        --system published=Published --system "A=Subexperiment A" --system "B=Subexperiment B" \
        --repo-batch published=25:1.0 --speedup A B
"""

import argparse
import colorsys
import os
from collections import namedtuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUMMARY_FILE = "paper_metrics_summary.csv"
COST_FILE = "paper_cost_table.csv"
REPO_DIR = "cosmic_ai_format"
WORKBOOK = "cosmic_ai_paper_tables.xlsx"

JUPYTER_INLINE_RC = {"figure.subplot.bottom": 0.125}
PAPER_RC = {**JUPYTER_INLINE_RC, "font.size": 24, "axes.titlesize": 24, "axes.labelsize": 28, "xtick.labelsize": 24,
            "ytick.labelsize": 24, "legend.fontsize": 24, "figure.titlesize": 36, "axes.titlepad": 15,
            "xtick.major.size": 15, "xtick.major.width": 2, "ytick.major.size": 14, "ytick.major.width": 2.4,
            "lines.linewidth": 2.5}
PAPER_DPI = 200
NOTEBOOK_SCALING_FIGSIZE = (12, 6.5)
NOTEBOOK_FIGSIZE = (11, 6)
PARTITION_COLORS = {25: "#1f77b4", 50: "#ff7f0e", 75: "#2ca02c", 100: "#d62728"}
PARTITION_MARKERS = ["o", "s", "h", "D"]
LINES_LIMITS = (0.95, 1.1)
LINE_LIMITS = (0.95, 1.05)

CAI_FIGSIZE = (10, 6)
CAI_BASE_PURPLE = (0.5, 0.2, 0.6)
CAI_LIGHTNESS_STEP = 0.15
CAI_LINEWIDTH = 2.75
CAI_MARKERSIZE = 12
CAI_MARKER_EDGE = ("w", 0.75)
CAI_TICK_FONT, CAI_LABEL_FONT, CAI_TITLE_FONT = 14, 16, 20
CAI_GRID = {"color": "gray", "linewidth": 0.5, "alpha": 0.5}

COMPARISON_FIGSIZE = (22, 14)
SYSTEM_COLORS = [CAI_BASE_PURPLE, (0.2, 0.6, 0.2), (0.8, 0.6, 0.1)]
SYSTEM_MARKERS = ["o", "s", "D"]
ERROR_CAPSIZE = 5
CONTROL_SYSTEM = "published"
LEGEND_FALLBACK_LOCATIONS = ["upper left", "upper right", "lower right", "lower left", "upper center", "center right",
                             "center left", "lower center"]
LEGEND_LINE_SAMPLES = 25
LEGEND_CLEARANCE_PX = 12

Metric = namedtuple("Metric", "key label unit scale decimals axis_label title repo_column")
METRICS = (
    Metric("duration_s", "End to end time", "s", 1.0, 2, "End to End Time (s)", "End to End Time",
           "total_duration (s)"),
    Metric("map_s", "Map state time", "s", 1.0, 2, "Map State Time (s)", "Map State (Inference) Time",
           "inference_duration (s)"),
    Metric("cpu_time_s", "CPU time per worker", "s", 1.0, 2, "CPU Time per Worker (s)", "Per Worker CPU Time",
           "total_cpu_time (seconds)"),
    Metric("throughput_map_bps", "Adjusted throughput", "billion bps", 1e9, 3, "Throughput (bps)",
           "Adjusted Throughput", "throughput_bps"),
)
COMPARISON_FILES = {"duration_s": "compare_end_to_end_time.png", "map_s": "compare_map_state_time.png",
                    "cpu_time_s": "compare_cpu_time.png", "throughput_map_bps": "compare_adjusted_throughput.png"}

PAPER_TABLE6_EXTENDED_ROW = {"Partition (MB)": 100, "Data": "1 TB", "Requests": 10240, "duration": 23.0,
                             "memory": 7.0, "estimate": 27.48}
FULL_DATASET_GB = 12.6
FULL_DATASET_LABEL = "12.6 GB"
NOTES = [
    "Published values come from the paper version of AI-for-Astronomy (commit 566e722). Subexperiment A collects "
    "results through S3 after the Map state. Subexperiment B gathers results along a binomial tree inside the Map "
    "state.",
    "End to end time is the fair comparison between Subexperiments A and B, because B gathers its results inside "
    "the Map state and A collects them after it. The Map state time is kept because it is the paper's own measure.",
    "Every mean is over the measured runs (3 per scaling configuration, 4 per batch configuration of A and B, 3 per "
    "published batch configuration) and every std is the sample standard deviation.",
    "11 Subexperiment A runs (1 GB at 50, 75 and 100 MB partitions and 2 GB at 100 MB) take the Map state time from "
    "the Lambda logs because their Step Functions history no longer exists. On 20 runs with history this reads "
    "0.13 s longer on average.",
    "Per worker CPU time depends on the AWS account. Subexperiment B ran entirely in the second account, so its "
    "lower CPU time in the batch sweep is not evidence for B.",
    "The paper's batch sweep used 25 MB partitions at 1 GB. This campaign's batch sweep used 100 MB partitions at "
    "1 GB, so the published batch values are not one to one comparable with A and B.",
    "The published end to end times come from result_stats_adjusted.csv. Their sum over the 28 configurations is "
    "1,216 s, against 1,239 s in the hand entered total_execution_time.csv.",
    "Table 6 cost estimate is requests x CPU time x peak memory x $0.0000166667 per GB second, as in the paper. "
    "Billed cost prices the configured 10,240 MB for each billed duration, plus Initialize and Summarize. The 1 TB "
    "row is the paper's own and was not measured in this campaign.",
    "The paper's printed 12.6 GB estimates (0.16, 0.20, 0.30, 0.38) are 10 to 19 percent above what its own formula "
    "gives from its printed inputs (0.14, 0.19, 0.27, 0.32). Its 1 TB row (27.48) does follow the formula. The "
    "Published columns copy the printed values.",
]


def mean_std_text(mean, std, decimals):
    """'mean ± std' with the given decimals, empty when the mean is missing."""
    if pd.isna(mean):
        return ""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _metric_column(metric):
    return f"{metric.label} ({metric.unit})"


def _speedup_column(systems, speedup):
    slower, faster = speedup
    return f"End to end speedup of {systems[faster]} over {slower} (x)"


def _scaling_rows(summary, system):
    rows = summary[(summary.system == system) & (summary.series == "scaling")]
    return rows.set_index(["partition_mb", "data_gb"]).sort_index()


def scaling_table(summary, systems, speedup=("A", "B")):
    """Mean and std of every metric per partition and data size, one block of columns per system."""
    frames = {system: _scaling_rows(summary, system) for system in systems}
    index = frames[next(iter(systems))].index
    workers = pd.DataFrame({s: f.workers_mean.reindex(index) for s, f in frames.items()})
    if (workers.nunique(axis=1) > 1).any():
        raise ValueError("systems disagree on the worker count of a scaling configuration")
    table = pd.DataFrame({"Partition (MB)": index.get_level_values(0), "Data (GB)": index.get_level_values(1),
                          "Workers": workers.iloc[:, 0].to_numpy()})
    for system, label in systems.items():
        table[f"{label} runs"] = frames[system].n.reindex(index).to_numpy()
    for metric in METRICS:
        for system, label in systems.items():
            frame = frames[system].reindex(index)
            mean = frame[f"{metric.key}_mean"].to_numpy() / metric.scale
            std = frame[f"{metric.key}_std"].to_numpy() / metric.scale
            column = f"{_metric_column(metric)} {label}"
            table[f"{column} mean"], table[f"{column} std"] = mean, std
            table[f"{column} mean ± std"] = [mean_std_text(m, s, metric.decimals) for m, s in zip(mean, std)]
    end_to_end = _metric_column(METRICS[0])
    slower, faster = speedup
    table[_speedup_column(systems, speedup)] = (table[f"{end_to_end} {systems[slower]} mean"]
                                                / table[f"{end_to_end} {systems[faster]} mean"])
    return table


def headline_table(summary, systems, speedup=("A", "B"), data_gb=FULL_DATASET_GB):
    """Every metric at the full dataset, one row per partition and metric, systems side by side."""
    frames = {system: _scaling_rows(summary, system).xs(data_gb, level="data_gb") for system in systems}
    slower, faster = speedup
    rows = []
    for partition in frames[next(iter(systems))].index:
        for metric in METRICS:
            row = {"Partition (MB)": partition, "Metric": metric.label, "Unit": metric.unit}
            for system, label in systems.items():
                mean = frames[system].loc[partition, f"{metric.key}_mean"] / metric.scale
                std = frames[system].loc[partition, f"{metric.key}_std"] / metric.scale
                row[f"{label} mean"], row[f"{label} std"] = mean, std
                row[f"{label} mean ± std"] = mean_std_text(mean, std, metric.decimals)
            row[_speedup_column(systems, speedup)] = (row[f"{systems[slower]} mean"] / row[f"{systems[faster]} mean"]
                                                      if metric is METRICS[0] else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def _repo_batch_summary(runs, partition_mb, data_gb):
    runs = runs[runs.run.astype(str) != "mean"]
    grouped = runs.groupby("batch_size")
    frame = pd.DataFrame({"partition_mb": partition_mb, "data_gb": data_gb,
                          "workers_mean": grouped.num_worlds.mean(), "n": grouped.size()})
    for metric in METRICS:
        frame[f"{metric.key}_mean"] = grouped[metric.repo_column].mean()
        frame[f"{metric.key}_std"] = grouped[metric.repo_column].std()
    return frame


def batch_table(summary, repo_batches, systems):
    """Batch size sweep per system. Systems in repo_batches take (runs, partition_mb, data_gb) from repo CSVs."""
    frames = {}
    for system in systems:
        if system in repo_batches:
            frames[system] = _repo_batch_summary(*repo_batches[system])
        else:
            frames[system] = summary[(summary.system == system) & (summary.series == "batch")].set_index("batch_size")
    index = sorted(set().union(*(f.index for f in frames.values())))
    table = pd.DataFrame({"Batch size": index})
    for system, label in systems.items():
        frame = frames[system].reindex(index)
        table[f"{label} partition (MB)"] = frame.partition_mb.to_numpy()
        table[f"{label} data (GB)"] = frame.data_gb.to_numpy()
        table[f"{label} workers"] = frame.workers_mean.to_numpy()
        table[f"{label} runs"] = frame.n.to_numpy()
    for metric in METRICS:
        for system, label in systems.items():
            frame = frames[system].reindex(index)
            mean = frame[f"{metric.key}_mean"].to_numpy() / metric.scale
            std = frame[f"{metric.key}_std"].to_numpy() / metric.scale
            column = f"{_metric_column(metric)} {label}"
            table[f"{column} mean"], table[f"{column} std"] = mean, std
            table[f"{column} mean ± std"] = [mean_std_text(m, s, metric.decimals) for m, s in zip(mean, std)]
    return table


def table6(cost, systems):
    """The paper's Table 6 at the full dataset per system, with the billed cost, then the paper's 1 TB row."""
    partitions = sorted(cost.partition_mb.unique())
    rows = []
    for partition in partitions:
        group = cost[cost.partition_mb == partition].set_index("system")
        requests = group.requests.reindex(list(systems)).unique()
        if len(requests) != 1:
            raise ValueError(f"systems disagree on the request count at {partition} MB")
        row = {"Partition (MB)": partition, "Data": FULL_DATASET_LABEL, "Requests": int(requests[0])}
        for system, label in systems.items():
            entry = group.loc[system]
            billed = pd.to_numeric(entry.billed_usd, errors="coerce")
            row[f"{label} Lambda duration (s)"] = entry.cpu_time_s
            row[f"{label} memory (GB)"] = entry.peak_memory_gb
            row[f"{label} cost estimate ($)"] = entry.paper_estimate_usd
            row[f"{label} billed cost ($)"] = billed
            row[f"{label} billed cost std ($)"] = pd.to_numeric(entry.billed_usd_std, errors="coerce")
            row[f"{label} billed over estimate (x)"] = billed / entry.paper_estimate_usd
        rows.append(row)
    extended = {k: PAPER_TABLE6_EXTENDED_ROW[k] for k in ("Partition (MB)", "Data", "Requests")}
    published = next(label for system, label in systems.items() if system == CONTROL_SYSTEM)
    extended[f"{published} Lambda duration (s)"] = PAPER_TABLE6_EXTENDED_ROW["duration"]
    extended[f"{published} memory (GB)"] = PAPER_TABLE6_EXTENDED_ROW["memory"]
    extended[f"{published} cost estimate ($)"] = PAPER_TABLE6_EXTENDED_ROW["estimate"]
    rows.append(extended)
    return pd.DataFrame(rows, columns=list(rows[0]))


def shared_limits(series, low, high, rc=None):
    """The paper's y limits (autoscaled limits times low and high) for all series drawn on one axis."""
    with plt.rc_context(rc or {}):
        fig, ax = plt.subplots()
        for values in series:
            ax.plot(values)
        ymin, ymax = ax.get_ylim()
        plt.close(fig)
    return ymin * low, ymax * high


def _legend_covers_data(ax, legend):
    ax.figure.canvas.draw()
    box = legend.get_window_extent().expanded(1, 1).padded(LEGEND_CLEARANCE_PX)
    for line in ax.get_lines():
        xy = line.get_xydata()
        if len(xy) == 0:
            continue
        steps = np.linspace(0, 1, LEGEND_LINE_SAMPLES)
        dense = np.vstack([a + (b - a) * steps[:, None] for a, b in zip(xy[:-1], xy[1:])] or [xy])
        points = line.get_transform().transform(dense)
        if any(box.contains(x, y) for x, y in points):
            return True
    return False


def _place_legend(ax, keep_location, **kwargs):
    """The paper's legend, moved to the first location clear of the data unless keep_location."""
    legend = ax.legend(**kwargs)
    if keep_location or not _legend_covers_data(ax, legend):
        return legend
    for location in LEGEND_FALLBACK_LOCATIONS:
        legend.remove()
        legend = ax.legend(**{**kwargs, "loc": location})
        if not _legend_covers_data(ax, legend):
            return legend
    legend.remove()
    return ax.legend(**{**kwargs, "loc": "best"})


def _repo_means(adjusted, column):
    means = adjusted[adjusted.run.astype(str) == "mean"]
    return {int(p): g.sort_values("data (GB)")[column].to_numpy() for p, g in means.groupby("partition (MB)")}


def _data_sizes(adjusted):
    return sorted(adjusted["data (GB)"].unique())


def _notebook_lines(series, sizes, ylim, title, ylabel, figsize, keep_legend, path):
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title, weight="bold")
        for i, partition in enumerate(sorted(series)):
            ax.plot(series[partition], marker=PARTITION_MARKERS[i], markersize=10, linewidth=1.5,
                    color=PARTITION_COLORS[partition], label=f"{partition}MB")
        ax.set_xticks(np.arange(len(sizes)))
        ax.set_xticklabels([f"{s:g}" for s in sizes])
        ax.set_ylim(*ylim)
        ax.set_xlabel("Dataset Size (GB)")
        ax.set_ylabel(ylabel)
        _place_legend(ax, keep_legend, ncols=2, loc="upper left")
        fig.savefig(path, bbox_inches="tight", dpi=PAPER_DPI)
        plt.close(fig)
    return path


def _notebook_line(values, batch_sizes, ylim, title, ylabel, path):
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=NOTEBOOK_FIGSIZE)
        ax.set_title(title, weight="bold")
        ax.plot(values, marker="o", markersize=12, color="black", linewidth=1.5)
        ax.set_xticks(np.arange(len(batch_sizes)))
        ax.set_xticklabels([str(int(b)) for b in batch_sizes])
        ax.set_ylim(*ylim)
        ax.set_xlabel("Batch Sizes")
        ax.set_ylabel(ylabel)
        fig.savefig(path, bbox_inches="tight", dpi=PAPER_DPI)
        plt.close(fig)
    return path


def _batch_means(batch, column):
    means = batch[batch.run.astype(str) == "mean"].sort_values("batch_size")
    return means.batch_size.to_numpy(), means[column].to_numpy()


def _lightness(color, amount):
    hue, lightness, saturation = colorsys.rgb_to_hls(*color)
    return colorsys.hls_to_rgb(hue, float(np.clip(lightness + amount, 0, 1)), saturation)


def _cai_figure(adjusted, column, ylim, metric_title, ylabel, label, legend, keep_legend, path):
    partitions = list(adjusted["partition (MB)"].unique())
    means = adjusted[adjusted.run.astype(str) == "mean"]
    with plt.style.context("default"), plt.rc_context(JUPYTER_INLINE_RC):
        fig = plt.figure(figsize=CAI_FIGSIZE)
        ax = fig.gca()
        for i, partition in enumerate(partitions):
            group = means[means["partition (MB)"] == partition].sort_values("data (GB)")
            ax.plot(group["data (GB)"], group[column], marker=PARTITION_MARKERS[i % len(PARTITION_MARKERS)],
                    linewidth=CAI_LINEWIDTH, markersize=CAI_MARKERSIZE,
                    color=_lightness(CAI_BASE_PURPLE, CAI_LIGHTNESS_STEP * i), markeredgecolor=CAI_MARKER_EDGE[0],
                    markeredgewidth=CAI_MARKER_EDGE[1], label=f"Partition {partition} MB")
        ax.set_ylim(*ylim)
        ax.tick_params(axis="both", which="major", labelsize=CAI_TICK_FONT)
        ax.yaxis.get_offset_text().set_fontsize(CAI_TICK_FONT)
        ax.set_xlabel("Dataset Size (GB)", fontsize=CAI_LABEL_FONT)
        ax.set_ylabel(ylabel, fontsize=CAI_LABEL_FONT)
        ax.set_title(f"CAI Varying Partition Sizes, {label}\n{metric_title} vs. Data Size (GB)",
                     fontsize=CAI_TITLE_FONT, fontweight="bold", y=1.0)
        handles, labels = ax.get_legend_handles_labels()
        if legend.pop("reverse"):
            handles, labels = handles[::-1], labels[::-1]
        _place_legend(ax, keep_legend, handles=handles, labels=labels, **legend)
        ax.grid(axis="y", **CAI_GRID)
        fig.savefig(path, dpi=PAPER_DPI)
        plt.close(fig)
    return path


def _cai_limits(tables, column):
    series = [g.sort_values("data (GB)")[column].to_numpy()
              for t in tables for _, g in t[t.run.astype(str) == "mean"].groupby("partition (MB)")]
    return shared_limits(series, 1.0, 1.0, rc=matplotlib.rcParamsDefault)


def render_system_figures(repo, systems, output_dir):
    """The plots.ipynb figures and the paper's Figs. 8 and 9 per system, y limits shared across systems."""
    notebook_dir, paper_dir = os.path.join(output_dir, "figures_plots_notebook"), os.path.join(output_dir,
                                                                                               "figures_paper_fig8_fig9")
    os.makedirs(notebook_dir, exist_ok=True)
    os.makedirs(paper_dir, exist_ok=True)
    adjusted = {s: repo[s]["result_stats_adjusted.csv"] for s in systems}
    written = []
    for column, name, title, ylabel, figsize in (
            ("inference_duration (s)", "inference_time", "Weak Scaling", "Inference Time (s)", NOTEBOOK_SCALING_FIGSIZE),
            ("throughput_bps", "throughput", None, "Throughput (bps)", NOTEBOOK_FIGSIZE)):
        series = {s: _repo_means(adjusted[s], column) for s in systems}
        ylim = shared_limits([v for s in systems for v in series[s].values()], *LINES_LIMITS, rc=PAPER_RC)
        for system, label in systems.items():
            written.append(_notebook_lines(series[system], _data_sizes(adjusted[system]), ylim,
                                           f"{title} ({label})" if title else label, ylabel, figsize,
                                           system == CONTROL_SYSTEM, os.path.join(notebook_dir, f"{system}_{name}.png")))
    batches = {s: repo[s]["batch_varying_adjusted.csv"] for s in systems}
    worker_groups = {}
    for system in systems:
        worker_groups.setdefault(tuple(sorted(batches[system].num_worlds.unique())), []).append(system)
    for column, name, ylabel in (("total_cpu_time (seconds)", "batch_varying_inference_time", "Inference Time (s)"),
                                 ("throughput_bps", "batch_varying_throughput", "Throughput (bps)")):
        for group in worker_groups.values():
            ylim = shared_limits([_batch_means(batches[s], column)[1] for s in group], *LINE_LIMITS, rc=PAPER_RC)
            for system in group:
                sizes, values = _batch_means(batches[system], column)
                written.append(_notebook_line(values, sizes, ylim, systems[system], ylabel,
                                              os.path.join(notebook_dir, f"{system}_{name}.png")))
    inference_legend = {"reverse": True, "fontsize": 12, "title": "Partition Sizes (MB)", "title_fontsize": 14,
                        "loc": "lower right", "ncols": 2}
    throughput_legend = {"reverse": False, "fontsize": 14, "title": "Partitions (MB)", "title_fontsize": 16,
                         "loc": "upper left"}
    for column, name, metric_title, ylabel, legend in (
            ("inference_duration (s)", "cai_inference_time", "Inference Time (s)", "Inference Time (s)",
             inference_legend),
            ("throughput_bps", "cai_throughput", "Throughput (bps)", "Throughput (bps)", throughput_legend)):
        ylim = _cai_limits(adjusted.values(), column)
        for system, label in systems.items():
            written.append(_cai_figure(adjusted[system], column, ylim, metric_title, ylabel, label, dict(legend),
                                       system == CONTROL_SYSTEM, os.path.join(paper_dir, f"{system}_{name}.png")))
    return written


def render_comparison(summary, systems, metric, path):
    """One panel per partition, one line per system with sample std error bars, shared legend."""
    frames = {s: _scaling_rows(summary, s) for s in systems}
    partitions = sorted(frames[next(iter(systems))].index.get_level_values(0).unique())
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, 2, figsize=COMPARISON_FIGSIZE, sharex=True)
        for ax, partition in zip(axes.flat, partitions):
            for i, (system, label) in enumerate(systems.items()):
                rows = frames[system].xs(partition, level="partition_mb")
                ax.errorbar(np.arange(len(rows)), rows[f"{metric.key}_mean"], yerr=rows[f"{metric.key}_std"],
                            capsize=ERROR_CAPSIZE, marker=SYSTEM_MARKERS[i], markersize=10, linewidth=1.5,
                            color=SYSTEM_COLORS[i], label=label)
            ax.set_title(f"{partition}MB", weight="bold")
            ax.set_xticks(np.arange(len(rows)))
            ax.set_xticklabels([f"{d:g}" for d in rows.index])
        for ax in axes[1]:
            ax.set_xlabel("Dataset Size (GB)")
        for ax in axes[:, 0]:
            ax.set_ylabel(metric.axis_label)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(metric.title, weight="bold", fontsize=PAPER_RC["axes.labelsize"], y=1.0)
        fig.legend(handles, labels, loc="upper center", ncols=len(systems), bbox_to_anchor=(0.5, 0.975))
        fig.tight_layout(rect=(0, 0, 1, 0.945))
        fig.savefig(path, bbox_inches="tight", dpi=PAPER_DPI)
        plt.close(fig)
    return path


def _write_workbook(tables, path):
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, table in tables.items():
            table.to_excel(writer, sheet_name=sheet, index=False)
            worksheet = writer.sheets[sheet]
            worksheet.freeze_panes = "A2"
            for i, column in enumerate(table.columns, start=1):
                cell = worksheet.cell(row=1, column=i)
                cell.font = Font(bold=True)
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                width = 110 if sheet == "Notes" else min(max(len(str(column)) * 0.6, 10), 28)
                worksheet.column_dimensions[get_column_letter(i)].width = width
            if sheet == "Notes":
                for row in worksheet.iter_rows(min_row=2):
                    row[0].alignment = Alignment(wrap_text=True, vertical="top")


def write_deliverables(metrics_dir, output_dir, systems, repo_batch_configs, speedup=("A", "B")):
    """Write the tables, workbook and figures, return the written paths."""
    summary = pd.read_csv(os.path.join(metrics_dir, SUMMARY_FILE))
    cost = pd.read_csv(os.path.join(metrics_dir, COST_FILE))
    repo = {s: {name: pd.read_csv(os.path.join(metrics_dir, REPO_DIR, s, "results", name))
                for name in ("result_stats_adjusted.csv", "batch_varying_adjusted.csv")} for s in systems}
    repo_batches = {s: (repo[s]["batch_varying_adjusted.csv"], *config) for s, config in repo_batch_configs.items()}
    tables = {"Table 6 cost": table6(cost, systems), "Scaling": scaling_table(summary, systems, speedup),
              "Headline 12.6 GB": headline_table(summary, systems, speedup),
              "Batch size sweep": batch_table(summary, repo_batches, systems),
              "Notes": pd.DataFrame({"Note": NOTES})}
    files = {"Table 6 cost": "table6_cost.csv", "Scaling": "scaling_results.csv",
             "Headline 12.6 GB": "headline_12_6_gb.csv", "Batch size sweep": "batch_size_sweep.csv"}
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    rounded = {sheet: table.round(4) for sheet, table in tables.items()}
    written = []
    for sheet, name in files.items():
        path = os.path.join(tables_dir, name)
        rounded[sheet].to_csv(path, index=False)
        written.append(path)
    workbook = os.path.join(tables_dir, WORKBOOK)
    _write_workbook(rounded, workbook)
    written.append(workbook)
    written += render_system_figures(repo, systems, output_dir)
    comparison_dir = os.path.join(output_dir, "figures_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    for metric in METRICS:
        written.append(render_comparison(summary, systems, metric,
                                         os.path.join(comparison_dir, COMPARISON_FILES[metric.key])))
    return written


def _system(spec):
    key, _, label = spec.partition("=")
    return key, label or key


def _repo_batch(spec):
    key, _, config = spec.partition("=")
    partition, _, data = config.partition(":")
    return key, (int(partition), float(data))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-dir", required=True, help="paper_metrics folder with the summary, cost and repo CSVs")
    p.add_argument("--output", required=True)
    p.add_argument("--system", action="append", type=_system, required=True, metavar="KEY=LABEL")
    p.add_argument("--repo-batch", action="append", type=_repo_batch, default=[], metavar="KEY=PARTITION_MB:DATA_GB",
                   help="system whose batch sweep is read from its repo CSV, with that sweep's partition and data size")
    p.add_argument("--speedup", nargs=2, required=True, metavar=("SLOWER", "FASTER"))
    args = p.parse_args(argv)
    for path in write_deliverables(args.metrics_dir, args.output, dict(args.system), dict(args.repo_batch),
                                   tuple(args.speedup)):
        print(path)


if __name__ == "__main__":
    main()