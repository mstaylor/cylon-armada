"""Cosmic AI campaign results written in the AI-for-Astronomy aws/results and aws/figures layout.

Reads paper_metrics_runs.csv from results.cosmic_paper_metrics and writes, per system, the files the
published Cosmic AI repository holds at the paper version (commit 566e722): result_stats.csv,
result_stats_adjusted.csv, state_machine_logs.csv, batch_varying_results.csv,
batch_varying_adjusted.csv and total_execution_time.csv, plus the four figures of aws/plots.ipynb
(inference_time, throughput, batch_varying_inference_time, batch_varying_throughput).

    python -m results.cosmic_repo_format --runs paper_metrics_runs.csv --output <dir> \
        --system A="Subexperiment A (S3)" --system B="Subexperiment B (direct communication)"
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from results.chart_cosmic_campaign import PARTITION_COLORS

STATS = ["total_cpu_time (seconds)", "total_cpu_memory (MB)", "throughput_bps", "total_bits"]
TIMES = ["num_worlds", "total_duration (s)", "inference_duration (s)"]
MARKERS = ["o", "s", "h", "D"]
FIGURE_FONT = 16
FIGURE_SIZE = (12, 6.5)
FIGURE_DPI = 200


def format_duration(seconds):
    minutes, rest = divmod(float(seconds), 60)
    return f"{int(minutes):02d}:{rest:06.3f}"


def _with_mean_rows(frame, keys, values):
    runs = frame.assign(run=frame.run.astype(str))
    means = frame.groupby(keys)[values].mean().reset_index().assign(run="mean")
    combined = pd.concat([runs, means], ignore_index=True)
    combined["_order"] = combined.run.map(lambda r: float("inf") if r == "mean" else float(r))
    return combined.sort_values(keys + ["_order"]).drop(columns="_order")[keys + ["run"] + values].reset_index(drop=True)


def _published_columns(runs, throughput):
    return pd.DataFrame({"partition (MB)": runs.partition_mb, "data (GB)": runs.data_gb,
                         "batch_size": runs.batch_size, "run": runs.run_id.astype(int),
                         "total_cpu_time (seconds)": runs.cpu_time_s, "total_cpu_memory (MB)": runs.memory_mb,
                         "throughput_bps": runs[throughput], "total_bits": runs.total_bits,
                         "num_worlds": runs.workers.astype(int), "total_duration (s)": runs.duration_s,
                         "inference_duration (s)": runs.map_s})


def repo_tables(runs):
    """The aws/results tables for one system's runs, as stats.py and collect_logs.ipynb lay them out."""
    scaling, batch = runs[runs.series == "scaling"], runs[runs.series == "batch"]
    keys = ["partition (MB)", "data (GB)"]
    raw, adjusted = _published_columns(scaling, "throughput_sum_bps"), _published_columns(scaling, "throughput_map_bps")
    batch_raw, batch_adjusted = (_published_columns(batch, "throughput_sum_bps"),
                                 _published_columns(batch, "throughput_map_bps"))
    logs = _published_columns(pd.concat([scaling, batch]), "throughput_map_bps").assign(
        batch_varying=[s == "batch" for s in pd.concat([scaling, batch]).series])
    total = pd.DataFrame({"partition(MB)": scaling.partition_mb, "data(GB)": scaling.data_gb,
                          "run": scaling.run_id.astype(int), "num_worlds": scaling.workers.astype(float),
                          "duration": [format_duration(s) for s in scaling.duration_s]})
    return {
        "result_stats.csv": _with_mean_rows(raw, keys, STATS),
        "result_stats_adjusted.csv": _with_mean_rows(adjusted, keys, STATS + TIMES),
        "batch_varying_results.csv": _with_mean_rows(batch_raw, ["batch_size"], STATS),
        "batch_varying_adjusted.csv": _with_mean_rows(batch_adjusted, ["batch_size"], STATS + TIMES),
        "state_machine_logs.csv": logs[["partition (MB)", "data (GB)", "batch_size", "batch_varying", "run",
                                        "num_worlds", "total_duration (s)", "inference_duration (s)"]]
        .sort_values(["partition (MB)", "data (GB)", "batch_size", "batch_varying", "run"]).reset_index(drop=True),
        "total_execution_time.csv": total.sort_values(["partition(MB)", "data(GB)", "run"]).reset_index(drop=True),
    }


def _figure():
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.tick_params(labelsize=FIGURE_FONT)
    return fig, ax


def _finish(fig, ax, xlabel, ylabel, title, output_dir, name, legend):
    ax.set_xlabel(xlabel, fontsize=FIGURE_FONT + 2)
    ax.set_ylabel(ylabel, fontsize=FIGURE_FONT + 2)
    ax.set_title(title, fontsize=FIGURE_FONT + 2, weight="bold")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin * 0.95, ymax * 1.05)
    if legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=4, frameon=True, fontsize=FIGURE_FONT)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, name), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return name


def _partition_lines(adjusted, column, scale, label, ylabel, title, output_dir, name):
    means = adjusted[adjusted.run == "mean"]
    sizes = sorted(means["data (GB)"].unique())
    fig, ax = _figure()
    for i, (partition, group) in enumerate(means.groupby("partition (MB)")):
        group = group.sort_values("data (GB)")
        ax.plot([sizes.index(d) for d in group["data (GB)"]], group[column] / scale, marker=MARKERS[i % len(MARKERS)],
                markersize=10, linewidth=1.5, color=PARTITION_COLORS.get(int(partition)), label=f"{int(partition)}MB")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{d:g}" for d in sizes])
    return _finish(fig, ax, "Dataset Size (GB)", ylabel, f"{title}, {label}", output_dir, name, legend=True)


def _batch_line(batch, column, scale, ylabel, title, output_dir, name):
    means = batch[batch.run == "mean"].sort_values("batch_size")
    fig, ax = _figure()
    ax.plot(range(len(means)), means[column] / scale, marker="o", markersize=12, color="black", linewidth=1.5)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([str(int(b)) for b in means.batch_size])
    return _finish(fig, ax, "Batch Sizes", ylabel, title, output_dir, name, legend=False)


def render_repo_figures(tables, output_dir, label):
    """The four figures of aws/plots.ipynb at the paper version, drawn from this system's tables."""
    adjusted, batch = tables["result_stats_adjusted.csv"], tables["batch_varying_adjusted.csv"]
    names =[_partition_lines(adjusted, "inference_duration (s)", 1.0, label, "Inference Time (s)", "Weak Scaling",
                              output_dir, "inference_time.jpg"),
             _partition_lines(adjusted, "throughput_bps", 1e9, label, "Throughput (billion bps)", "Throughput",
                              output_dir, "throughput.jpg")]
    if len(batch):
        names += [_batch_line(batch, "total_cpu_time (seconds)", 1.0, "Inference Time (s)",
                              f"Batch size sweep, {label}", output_dir, "batch_varying_inference_time.jpg"),
                  _batch_line(batch, "throughput_bps", 1e9, "Throughput (billion bps)",
                              f"Batch size sweep, {label}", output_dir, "batch_varying_throughput.jpg")]
    return names


def _system(spec):
    key, _, label = spec.partition("=")
    return key, label or key


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="paper_metrics_runs.csv from results.cosmic_paper_metrics")
    p.add_argument("--system", action="append", type=_system, required=True, metavar="KEY=LABEL")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    runs = pd.read_csv(args.runs)
    for key, label in args.system:
        tables = repo_tables(runs[runs.system == key])
        results_dir = os.path.join(args.output, key, "results")
        os.makedirs(results_dir, exist_ok=True)
        for name, table in tables.items():
            table.round(2).to_csv(os.path.join(results_dir, name), index=False)
            print(os.path.join(results_dir, name))
        for name in render_repo_figures(tables, os.path.join(args.output, key, "figures"), label):
            print(os.path.join(args.output, key, "figures", name))


if __name__ == "__main__":
    main()
