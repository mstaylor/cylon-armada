"""Charts and tables for Cosmic AI Experiment 2: Arm A (S3) against Arm B (direct communication)
beside the published timings, Experiment 1 against Experiment 2 (the agent's cost), and the
planning metrics for every arm and cache mode.

Reads the summary CSV written by results.cosmic_lambda_results over the Experiment 1 and 2
manifests, the published AI-for-Astronomy aws/results/total_execution_time.csv, and the
plans_<arm>_<mode>/plan_metrics.json files written by cosmic_campaign_exp2.py merge.

Usage:
    python -m results.chart_cosmic_comparison --summary exp2_summary.csv \
        --published total_execution_time.csv --plans-root results/cosmic_exp2 \
        --output docs/cosmic_campaign/exp2 [--format png] [--dpi 300]
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from results.chart_cosmic_campaign import (
    BAR_EDGE, BAR_LW, CAPSIZE, FONT_SIZE, PARTITION_COLORS, PUBLISHED, TITLE_SIZE, _legend_below, _save,
    published_seconds,
)

ARM_COLORS = {"A": "#1f77b4", "B": "#2ca02c"}
ARM_LABELS = {"A": "Arm A (S3, published pipeline)", "B": "Arm B (direct communication)"}
EXPERIMENT_COLORS = {"exp1": "#7f7f7f", "exp2": "#ff7f0e"}
CONFIGURATION = ["series", "partition_mb", "data_gb", "batch_size", "workers"]


def _measured(summary, arm):
    return summary[(summary.phase == "measured") & (summary.arm == arm)]


def _arm_rows(summary, arm, experiment):
    rows = _measured(summary, arm)
    preferred = rows[rows.experiment == experiment]
    have = set(map(tuple, preferred[CONFIGURATION].values.tolist()))
    fallback = rows[(rows.experiment != experiment) & (rows.series == "baseline")]
    fallback = fallback[[tuple(v) not in have for v in fallback[CONFIGURATION].values.tolist()]]
    picked = pd.concat([preferred, fallback]).drop_duplicates(CONFIGURATION)
    prefix = arm.lower()
    return picked[CONFIGURATION + ["experiment", "n", "duration_s_mean", "duration_s_std"]].rename(columns={
        "experiment": f"{prefix}_experiment", "n": f"{prefix}_n",
        "duration_s_mean": f"{prefix}_mean_s", "duration_s_std": f"{prefix}_std_s"})


def compare_arms(summary, experiment="exp2"):
    """Arm A against Arm B per configuration, from `experiment`; baselines, which Experiment 2
    does not plan, come from whichever experiment ran them."""
    joined = _arm_rows(summary, "A", experiment).merge(_arm_rows(summary, "B", experiment), on=CONFIGURATION)
    joined = joined.assign(speedup=joined.a_mean_s / joined.b_mean_s)
    return joined.sort_values(["workers", "series", "partition_mb", "batch_size"]).reset_index(drop=True)


def compare_experiments(summary, arm="A"):
    """Experiment 1 against Experiment 2 per configuration for one arm: the agent's cost."""
    rows = _measured(summary, arm)
    columns = CONFIGURATION + ["duration_s_mean", "duration_s_std", "n"]
    exp1 = rows[rows.experiment == "exp1"][columns].rename(
        columns={"duration_s_mean": "exp1_mean_s", "duration_s_std": "exp1_std_s", "n": "exp1_n"})
    exp2 = rows[rows.experiment == "exp2"][columns].rename(
        columns={"duration_s_mean": "exp2_mean_s", "duration_s_std": "exp2_std_s", "n": "exp2_n"})
    joined = exp1.merge(exp2, on=CONFIGURATION)
    joined = joined.assign(difference_s=joined.exp2_mean_s - joined.exp1_mean_s)
    return joined.sort_values(["series", "workers", "partition_mb", "batch_size"]).reset_index(drop=True)


def planning_table(plans_root):
    rows = []
    for path in sorted(glob.glob(os.path.join(plans_root, "plans_*_*", "plan_metrics.json"))):
        arm, mode = os.path.basename(os.path.dirname(path)).split("_", 2)[1:]
        metrics = json.load(open(path))
        hits = metrics.get("cache_hits_exact", 0) + metrics.get("cache_hits_structural", 0)
        requests = metrics["requests"]
        rows.append({"arm": arm, "cache_mode": mode, **metrics,
                     "llm_calls_total": metrics.get("llm_calls", 0) + metrics.get("llm_tool_calls", 0),
                     "cache_hits": hits,
                     "cache_hit_rate": hits / requests,
                     "plan_correct_rate": 1 - metrics["wrong_plans"] / requests,
                     "wrong_reuse_rate": metrics["wrong_reuse"] / hits if hits else 0.0,
                     "planning_wall_s": (metrics.get("planning_wall_ms") or 0) / 1000})
    return pd.DataFrame(rows)


def _published_means(published):
    pub = published.assign(seconds=published_seconds(published["duration"]))
    return (pub.groupby(["partition(MB)", "data(GB)"])["seconds"].agg(["mean", "std"]).reset_index()
            .rename(columns={"partition(MB)": "partition_mb", "data(GB)": "data_gb",
                             "mean": "published_mean_s", "std": "published_std_s"}))


def chart_arms_vs_published(joined, published, output_dir, chart_format, dpi):
    rows = joined[joined.series == "scaling"].merge(_published_means(published), on=["partition_mb", "data_gb"],
                                                    how="left")
    partitions = sorted(rows.partition_mb.unique())
    fig, axes = plt.subplots(1, len(partitions), figsize=(4.5 * len(partitions), 4.5), sharey=True, squeeze=False)
    width = 0.27
    for ax, partition in zip(axes[0], partitions):
        part = rows[rows.partition_mb == partition].sort_values("data_gb")
        x = list(range(len(part)))
        bars = (("Published", part.published_mean_s, part.published_std_s, PUBLISHED, -width),
                (ARM_LABELS["A"], part.a_mean_s, part.a_std_s, ARM_COLORS["A"], 0.0),
                (ARM_LABELS["B"], part.b_mean_s, part.b_std_s, ARM_COLORS["B"], width))
        for label, mean, std, color, offset in bars:
            ax.bar([i + offset for i in x], mean, width, yerr=std.fillna(0), capsize=CAPSIZE, color=color,
                   edgecolor=BAR_EDGE, linewidth=BAR_LW, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d:g}\n({w})" for d, w in zip(part.data_gb, part.workers)])
        ax.set_xlabel("Data size in GB (workers)", fontsize=FONT_SIZE)
        ax.set_title(f"{partition} MB partitions", fontsize=FONT_SIZE)
    axes[0][0].set_ylabel("Execution time (s)", fontsize=FONT_SIZE)
    fig.suptitle("Cosmic AI on Lambda: S3 against direct communication", fontsize=TITLE_SIZE)
    handles, labels = axes[0][0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=3, anchor=(0.5, -0.14))
    return _save(fig, output_dir, "exp2_arms_vs_published", chart_format, dpi)


def chart_speedup(joined, output_dir, chart_format, dpi):
    rows = joined[joined.series.isin(["scaling", "baseline"])]
    fig, ax = plt.subplots(figsize=(10, 6))
    for partition in sorted(rows.partition_mb.unique()):
        series = rows[rows.partition_mb == partition].sort_values("workers")
        ax.plot(series.workers, series.speedup, marker="o", color=PARTITION_COLORS.get(partition),
                markeredgecolor=BAR_EDGE, label=f"{partition} MB partitions")
    ax.set_xscale("log")
    ax.minorticks_off()
    ticks = sorted(rows.workers.unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(t)) for t in ticks], rotation=45)
    ax.set_xlabel("Workers", fontsize=FONT_SIZE)
    ax.set_ylabel("Speedup of Arm B over Arm A (A time / B time)", fontsize=FONT_SIZE)
    ax.set_title("Direct communication against S3 by worker count", fontsize=TITLE_SIZE)
    _legend_below(ax, ncol=4)
    return _save(fig, output_dir, "exp2_speedup", chart_format, dpi)


def chart_agent_cost(experiments, arm, output_dir, chart_format, dpi):
    rows = experiments[experiments.series == "scaling"].sort_values(["workers", "partition_mb"])
    fig, ax = plt.subplots(figsize=(12, 6))
    x = list(range(len(rows)))
    width = 0.4
    for experiment, offset, label in (("exp1", -width / 2, "Experiment 1 (hand-built inputs)"),
                                      ("exp2", width / 2, "Experiment 2 (agent-planned inputs)")):
        ax.bar([i + offset for i in x], rows[f"{experiment}_mean_s"], width,
               yerr=rows[f"{experiment}_std_s"].fillna(0), capsize=CAPSIZE, color=EXPERIMENT_COLORS[experiment],
               edgecolor=BAR_EDGE, linewidth=BAR_LW, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(w)}\n{int(p)}MB" for w, p in zip(rows.workers, rows.partition_mb)], fontsize=8)
    ax.set_xlabel("Workers and partition size", fontsize=FONT_SIZE)
    ax.set_ylabel("Execution time (s)", fontsize=FONT_SIZE)
    ax.set_title(f"Execution time with and without the planning agent (Arm {arm})", fontsize=TITLE_SIZE)
    _legend_below(ax, ncol=2)
    return _save(fig, output_dir, f"exp1_vs_exp2_arm{arm}", chart_format, dpi)


def chart_planning(table, output_dir, chart_format, dpi):
    modes = [m for m in ("off", "exact", "structural") if m in set(table.cache_mode)]
    arms = sorted(table.arm.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    width = 0.8 / max(len(arms), 1)
    for ax, column, ylabel in ((axes[0][0], "llm_calls_total", "LLM calls"),
                               (axes[0][1], "wrong_plans", "Wrong plans")):
        for k, arm in enumerate(arms):
            values = [table[(table.arm == arm) & (table.cache_mode == m)][column].sum() for m in modes]
            ax.bar([i + (k - (len(arms) - 1) / 2) * width for i in range(len(modes))], values, width,
                   color=ARM_COLORS.get(arm), edgecolor=BAR_EDGE, linewidth=BAR_LW, label=f"Arm {arm}")
        ax.set_xticks(list(range(len(modes))))
        ax.set_xticklabels(modes)
        ax.set_xlabel("Plan cache mode", fontsize=FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    fig.suptitle("Planning the campaign: LLM calls and wrong plans by cache mode", fontsize=TITLE_SIZE)
    handles, labels = axes[0][0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=len(arms), anchor=(0.5, -0.1))
    return _save(fig, output_dir, "exp2_planning", chart_format, dpi)


def _write_csv(frame, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    frame.to_csv(os.path.join(output_dir, name), index=False)
    return name


def render_all(summary, published, plans_root, output_dir, chart_format, dpi):
    arms = compare_arms(summary)
    experiments = compare_experiments(summary, arm="A")
    table = planning_table(plans_root)
    files = [_write_csv(arms, output_dir, "exp2_arm_comparison.csv"),
             _write_csv(experiments, output_dir, "exp1_vs_exp2_armA.csv"),
             _write_csv(table, output_dir, "exp2_planning_metrics.csv")]
    if (arms.series == "scaling").any():
        files.append(chart_arms_vs_published(arms, published, output_dir, chart_format, dpi))
    if len(arms):
        files.append(chart_speedup(arms, output_dir, chart_format, dpi))
    if len(experiments):
        files.append(chart_agent_cost(experiments, "A", output_dir, chart_format, dpi))
    if len(table):
        files.append(chart_planning(table, output_dir, chart_format, dpi))
    return files


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--published", required=True)
    p.add_argument("--plans-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--format", default="png")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    for name in render_all(pd.read_csv(args.summary), pd.read_csv(args.published), args.plans_root,
                           args.output, args.format, args.dpi):
        print(os.path.join(args.output, name))


if __name__ == "__main__":
    main()