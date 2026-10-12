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
    BAR_EDGE, BAR_LW, FONT_SIZE, PARTITION_COLORS, PARTITION_MARKERS, SERIES_STYLES, TITLE_SIZE, _legend_below,
    _save, data_size_ticks, line_chart, published_seconds,
)

ARM_COLORS = {"A": "#1f77b4", "B": "#2ca02c"}
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


def _run_group_speedups(joined, runs):
    measured = runs[runs.phase == "measured"]
    arms = []
    for arm in ("A", "B"):
        prefix = arm.lower()
        picked = joined[CONFIGURATION + [f"{prefix}_experiment"]].merge(
            measured[measured.arm == arm], left_on=CONFIGURATION + [f"{prefix}_experiment"],
            right_on=CONFIGURATION + ["experiment"])
        arms.append(picked[CONFIGURATION + ["run", "duration_s"]].rename(columns={"duration_s": f"{prefix}_run_s"}))
    pairs = arms[0].merge(arms[1], on=CONFIGURATION + ["run"])
    pairs = pairs.assign(speedup=pairs.a_run_s / pairs.b_run_s)
    return (pairs.groupby(CONFIGURATION)["speedup"].agg(run_pairs="count", run_speedup_mean="mean",
                                                         run_speedup_std="std").reset_index())


def compare_arms(summary, experiment="exp2", runs=None):
    """Arm A against Arm B per configuration, from `experiment`; baselines, which Experiment 2
    does not plan, come from whichever experiment ran them.

    With `runs` (one row per validated execution), the speedup is also taken per run group, pairing
    run i of each arm, and reported as the mean and sample std across the run groups.
    """
    joined = _arm_rows(summary, "A", experiment).merge(_arm_rows(summary, "B", experiment), on=CONFIGURATION)
    joined = joined.assign(speedup=joined.a_mean_s / joined.b_mean_s, a_cv=joined.a_std_s / joined.a_mean_s,
                           b_cv=joined.b_std_s / joined.b_mean_s)
    if runs is not None:
        joined = joined.merge(_run_group_speedups(joined, runs), on=CONFIGURATION, how="left")
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


def chart_arm_scaling(joined, published, output_dir, chart_format, dpi):
    """One line chart per partition size: published, Arm A and Arm B execution time by data size."""
    rows = joined[joined.series == "scaling"].merge(_published_means(published), on=["partition_mb", "data_gb"],
                                                    how="left")
    files = []
    for partition in sorted(rows.partition_mb.unique()):
        part = rows[rows.partition_mb == partition].sort_values("data_gb")
        series = [(*SERIES_STYLES["published"], part.published_mean_s, part.published_std_s),
                  (*SERIES_STYLES["A"], part.a_mean_s, part.a_std_s),
                  (*SERIES_STYLES["B"], part.b_mean_s, part.b_std_s)]
        files.append(line_chart(series, data_size_ticks(part), "Dataset size in GB (workers)", "Execution time (s)",
                                f"S3 against direct communication, {partition} MB partitions",
                                output_dir, f"exp2_scaling_{partition}MB", chart_format, dpi))
    return files


def chart_speedup(joined, output_dir, chart_format, dpi):
    """Speedup of Arm B over Arm A by data size, one line per partition size: the mean over run groups
    with the sample std across them when per-run results are available, the ratio of means otherwise."""
    rows = joined[joined.series == "scaling"]
    per_run = "run_speedup_mean" in rows
    sizes = sorted(rows.data_gb.unique())
    series = []
    for partition in sorted(rows.partition_mb.unique()):
        part = rows[rows.partition_mb == partition].set_index("data_gb").reindex(sizes)
        means = part.run_speedup_mean if per_run else part.speedup
        series.append((f"{partition} MB partitions", PARTITION_COLORS.get(partition),
                       PARTITION_MARKERS.get(partition, "o"), means, part.run_speedup_std if per_run else None))
    return line_chart(series, [f"{s:g}" for s in sizes], "Dataset size (GB)",
                      "Speedup of Arm B over Arm A (A time / B time)",
                      "Direct communication against S3", output_dir, "exp2_speedup", chart_format, dpi)


def chart_agent_cost(experiments, arm, output_dir, chart_format, dpi):
    """One line chart per partition size: Arm `arm` with hand-built against agent-planned inputs."""
    rows = experiments[experiments.series == "scaling"]
    files = []
    for partition in sorted(rows.partition_mb.unique()):
        part = rows[rows.partition_mb == partition].sort_values("data_gb")
        series = [("Experiment 1 (hand-built inputs)", EXPERIMENT_COLORS["exp1"], "o", part.exp1_mean_s,
                   part.exp1_std_s),
                  ("Experiment 2 (agent-planned inputs)", EXPERIMENT_COLORS["exp2"], "^", part.exp2_mean_s,
                   part.exp2_std_s)]
        files.append(line_chart(series, data_size_ticks(part), "Dataset size in GB (workers)", "Execution time (s)",
                                f"With and without the planning agent (Arm {arm}), {partition} MB partitions",
                                output_dir, f"exp1_vs_exp2_arm{arm}_{partition}MB", chart_format, dpi))
    return files


PLANNING_CHARTS = (("llm_calls_total", "LLM calls", "exp2_planning_llm_calls"),
                   ("wrong_plans", "Wrong plans", "exp2_planning_wrong_plans"))


def chart_planning(table, output_dir, chart_format, dpi):
    """One bar chart per planning measure, by plan cache mode and arm."""
    modes = [m for m in ("off", "exact", "structural") if m in set(table.cache_mode)]
    arms = sorted(table.arm.unique())
    width = 0.8 / max(len(arms), 1)
    files = []
    for column, ylabel, name in PLANNING_CHARTS:
        fig, ax = plt.subplots(figsize=(10, 6))
        for k, arm in enumerate(arms):
            values = [table[(table.arm == arm) & (table.cache_mode == m)][column].sum() for m in modes]
            ax.bar([i + (k - (len(arms) - 1) / 2) * width for i in range(len(modes))], values, width,
                   color=ARM_COLORS.get(arm), edgecolor=BAR_EDGE, linewidth=BAR_LW, label=f"Arm {arm}")
        ax.set_xticks(list(range(len(modes))))
        ax.set_xticklabels(modes)
        ax.set_xlabel("Plan cache mode", fontsize=FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
        ax.set_title(f"Planning the campaign: {ylabel.lower()} by plan cache mode", fontsize=TITLE_SIZE)
        _legend_below(ax, ncol=len(arms), anchor=(0.5, -0.3))
        files.append(_save(fig, output_dir, name, chart_format, dpi))
    return files


PAIR_LATENCY_VARIANTS = {"accept_timeout_1s": ("Listener accept timeout 1 s (before the fix)", "#d62728"),
                         "accept_timeout_50ms": ("Listener accept timeout 50 ms (after the fix)", "#2ca02c")}


def pair_latency_summary(frame):
    """Per-variant TCPunch pair() latency from successful pairings, with the speedup over the 1 s timeout."""
    ok = frame[frame.pair_ms >= 0]
    summary = (ok.groupby("variant")["pair_ms"].agg(n="count", mean_ms="mean", std_ms="std", median_ms="median",
                                                    min_ms="min", max_ms="max").reset_index())
    baseline = summary.loc[summary.variant == "accept_timeout_1s", "mean_ms"]
    if len(baseline):
        summary["speedup_vs_1s"] = baseline.iloc[0] / summary.mean_ms
    return summary


def chart_pair_latency(frame, output_dir, chart_format, dpi):
    """Mean pair() time per listener accept timeout, as bars without error bars."""
    summary = pair_latency_summary(frame)
    variants = [v for v in PAIR_LATENCY_VARIANTS if v in set(summary.variant)]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, variant in enumerate(variants):
        label, color = PAIR_LATENCY_VARIANTS[variant]
        row = summary[summary.variant == variant].iloc[0]
        ax.bar(i, row.mean_ms, color=color, edgecolor=BAR_EDGE, linewidth=BAR_LW, label=label)
    ax.set_xticks(list(range(len(variants))))
    ax.set_xticklabels(["1 s" if v == "accept_timeout_1s" else "50 ms" for v in variants])
    ax.set_xlabel("TCPunch listener accept timeout", fontsize=FONT_SIZE)
    ax.set_ylabel("Mean pair() time (ms)", fontsize=FONT_SIZE)
    ax.set_title("TCPunch pairing latency (local loopback)", fontsize=TITLE_SIZE)
    _legend_below(ax, ncol=len(variants), anchor=(0.5, -0.3))
    return _save(fig, output_dir, "tcpunch_pair_latency", chart_format, dpi)


def _write_csv(frame, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    frame.to_csv(os.path.join(output_dir, name), index=False)
    return name


def render_all(summary, published, plans_root, output_dir, chart_format, dpi, pair_latency=None, runs=None):
    arms = compare_arms(summary, runs=runs)
    experiments = compare_experiments(summary, arm="A")
    table = planning_table(plans_root)
    files = [_write_csv(arms, output_dir, "exp2_arm_comparison.csv"),
             _write_csv(experiments, output_dir, "exp1_vs_exp2_armA.csv"),
             _write_csv(table, output_dir, "exp2_planning_metrics.csv")]
    if (arms.series == "scaling").any():
        files += chart_arm_scaling(arms, published, output_dir, chart_format, dpi)
        files.append(chart_speedup(arms, output_dir, chart_format, dpi))
    if (experiments.series == "scaling").any():
        files += chart_agent_cost(experiments, "A", output_dir, chart_format, dpi)
    if len(table):
        files += chart_planning(table, output_dir, chart_format, dpi)
    if pair_latency is not None and len(pair_latency):
        files += [_write_csv(pair_latency_summary(pair_latency), output_dir, "tcpunch_pair_latency_summary.csv"),
                  chart_pair_latency(pair_latency, output_dir, chart_format, dpi)]
    return files


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--published", required=True)
    p.add_argument("--plans-root", required=True)
    p.add_argument("--pair-latency", help="local TCPunch pair() latency CSV (variant, trial, side, pair_ms)")
    p.add_argument("--runs", help="per-run CSV from results.cosmic_lambda_results --runs-out; gives the speedup "
                                  "per run group with the sample std across run groups")
    p.add_argument("--output", required=True)
    p.add_argument("--format", default="png")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    pair_latency = pd.read_csv(args.pair_latency) if args.pair_latency else None
    runs = pd.read_csv(args.runs) if args.runs else None
    for name in render_all(pd.read_csv(args.summary), pd.read_csv(args.published), args.plans_root,
                           args.output, args.format, args.dpi, pair_latency=pair_latency, runs=runs):
        print(os.path.join(args.output, name))


if __name__ == "__main__":
    main()