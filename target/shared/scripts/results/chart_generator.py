"""
Chart generator for cylon-armada experiment results.

Generates publication-quality charts from aggregated CSV data.
Follows the cylon chart_generator.py pattern — each chart function
takes a DataFrame and PipelineConfig, produces SVG/PNG output.

Charts specific to context-reuse experiments:
  - Cost savings (reuse vs. baseline)
  - Reuse rate across configurations
  - Latency breakdown (search, LLM, total)
  - Scaling: cost/latency vs. task count
  - Infrastructure comparison across platforms
  - Threshold sensitivity analysis
"""

import logging
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Publication-standard styling
FONT_SIZE = 12
TITLE_SIZE = 14
TICK_SIZE = 10
LEGEND_SIZE = 10

PLATFORM_COLORS = {
    "lambda": "#FF9900",
    "fargate": "#3F8624",
    "ecs": "#146EB4",
    "rivanna": "#8B0000",
    "local": "#555555",
}

PLATFORM_MARKERS = {
    "lambda": "o",
    "fargate": "s",
    "ecs": "D",
    "rivanna": "^",
    "local": "v",
}

PLATFORM_NAMES = {
    "lambda": "AWS Lambda",
    "fargate": "AWS Fargate",
    "ecs": "AWS ECS (GPU)",
    "rivanna": "Rivanna HPC",
    "local": "Local",
}

BACKEND_COLORS = {
    "NUMPY": "#1f77b4",
    "PYCYLON": "#ff7f0e",
    "CYTHON_BATCH": "#2ca02c",
}

CONTEXT_BACKEND_COLORS = {
    "redis": "#d62728",
    "cylon": "#9467bd",
    "wasm": "#8c564b",
}


def _save_chart(fig, config, name: str):
    """Save chart to output directory."""
    path = os.path.join(config.output_dir, f"{name}.{config.chart_format}")
    fig.savefig(path, dpi=config.chart_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: %s", path)


def _platform_name(platform: str) -> str:
    return PLATFORM_NAMES.get(platform, platform.upper())


# ---------------------------------------------------------------------------
# Chart: Cost Savings — Reuse vs. Baseline
# ---------------------------------------------------------------------------

def _cylon_reuse(df: pd.DataFrame) -> pd.DataFrame:
    """Return cylon context-reuse rows only (excludes baseline and LlamaIndex)."""
    mask = (df["baseline"] == False)
    if "system" in df.columns:
        mask = mask & (df["system"].fillna("cylon") == "cylon")
    return df[mask].copy()


def chart_cost_savings(df: pd.DataFrame, config) -> None:
    """Bar chart comparing total cost (reuse) vs. baseline cost.

    Groups by task_count, shows savings_pct as annotation.
    """
    reuse_df = _cylon_reuse(df)
    if reuse_df.empty:
        logger.warning("No reuse data for cost savings chart")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    task_counts = sorted(reuse_df["task_count"].dropna().unique())
    x = np.arange(len(task_counts))
    width = 0.35

    baseline_costs = []
    reuse_costs = []
    savings = []

    for tc in task_counts:
        tc_data = reuse_df[reuse_df["task_count"] == tc]
        baseline_costs.append(tc_data["baseline_cost_mean"].mean())
        reuse_costs.append(tc_data["total_cost_mean"].mean())
        savings.append(tc_data["savings_pct_mean"].mean())

    bars1 = ax.bar(x - width / 2, baseline_costs, width, label="Baseline (no reuse)",
                   color="#d62728", alpha=0.8)
    bars2 = ax.bar(x + width / 2, reuse_costs, width, label="With context reuse",
                   color="#2ca02c", alpha=0.8)

    # Annotate savings percentage
    for i, (b1, b2, s) in enumerate(zip(bars1, bars2, savings)):
        ax.annotate(
            f"{s:.1f}%\nsaved",
            xy=(x[i], max(b1.get_height(), b2.get_height())),
            xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=TICK_SIZE, color="#2ca02c", fontweight="bold",
        )

    ax.set_xlabel("Task Count", fontsize=FONT_SIZE)
    ax.set_ylabel("Cost (USD)", fontsize=FONT_SIZE)
    ax.set_title("Cost Savings: Context Reuse vs. Baseline", fontsize=TITLE_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in task_counts], fontsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)

    _save_chart(fig, config, "cost_savings")


# ---------------------------------------------------------------------------
# Chart: Reuse Rate across Configurations
# ---------------------------------------------------------------------------

def chart_reuse_rate(df: pd.DataFrame, config) -> None:
    """Grouped bar chart of reuse rate by threshold and context backend."""
    reuse_df = _cylon_reuse(df)
    if reuse_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    thresholds = sorted(reuse_df["similarity_threshold"].dropna().unique())
    backends = sorted(reuse_df["context_backend"].dropna().unique())
    x = np.arange(len(thresholds))
    width = 0.8 / max(len(backends), 1)

    for i, cb in enumerate(backends):
        rates = []
        errs = []
        for th in thresholds:
            mask = (reuse_df["similarity_threshold"] == th) & (reuse_df["context_backend"] == cb)
            subset = reuse_df[mask]
            rates.append(subset["reuse_rate_mean"].mean() if not subset.empty else 0)
            errs.append(subset["reuse_rate_std"].mean() if not subset.empty else 0)

        color = CONTEXT_BACKEND_COLORS.get(cb, f"C{i}")
        ax.bar(x + i * width, rates, width, yerr=errs, label=cb,
               color=color, alpha=0.8, capsize=3)

    ax.set_xlabel("Similarity Threshold", fontsize=FONT_SIZE)
    ax.set_ylabel("Reuse Rate (%)", fontsize=FONT_SIZE)
    ax.set_title("Context Reuse Rate by Threshold and Backend", fontsize=TITLE_SIZE)
    ax.set_xticks(x + width * (len(backends) - 1) / 2)
    ax.set_xticklabels([str(t) for t in thresholds], fontsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_ylim(0, 105)

    _save_chart(fig, config, "reuse_rate")


# ---------------------------------------------------------------------------
# Chart: Latency Breakdown
# ---------------------------------------------------------------------------

def chart_latency_breakdown(df: pd.DataFrame, config) -> None:
    """Stacked bar chart of latency breakdown: search vs. LLM."""
    reuse_df = _cylon_reuse(df)
    if reuse_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    task_counts = sorted(reuse_df["task_count"].dropna().unique())
    x = np.arange(len(task_counts))

    search_times = []
    llm_times = []

    for tc in task_counts:
        tc_data = reuse_df[reuse_df["task_count"] == tc]
        search_times.append(tc_data["search_latency_ms_mean"].mean() if "search_latency_ms_mean" in tc_data else 0)
        llm_times.append(tc_data["llm_latency_ms_mean"].mean() if "llm_latency_ms_mean" in tc_data else 0)

    ax.bar(x, search_times, label="Similarity Search", color="#1f77b4", alpha=0.8)
    ax.bar(x, llm_times, bottom=search_times, label="LLM Invocation", color="#ff7f0e", alpha=0.8)

    ax.set_xlabel("Task Count", fontsize=FONT_SIZE)
    ax.set_ylabel("Latency (seconds)", fontsize=FONT_SIZE)
    ax.set_title("Latency Breakdown: Search vs. LLM", fontsize=TITLE_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in task_counts], fontsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)

    _save_chart(fig, config, "latency_breakdown")


# ---------------------------------------------------------------------------
# Chart: Scaling — Cost vs. Task Count
# ---------------------------------------------------------------------------

def chart_cost_scaling(df: pd.DataFrame, config) -> None:
    """Line chart of cost vs. task count, one line per platform."""
    reuse_df = _cylon_reuse(df)
    if reuse_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    platforms = sorted(reuse_df["platform"].dropna().unique())

    for platform in platforms:
        pdata = reuse_df[reuse_df["platform"] == platform]
        task_counts = sorted(pdata["task_count"].dropna().unique())

        costs = []
        errs = []
        for tc in task_counts:
            tc_data = pdata[pdata["task_count"] == tc]
            costs.append(tc_data["total_cost_mean"].mean())
            errs.append(tc_data["total_cost_std"].mean())

        color = PLATFORM_COLORS.get(platform, "gray")
        marker = PLATFORM_MARKERS.get(platform, "o")
        ax.errorbar(
            task_counts, costs, yerr=errs,
            label=_platform_name(platform),
            color=color, marker=marker, linewidth=2, capsize=4,
        )

    ax.set_xlabel("Task Count", fontsize=FONT_SIZE)
    ax.set_ylabel("Total Cost (USD)", fontsize=FONT_SIZE)
    ax.set_title("Cost Scaling Across Platforms", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    _save_chart(fig, config, "cost_scaling")


# ---------------------------------------------------------------------------
# Chart: Infrastructure Comparison
# ---------------------------------------------------------------------------

def chart_infrastructure_comparison(df: pd.DataFrame, config) -> None:
    """Grouped bar chart comparing platforms on cost and latency."""
    reuse_df = _cylon_reuse(df)
    if reuse_df.empty:
        return

    platforms = sorted(reuse_df["platform"].dropna().unique())
    if len(platforms) < 2:
        logger.info("Skipping infrastructure comparison (only %d platform)", len(platforms))
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Cost comparison
    costs = []
    cost_errs = []
    for p in platforms:
        pdata = reuse_df[reuse_df["platform"] == p]
        costs.append(pdata["total_cost_mean"].mean())
        cost_errs.append(pdata["total_cost_std"].mean())

    colors = [PLATFORM_COLORS.get(p, "gray") for p in platforms]
    ax1.bar(range(len(platforms)), costs, yerr=cost_errs,
            color=colors, alpha=0.8, capsize=4)
    ax1.set_xticks(range(len(platforms)))
    ax1.set_xticklabels([_platform_name(p) for p in platforms],
                        fontsize=TICK_SIZE, rotation=15)
    ax1.set_ylabel("Cost (USD)", fontsize=FONT_SIZE)
    ax1.set_title("Cost by Platform", fontsize=TITLE_SIZE)
    ax1.tick_params(axis="y", labelsize=TICK_SIZE)

    # Latency comparison
    latencies = []
    lat_errs = []
    for p in platforms:
        pdata = reuse_df[reuse_df["platform"] == p]
        latencies.append(pdata["total_ms_mean"].mean())
        lat_errs.append(pdata["total_ms_std"].mean())

    ax2.bar(range(len(platforms)), latencies, yerr=lat_errs,
            color=colors, alpha=0.8, capsize=4)
    ax2.set_xticks(range(len(platforms)))
    ax2.set_xticklabels([_platform_name(p) for p in platforms],
                        fontsize=TICK_SIZE, rotation=15)
    ax2.set_ylabel("Total Latency (seconds)", fontsize=FONT_SIZE)
    ax2.set_title("Latency by Platform", fontsize=TITLE_SIZE)
    ax2.tick_params(axis="y", labelsize=TICK_SIZE)

    fig.suptitle("Infrastructure Comparison", fontsize=TITLE_SIZE, y=1.02)
    fig.tight_layout()
    _save_chart(fig, config, "infrastructure_comparison")


# ---------------------------------------------------------------------------
# Chart: Threshold Sensitivity
# ---------------------------------------------------------------------------

def chart_threshold_sensitivity(df: pd.DataFrame, config) -> None:
    """Line chart: reuse_rate and savings_pct vs. similarity_threshold."""
    reuse_df = _cylon_reuse(df)
    if reuse_df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    thresholds = sorted(reuse_df["similarity_threshold"].dropna().unique())

    reuse_rates = []
    savings = []
    for th in thresholds:
        th_data = reuse_df[reuse_df["similarity_threshold"] == th]
        reuse_rates.append(th_data["reuse_rate_mean"].mean())
        savings.append(th_data["savings_pct_mean"].mean())

    line1 = ax1.plot(thresholds, reuse_rates, "o-", color="#1f77b4",
                      linewidth=2, markersize=8, label="Reuse Rate (%)")
    line2 = ax2.plot(thresholds, savings, "s--", color="#2ca02c",
                      linewidth=2, markersize=8, label="Cost Savings (%)")

    ax1.set_xlabel("Similarity Threshold", fontsize=FONT_SIZE)
    ax1.set_ylabel("Reuse Rate (%)", fontsize=FONT_SIZE, color="#1f77b4")
    ax2.set_ylabel("Cost Savings (%)", fontsize=FONT_SIZE, color="#2ca02c")
    ax1.set_title("Threshold Sensitivity Analysis", fontsize=TITLE_SIZE)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=LEGEND_SIZE, loc="upper right")

    ax1.tick_params(axis="both", labelsize=TICK_SIZE)
    ax2.tick_params(axis="y", labelsize=TICK_SIZE)

    _save_chart(fig, config, "threshold_sensitivity")


# ---------------------------------------------------------------------------
# Chart: SIMD Backend Comparison
# ---------------------------------------------------------------------------

def chart_simd_comparison(df: pd.DataFrame, config) -> None:
    """Bar chart comparing search latency across SIMD backends."""
    reuse_df = _cylon_reuse(df)
    if reuse_df.empty:
        return

    backends = sorted(reuse_df["backend"].dropna().unique())
    if len(backends) < 2:
        logger.info("Skipping SIMD comparison (only %d backend)", len(backends))
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    latencies = []
    errs = []
    colors = []
    for be in backends:
        be_data = reuse_df[reuse_df["backend"] == be]
        latencies.append(be_data["search_latency_ms_mean"].mean())
        errs.append(be_data["search_latency_ms_std"].mean())
        colors.append(BACKEND_COLORS.get(be, "gray"))

    ax.bar(range(len(backends)), latencies, yerr=errs,
           color=colors, alpha=0.8, capsize=4)
    ax.set_xticks(range(len(backends)))
    ax.set_xticklabels(backends, fontsize=TICK_SIZE)
    ax.set_ylabel("Search Latency (seconds)", fontsize=FONT_SIZE)
    ax.set_title("Similarity Search Latency by SIMD Backend", fontsize=TITLE_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)

    _save_chart(fig, config, "simd_comparison")


# ---------------------------------------------------------------------------
# Chart: Embedding Dimension Impact
# ---------------------------------------------------------------------------

def chart_dimension_impact(df: pd.DataFrame, config) -> None:
    """Line chart: search latency and reuse rate vs. embedding dimensions."""
    reuse_df = _cylon_reuse(df)
    if reuse_df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    dimensions = sorted(reuse_df["embedding_dimensions"].dropna().unique())

    latencies = []
    rates = []
    for dim in dimensions:
        dim_data = reuse_df[reuse_df["embedding_dimensions"] == dim]
        latencies.append(dim_data["search_latency_ms_mean"].mean())
        rates.append(dim_data["reuse_rate_mean"].mean())

    line1 = ax1.plot(dimensions, latencies, "o-", color="#d62728",
                      linewidth=2, markersize=8, label="Search Latency (s)")
    line2 = ax2.plot(dimensions, rates, "s--", color="#1f77b4",
                      linewidth=2, markersize=8, label="Reuse Rate (%)")

    ax1.set_xlabel("Embedding Dimensions", fontsize=FONT_SIZE)
    ax1.set_ylabel("Search Latency (seconds)", fontsize=FONT_SIZE, color="#d62728")
    ax2.set_ylabel("Reuse Rate (%)", fontsize=FONT_SIZE, color="#1f77b4")
    ax1.set_title("Impact of Embedding Dimensions", fontsize=TITLE_SIZE)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=LEGEND_SIZE)

    ax1.tick_params(axis="both", labelsize=TICK_SIZE)
    ax2.tick_params(axis="y", labelsize=TICK_SIZE)

    _save_chart(fig, config, "dimension_impact")


# ---------------------------------------------------------------------------
# Chart: LlamaIndex Baseline Comparison
# ---------------------------------------------------------------------------

def chart_llamaindex_comparison(df: pd.DataFrame, config) -> None:
    """Bar chart comparing cylon context-reuse vs. LlamaIndex RAG baseline.

    Shows total cost side-by-side per task count.
    Only generated when LlamaIndex rows are present in the data.
    """
    if "system" not in df.columns:
        return

    cylon_df = df[(df["baseline"] == False) & (df["system"].fillna("cylon") == "cylon")].copy()
    llama_df = df[df["system"] == "llamaindex"].copy()

    if llama_df.empty:
        logger.info("Skipping LlamaIndex comparison chart (no llamaindex rows)")
        return

    task_counts = sorted(
        set(cylon_df["task_count"].dropna().unique()) |
        set(llama_df["task_count"].dropna().unique())
    )
    x = np.arange(len(task_counts))
    width = 0.25

    cylon_costs, cylon_errs = [], []
    llama_costs, llama_errs = [], []
    baseline_costs, baseline_errs = [], []

    baseline_df = df[df["baseline"] == True].copy() if "baseline" in df.columns else pd.DataFrame()

    for tc in task_counts:
        c = cylon_df[cylon_df["task_count"] == tc]
        cylon_costs.append(c["total_cost_mean"].mean() if not c.empty else 0)
        cylon_errs.append(c["total_cost_std"].mean() if not c.empty else 0)

        l = llama_df[llama_df["task_count"] == tc]
        llama_costs.append(l["total_cost_mean"].mean() if not l.empty else 0)
        llama_errs.append(l["total_cost_std"].mean() if not l.empty else 0)

        if not baseline_df.empty:
            b = baseline_df[baseline_df["task_count"] == tc]
            baseline_costs.append(b["total_cost_mean"].mean() if not b.empty else 0)
            baseline_errs.append(b["total_cost_std"].mean() if not b.empty else 0)

    fig, ax = plt.subplots(figsize=(11, 6))

    if baseline_costs:
        ax.bar(x - width, baseline_costs, width, yerr=baseline_errs,
               label="Cylon baseline (no reuse)", color="#d62728", alpha=0.8, capsize=3)

    ax.bar(x, cylon_costs, width, yerr=cylon_errs,
           label="Cylon (context reuse)", color="#2ca02c", alpha=0.8, capsize=3)
    ax.bar(x + width, llama_costs, width, yerr=llama_errs,
           label="LlamaIndex RAG (always LLM)", color="#ff7f0e", alpha=0.8, capsize=3)

    ax.set_xlabel("Task Count", fontsize=FONT_SIZE)
    ax.set_ylabel("Total Cost (USD)", fontsize=FONT_SIZE)
    ax.set_title("Cost Comparison: Cylon Context Reuse vs. LlamaIndex RAG Baseline",
                 fontsize=TITLE_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in task_counts], fontsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)

    _save_chart(fig, config, "llamaindex_comparison")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all_charts(
    df: pd.DataFrame,
    config,
    micro_df: Optional[pd.DataFrame] = None,
) -> None:
    """Generate all charts from aggregated results."""
    os.makedirs(config.output_dir, exist_ok=True)

    chart_cost_savings(df, config)
    chart_reuse_rate(df, config)
    chart_latency_breakdown(df, config)
    chart_cost_scaling(df, config)
    chart_infrastructure_comparison(df, config)
    chart_threshold_sensitivity(df, config)
    chart_simd_comparison(df, config)
    chart_dimension_impact(df, config)
    chart_llamaindex_comparison(df, config)

    logger.info("All charts generated in %s", config.output_dir)