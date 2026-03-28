"""
Notebook generator for cylon-armada experiment results.

Creates a Jupyter notebook with cells for loading aggregated CSV data
and generating each chart interactively. Follows the cylon
notebook_generator.py pattern.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def _make_cell(cell_type: str, source: str, **kwargs) -> dict:
    """Create a Jupyter notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": kwargs.get("metadata", {}),
        "source": source.split("\n") if isinstance(source, str) else source,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _cell_imports() -> str:
    return """import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.figsize': (10, 6),
})

PLATFORM_COLORS = {
    'lambda': '#FF9900', 'fargate': '#3F8624',
    'ecs': '#146EB4', 'rivanna': '#8B0000', 'local': '#555555',
}
PLATFORM_NAMES = {
    'lambda': 'AWS Lambda', 'fargate': 'AWS Fargate',
    'ecs': 'AWS ECS (GPU)', 'rivanna': 'Rivanna HPC', 'local': 'Local',
}
BACKEND_COLORS = {'NUMPY': '#1f77b4', 'PYCYLON': '#ff7f0e', 'CYTHON_BATCH': '#2ca02c'}
CONTEXT_BACKEND_COLORS = {'redis': '#d62728', 'cylon': '#9467bd', 'wasm': '#8c564b'}
"""


def _cell_load_data(csv_path: str) -> str:
    return f"""df = pd.read_csv('{csv_path}')
print(f"Loaded {{len(df)}} rows, {{df['experiment_name'].nunique()}} unique configs")
print(f"Platforms: {{df['platform'].unique()}}")
display(df.head(10))
"""


def _cell_cost_savings() -> str:
    return """reuse_df = df[df['baseline'] == False]
task_counts = sorted(reuse_df['task_count'].dropna().unique())
x = np.arange(len(task_counts))
width = 0.35

baseline_costs = [reuse_df[reuse_df['task_count'] == tc]['baseline_cost_mean'].mean() for tc in task_counts]
reuse_costs = [reuse_df[reuse_df['task_count'] == tc]['total_cost_mean'].mean() for tc in task_counts]
savings = [reuse_df[reuse_df['task_count'] == tc]['savings_pct_mean'].mean() for tc in task_counts]

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, baseline_costs, width, label='Baseline', color='#d62728', alpha=0.8)
bars2 = ax.bar(x + width/2, reuse_costs, width, label='With Reuse', color='#2ca02c', alpha=0.8)
for i, s in enumerate(savings):
    ax.annotate(f'{s:.1f}% saved', xy=(x[i], max(baseline_costs[i], reuse_costs[i])),
                xytext=(0, 8), textcoords='offset points', ha='center', fontweight='bold', color='#2ca02c')
ax.set_xlabel('Task Count'); ax.set_ylabel('Cost (USD)')
ax.set_title('Cost Savings: Context Reuse vs. Baseline')
ax.set_xticks(x); ax.set_xticklabels([str(int(t)) for t in task_counts]); ax.legend()
plt.tight_layout(); plt.savefig('cost_savings.svg', bbox_inches='tight'); plt.show()
"""


def _cell_reuse_rate() -> str:
    return """reuse_df = df[df['baseline'] == False]
thresholds = sorted(reuse_df['similarity_threshold'].dropna().unique())
backends = sorted(reuse_df['context_backend'].dropna().unique())
x = np.arange(len(thresholds))
width = 0.8 / max(len(backends), 1)

fig, ax = plt.subplots()
for i, cb in enumerate(backends):
    rates = [reuse_df[(reuse_df['similarity_threshold'] == th) & (reuse_df['context_backend'] == cb)]['reuse_rate_mean'].mean()
             for th in thresholds]
    errs = [reuse_df[(reuse_df['similarity_threshold'] == th) & (reuse_df['context_backend'] == cb)]['reuse_rate_std'].mean()
            for th in thresholds]
    ax.bar(x + i*width, rates, width, yerr=errs, label=cb,
           color=CONTEXT_BACKEND_COLORS.get(cb, f'C{i}'), alpha=0.8, capsize=3)
ax.set_xlabel('Similarity Threshold'); ax.set_ylabel('Reuse Rate (%)')
ax.set_title('Context Reuse Rate by Threshold and Backend')
ax.set_xticks(x + width*(len(backends)-1)/2)
ax.set_xticklabels([str(t) for t in thresholds]); ax.legend(); ax.set_ylim(0, 105)
plt.tight_layout(); plt.savefig('reuse_rate.svg', bbox_inches='tight'); plt.show()
"""


def _cell_latency_breakdown() -> str:
    return """reuse_df = df[df['baseline'] == False]
task_counts = sorted(reuse_df['task_count'].dropna().unique())
x = np.arange(len(task_counts))

search = [reuse_df[reuse_df['task_count'] == tc]['search_latency_ms_mean'].mean() for tc in task_counts]
llm = [reuse_df[reuse_df['task_count'] == tc]['llm_latency_ms_mean'].mean() for tc in task_counts]

fig, ax = plt.subplots()
ax.bar(x, search, label='Similarity Search', color='#1f77b4', alpha=0.8)
ax.bar(x, llm, bottom=search, label='LLM Invocation', color='#ff7f0e', alpha=0.8)
ax.set_xlabel('Task Count'); ax.set_ylabel('Latency (seconds)')
ax.set_title('Latency Breakdown: Search vs. LLM')
ax.set_xticks(x); ax.set_xticklabels([str(int(t)) for t in task_counts]); ax.legend()
plt.tight_layout(); plt.savefig('latency_breakdown.svg', bbox_inches='tight'); plt.show()
"""


def _cell_threshold_sensitivity() -> str:
    return """reuse_df = df[df['baseline'] == False]
thresholds = sorted(reuse_df['similarity_threshold'].dropna().unique())
rates = [reuse_df[reuse_df['similarity_threshold'] == th]['reuse_rate_mean'].mean() for th in thresholds]
savings = [reuse_df[reuse_df['similarity_threshold'] == th]['savings_pct_mean'].mean() for th in thresholds]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
l1 = ax1.plot(thresholds, rates, 'o-', color='#1f77b4', lw=2, ms=8, label='Reuse Rate (%)')
l2 = ax2.plot(thresholds, savings, 's--', color='#2ca02c', lw=2, ms=8, label='Cost Savings (%)')
ax1.set_xlabel('Similarity Threshold')
ax1.set_ylabel('Reuse Rate (%)', color='#1f77b4')
ax2.set_ylabel('Cost Savings (%)', color='#2ca02c')
ax1.set_title('Threshold Sensitivity Analysis')
lines = l1 + l2; ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
plt.tight_layout(); plt.savefig('threshold_sensitivity.svg', bbox_inches='tight'); plt.show()
"""


def _cell_infrastructure_comparison() -> str:
    return """reuse_df = df[df['baseline'] == False]
platforms = sorted(reuse_df['platform'].dropna().unique())
if len(platforms) >= 2:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    costs = [reuse_df[reuse_df['platform'] == p]['total_cost_mean'].mean() for p in platforms]
    latencies = [reuse_df[reuse_df['platform'] == p]['total_ms_mean'].mean() for p in platforms]
    colors = [PLATFORM_COLORS.get(p, 'gray') for p in platforms]
    names = [PLATFORM_NAMES.get(p, p) for p in platforms]
    ax1.bar(range(len(platforms)), costs, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(platforms))); ax1.set_xticklabels(names, rotation=15)
    ax1.set_ylabel('Cost (USD)'); ax1.set_title('Cost by Platform')
    ax2.bar(range(len(platforms)), latencies, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(platforms))); ax2.set_xticklabels(names, rotation=15)
    ax2.set_ylabel('Total Latency (s)'); ax2.set_title('Latency by Platform')
    fig.suptitle('Infrastructure Comparison', y=1.02); plt.tight_layout()
    plt.savefig('infrastructure_comparison.svg', bbox_inches='tight'); plt.show()
else:
    print(f'Only {len(platforms)} platform(s) - skipping infrastructure comparison')
"""


def _cell_summary_table() -> str:
    return """reuse_df = df[df['baseline'] == False]
cols = ['experiment_name', 'platform', 'task_count', 'similarity_threshold',
        'embedding_dimensions', 'backend', 'context_backend', 'num_runs',
        'total_cost_mean', 'total_cost_std', 'savings_pct_mean', 'savings_pct_std',
        'reuse_rate_mean', 'reuse_rate_std', 'total_ms_mean', 'total_ms_std']
available = [c for c in cols if c in reuse_df.columns]
display(reuse_df[available].sort_values('savings_pct_mean', ascending=False))
"""


def generate_notebook(
    aggregated_csv_path: str,
    output_path: str,
    output_chart_dir: str = ".",
) -> None:
    """Generate a Jupyter notebook with all chart cells."""
    cells = [
        _make_cell("markdown", "# Cylon-Armada: Context Reuse Experiment Results\n\nGenerated by the results pipeline."),
        _make_cell("code", _cell_imports()),
        _make_cell("code", _cell_load_data(aggregated_csv_path)),
        _make_cell("markdown", "## Cost Savings"),
        _make_cell("code", _cell_cost_savings()),
        _make_cell("markdown", "## Reuse Rate"),
        _make_cell("code", _cell_reuse_rate()),
        _make_cell("markdown", "## Latency Breakdown"),
        _make_cell("code", _cell_latency_breakdown()),
        _make_cell("markdown", "## Threshold Sensitivity"),
        _make_cell("code", _cell_threshold_sensitivity()),
        _make_cell("markdown", "## Infrastructure Comparison"),
        _make_cell("code", _cell_infrastructure_comparison()),
        _make_cell("markdown", "## Summary Table"),
        _make_cell("code", _cell_summary_table()),
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=2)

    logger.info("Notebook saved: %s", output_path)