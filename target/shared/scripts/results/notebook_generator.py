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
        # nbformat joins the source list with "" (no separators), so each element
        # must keep its trailing newline. keepends=True preserves them; a plain
        # split("\n") would strip them and run all lines together.
        "source": source.splitlines(keepends=True) if isinstance(source, str) else source,
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


# ---------------------------------------------------------------------------
# Experiment A / A2 — zero-copy data plane notebook
#
# The zero-copy metric family (throughput by serialization format, memory
# copies) does not fit the cost/reuse cells above, so it has its own cells and
# generator, mirroring how chart_zerocopy.py parallels chart_generator.py. Cells
# reuse _make_cell and the same style/structure.
# ---------------------------------------------------------------------------

def _zc_cell_imports() -> str:
    # Inline backend (not Agg): this notebook is meant to be opened and tweaked
    # in Jupyter, so figures must render inline. The cells also savefig() to
    # charts/, so files are refreshed either way. Under headless nbconvert the
    # ipython kernel defaults to the inline backend, so this works there too.
    return """%matplotlib inline
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    # Match the quals scaling notebook: keep the box frame, no gridlines,
    # white background, black-edged bars, legend in a bordered box below.
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'axes.grid': False,
})
BAR_EDGE, BAR_LW = 'black', 0.8

# Semantic colors: proposed system green, naive text red, zero-copy binary orange.
FORMAT_COLORS = {
    'arrow_ipc': '#2ca02c', 'contexttable_ipc': '#17becf', 'flatbuffers': '#ff7f0e',
    'pickle': '#7f7f7f', 'protobuf': '#8c564b', 'base64_tobytes': '#9467bd', 'json': '#d62728',
}
# Display order (zero-copy formats grouped first). Reorder to taste.
FORMAT_ORDER = ['arrow_ipc', 'contexttable_ipc', 'flatbuffers', 'pickle',
                'base64_tobytes', 'protobuf', 'json']
"""


def _zc_cell_load_data(results_csv: str, a2_csv: str) -> str:
    return f"""df = pd.read_csv('{results_csv}')
a2 = pd.read_csv('{a2_csv}')
# Representative (largest) cell for the bar charts.
N_REP = int(df['n'].max())
D_REP = int(df[df['n'] == N_REP]['d'].max())
cell = df[(df['n'] == N_REP) & (df['d'] == D_REP)].set_index('format')
fmts = [f for f in FORMAT_ORDER if f in cell.index]
print(f"formats: {{list(df['format'].unique())}}")
print(f"representative cell: N={{N_REP}}, D={{D_REP}}")
display(df.head(10))
"""


def _zc_cell_throughput_by_format() -> str:
    return """vals = [cell.loc[f, 'throughput_roundtrip_MBps'] for f in fmts]
sp = [cell.loc[f, 'speedup_vs_json'] for f in fmts]
colors = [FORMAT_COLORS.get(f, '#333') for f in fmts]

fig, ax = plt.subplots()
y = range(len(fmts))
# Bar chart is a representative-cell snapshot — no error bars (per convention).
ax.barh(list(y), vals, color=colors, alpha=0.9, edgecolor=BAR_EDGE, linewidth=BAR_LW)
ax.set_yticks(list(y)); ax.set_yticklabels(fmts); ax.invert_yaxis()
ax.set_xscale('log'); ax.minorticks_off()
ax.set_xlabel('Round-trip throughput (MB/s, log scale)')
ax.set_title(f'Zero-copy data plane throughput by format  (N={N_REP}, D={D_REP})')
for i, (v, s) in enumerate(zip(vals, sp)):
    lbl = f'{v:,.0f} MB/s' + (f'  ({s:,.0f}x vs JSON)' if pd.notna(s) else '')
    ax.text(v * 1.05, i, lbl, va='center', fontsize=9)
ax.set_xlim(right=max(vals) * 12)
plt.tight_layout(); plt.savefig('charts/exp_a_throughput_by_format.svg', bbox_inches='tight'); plt.show()
"""


def _zc_cell_memory_copies() -> str:
    return """peaks = [max(cell.loc[f, 'deserialize_peak_kb'], 0.1) for f in fmts]
copies = [cell.loc[f, 'memory_copies'] for f in fmts]
colors = [FORMAT_COLORS.get(f, '#333') for f in fmts]
payload_kb = N_REP * D_REP * 4 / 1024

fig, ax = plt.subplots()
x = range(len(fmts))
ax.bar(list(x), peaks, color=colors, alpha=0.9, edgecolor=BAR_EDGE, linewidth=BAR_LW); ax.set_yscale('log'); ax.minorticks_off()
ax.set_xticks(list(x)); ax.set_xticklabels(fmts, rotation=30, ha='right')
ax.set_ylabel('Deserialize peak allocation (KB, log scale)')
ax.set_title(f'Memory copies on read  (N={N_REP}, D={D_REP}; payload = {payload_kb:,.0f} KB)')
for i, (p, c) in enumerate(zip(peaks, copies)):
    ax.text(i, p * 1.3, f"{int(c)} cop{'y' if int(c) == 1 else 'ies'}", ha='center', fontsize=9)
plt.tight_layout(); plt.savefig('charts/exp_a_memory_copies.svg', bbox_inches='tight'); plt.show()
"""


def _zc_cell_throughput_scaling() -> str:
    return """dims = sorted(df['d'].unique())
present = [f for f in FORMAT_ORDER if f in set(df['format'])]
STD = 'throughput_roundtrip_MBps_std'
fig, ax = plt.subplots()
for f in present:
    ys, es = [], []
    for dd in dims:
        cand = df[(df['format'] == f) & (df['d'] == dd)]
        if len(cand):
            top = cand.sort_values('n').iloc[-1]
            ys.append(top['throughput_roundtrip_MBps'])
            es.append(top[STD] if STD in df.columns else 0)
        else:
            ys.append(None); es.append(0)
    color = FORMAT_COLORS.get(f, '#333')
    yerr = es if any(e > 0 for e in es) else None
    ax.errorbar(dims, ys, yerr=yerr, marker='o', label=f, color=color, ecolor=color, lw=2, capsize=5)
ax.set_yscale('log'); ax.minorticks_off(); ax.set_xticks(dims)
ax.set_xlabel('Embedding dimension D'); ax.set_ylabel('Round-trip throughput (MB/s, log scale)')
ax.set_title('Throughput vs embedding dimension')
ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.32), frameon=True)
plt.tight_layout(); plt.savefig('charts/exp_a_throughput_scaling.svg', bbox_inches='tight'); plt.show()
"""


def _zc_cell_schema_compat() -> str:
    # A2 results table: every operator-DAG edge + the incompatible / evolution
    # test cases, with the compatibility case and the compiler's result.
    return """NAVY = '#1E2761'
RCOLOR = {'zero_copy': '#2ca02c', 'tolerated': '#2ca02c', 'convert': '#ff7f0e', 'rejected': '#d62728'}
RLABEL = {'zero_copy': 'zero-copy', 'convert': 'convert', 'rejected': 'rejected', 'tolerated': 'tolerated'}
def _abbrev(s):
    s = str(s or '').strip(); nn = s.endswith('+nullable')
    if nn: s = s[:-9].strip()
    if s.startswith('struct'): o = 'struct'
    elif 'fixed_size_list' in s:
        d = s.split('[')[-1].rstrip(']') if '[' in s else ''
        o = 'FixedSizeList[' + d + ']' if d else 'FixedSizeList'
    elif s.startswith('large_'): o = 'LargeUtf8'
    elif s.startswith('bool'): o = 'bool'
    else: o = s
    return o + ' +null' if nn else o
ct, rc = [], []
for _, r in a2.iterrows():
    res = str(r.get('result', ''))
    ct.append([r['edge'], _abbrev(r['schema_out']) + '  ->  ' + _abbrev(r['schema_in']),
               str(r.get('case', '')), RLABEL.get(res, res)])
    rc.append(RCOLOR.get(res, '#999'))
n = len(ct)
fig, ax = plt.subplots(figsize=(11, 0.62 * n + 1.5)); ax.axis('off')
tb = ax.table(cellText=ct, colLabels=['Operator edge / test', 'schema_out  ->  schema_in', 'Case', 'Result'],
              loc='center', cellLoc='left', colWidths=[0.24, 0.48, 0.13, 0.15])
tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.9)
for j in range(4):
    tb[0, j].set_facecolor(NAVY); tb[0, j].set_text_props(color='white', fontweight='bold')
for i, c in enumerate(rc):
    tb[i + 1, 3].set_facecolor(c); tb[i + 1, 3].set_alpha(0.30); tb[i + 1, 3].set_text_props(fontweight='bold')
ax.set_title('A2: Arrow schema compatibility results  (AgentDAGCompiler enforcement)', fontsize=14, color=NAVY, pad=16)
plt.savefig('charts/exp_a2_schema_compat.svg', bbox_inches='tight'); plt.show()
"""


def _zc_cell_summary_table() -> str:
    return """cols = ['format', 'n', 'd', 'wire_ratio', 'roundtrip_ms',
        'throughput_roundtrip_MBps', 'memory_copies', 'deserialize_peak_kb', 'speedup_vs_json']
available = [c for c in cols if c in df.columns]
display(df[available].sort_values(['d', 'throughput_roundtrip_MBps'], ascending=[True, False]))
"""


def generate_zerocopy_notebook(
    results_csv: str,
    a2_csv: str,
    output_path: str,
    output_chart_dir: str = "charts",
) -> None:
    """Generate a tweakable Jupyter notebook for the Experiment A / A2 charts.

    Cells contain self-contained plotting code (no dependency on chart_zerocopy),
    so charts can be adjusted by hand. Run the notebook from the results directory
    (e.g. results/exp_a_zerocopy) so the relative CSV and chart paths resolve.
    """
    cells = [
        _make_cell("markdown",
                   "# Experiment A / A2: Zero-Copy Data Plane Charts\n\n"
                   "Tweakable notebook. Each chart's plotting code is inline and self-contained "
                   "— edit colors, labels, scales, and figure sizes here, then re-run the cell. "
                   "Run this notebook from its own directory so the relative paths resolve."),
        _make_cell("code", _zc_cell_imports()),
        _make_cell("code", _zc_cell_load_data(results_csv, a2_csv)),
        _make_cell("markdown", "## Throughput by format (log scale)"),
        _make_cell("code", _zc_cell_throughput_by_format()),
        _make_cell("markdown", "## Memory copies on read"),
        _make_cell("code", _zc_cell_memory_copies()),
        _make_cell("markdown", "## Throughput vs embedding dimension"),
        _make_cell("code", _zc_cell_throughput_scaling()),
        _make_cell("markdown", "## A2 schema compatibility matrix"),
        _make_cell("code", _zc_cell_schema_compat()),
        _make_cell("markdown", "## Summary table"),
        _make_cell("code", _zc_cell_summary_table()),
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(output_path) or ".", output_chart_dir), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=2)

    logger.info("Zero-copy notebook saved: %s", output_path)


def _cmp_cell_imports() -> str:
    return """%matplotlib inline
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.figsize': (11, 7),
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'axes.grid': False,
})
BAR_EDGE, BAR_LW = 'black', 0.8

COLLECTIVE_COLORS = {
    'scatter': '#2ca02c', 'scatterv': '#17becf', 'gather': '#1f77b4', 'allgather': '#9467bd',
    'reduce': '#ff7f0e', 'allreduce': '#d62728', 'broadcast': '#8c564b', 'barrier': '#7f7f7f',
}
COLLECTIVE_ORDER = ['scatter', 'scatterv', 'gather', 'allgather', 'reduce', 'allreduce', 'broadcast', 'barrier']

PLATFORM_STYLE = {
    'rivanna': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'label': 'Rivanna (UCC)'},
    'fargate': {'color': '#d62728', 'linestyle': '--', 'marker': 's', 'label': 'Fargate (direct-redis)'},
}
PLATFORM_ORDER = ['rivanna', 'fargate']
"""


def _cmp_cell_load_data(results_csv: str) -> str:
    return f"""df = pd.read_csv('{results_csv}')
df = df[df['unsupported'] != True]
print(f"platforms: {{list(df['platform'].unique())}}")
print(f"collectives: {{list(df['collective'].unique())}}")
print(f"world sizes: {{sorted(df['N'].unique())}}")
display(df.head(10))
"""


def _cmp_cell_latency_vs_n() -> str:
    return """fig, ax = plt.subplots()
present = [c for c in COLLECTIVE_ORDER if c in set(df['collective'])]
for coll in present:
    color = COLLECTIVE_COLORS.get(coll, '#333')
    for plat in PLATFORM_ORDER:
        sub = df[(df['collective'] == coll) & (df['platform'] == plat)]
        if sub.empty:
            continue
        rep_size = sub['msg_size'].max()
        pts = sub[sub['msg_size'] == rep_size].sort_values('N')
        if pts.empty:
            continue
        style = PLATFORM_STYLE[plat]
        yerr = pts['latency_p50_ms_std'] if 'latency_p50_ms_std' in pts.columns else None
        ax.errorbar(pts['N'], pts['latency_p50_ms'], yerr=yerr, marker=style['marker'],
                    linestyle=style['linestyle'], color=color, lw=2, capsize=5,
                    label=f"{coll} — {style['label']}")
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.minorticks_off()
ax.set_xlabel('World size N (log2)'); ax.set_ylabel('P50 latency (ms, log scale)')
ax.set_title('Collective latency vs N — Rivanna (UCC) vs Fargate (direct-redis)')
ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.38), frameon=True, fontsize=9)
plt.tight_layout(); plt.savefig('charts/collectives_latency_vs_N_rivanna_vs_fargate.svg', bbox_inches='tight'); plt.show()
"""


def _cmp_cell_throughput_vs_msgsize() -> str:
    return """size_swept = [c for c in COLLECTIVE_ORDER if c in set(df['collective']) and c not in ('reduce', 'allreduce', 'barrier')]
n_common = min(df[df['platform'] == p]['N'].max() for p in PLATFORM_ORDER if (df['platform'] == p).any())

fig, ax = plt.subplots()
for coll in size_swept:
    color = COLLECTIVE_COLORS.get(coll, '#333')
    for plat in PLATFORM_ORDER:
        pts = df[(df['collective'] == coll) & (df['platform'] == plat) & (df['N'] == n_common)].sort_values('msg_size')
        if pts.empty:
            continue
        style = PLATFORM_STYLE[plat]
        yerr = pts['throughput_MBps_std'] if 'throughput_MBps_std' in pts.columns else None
        ax.errorbar(pts['msg_size'], pts['throughput_MBps'], yerr=yerr, marker=style['marker'],
                    linestyle=style['linestyle'], color=color, lw=2, capsize=5,
                    label=f"{coll} — {style['label']}")
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.minorticks_off()
ax.set_xlabel('Message size (bytes, log2)'); ax.set_ylabel('Throughput (MB/s, log scale)')
ax.set_title(f'Collective throughput vs message size — Rivanna vs Fargate  (N={n_common})')
ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.38), frameon=True, fontsize=9)
plt.tight_layout(); plt.savefig('charts/collectives_throughput_vs_msgsize_rivanna_vs_fargate.svg', bbox_inches='tight'); plt.show()
"""


def _cmp_cell_barrier_vs_n() -> str:
    return """import numpy as np
bar = df[df['collective'] == 'barrier']
ns = sorted(bar['N'].unique())
x = np.arange(len(ns))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
for i, plat in enumerate(PLATFORM_ORDER):
    style = PLATFORM_STYLE[plat]
    lat, err = [], []
    for n in ns:
        cell = bar[(bar['platform'] == plat) & (bar['N'] == n)]
        lat.append(cell['latency_p50_ms'].iloc[0] if len(cell) else 0.0)
        err.append(cell['latency_p50_ms_std'].iloc[0] if len(cell) and 'latency_p50_ms_std' in cell.columns else 0.0)
    offset = (i - 0.5) * width
    ax.bar(x + offset, lat, width=width, yerr=err if any(e > 0 for e in err) else None,
          color=style['color'], alpha=0.9, edgecolor=BAR_EDGE, linewidth=BAR_LW, capsize=5, label=style['label'])
ax.set_xticks(x); ax.set_xticklabels([str(n) for n in ns])
ax.set_xlabel('World size N'); ax.set_ylabel('Barrier P50 latency (ms)')
ax.set_title('Barrier synchronization latency vs N — Rivanna vs Fargate')
ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.28), frameon=True)
plt.tight_layout(); plt.savefig('charts/collectives_barrier_vs_N_rivanna_vs_fargate.svg', bbox_inches='tight'); plt.show()
"""


def _cmp_cell_summary_table() -> str:
    return """cols = ['platform', 'channel', 'collective', 'msg_size', 'N', 'latency_p50_ms',
        'latency_p50_ms_std', 'throughput_MBps', 'throughput_MBps_std']
available = [c for c in cols if c in df.columns]
display(df[available].sort_values(['collective', 'N', 'msg_size', 'platform']))
"""


def generate_collectives_compare_notebook(
    results_csv: str,
    output_path: str,
    output_chart_dir: str = "charts",
) -> None:
    """Generate a tweakable Jupyter notebook for the Rivanna-vs-Fargate Exp B charts.

    Cells contain self-contained plotting code (no dependency on
    chart_collectives_compare), so charts can be adjusted by hand. Run the notebook
    from the results directory (e.g. charts_rivanna_vs_fargate/) so the relative CSV
    and chart paths resolve.
    """
    cells = [
        _make_cell("markdown",
                   "# Experiment B: Rivanna (UCC) vs Fargate (direct-redis) Collectives\n\n"
                   "Tweakable notebook. Each chart's plotting code is inline and self-contained "
                   "— edit colors, labels, scales, and figure sizes here, then re-run the cell. "
                   "Run this notebook from its own directory so the relative paths resolve."),
        _make_cell("code", _cmp_cell_imports()),
        _make_cell("code", _cmp_cell_load_data(results_csv)),
        _make_cell("markdown", "## Latency vs world size N"),
        _make_cell("code", _cmp_cell_latency_vs_n()),
        _make_cell("markdown", "## Throughput vs message size"),
        _make_cell("code", _cmp_cell_throughput_vs_msgsize()),
        _make_cell("markdown", "## Barrier synchronization latency vs N"),
        _make_cell("code", _cmp_cell_barrier_vs_n()),
        _make_cell("markdown", "## Summary table"),
        _make_cell("code", _cmp_cell_summary_table()),
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(output_path) or ".", output_chart_dir), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=2)

    logger.info("Rivanna-vs-Fargate collectives notebook saved: %s", output_path)