"""Charts for Experiment A / A2 (zero-copy data plane benchmark).

Standalone from chart_generator.py: the zero-copy metric family (throughput by
serialization format, memory copies) does not fit the cost/reuse charts, so this
module reads exp_a_zerocopy_results.csv / exp_a2_schema_compat.csv directly and
emits its own figures. It matches the existing chart style constants
(FONT_SIZE=12, figsize=(10,6), svg, dpi=300).

Usage:
    python chart_zerocopy.py --results results/exp_a_zerocopy \
        --output results/exp_a_zerocopy [--format svg] [--dpi 300]
"""

import argparse
import csv
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Match chart_generator.py style constants.
FONT_SIZE = 12
TITLE_SIZE = 14
TICK_SIZE = 10
LEGEND_SIZE = 10

# Semantic colors: the proposed system green, naive text red, binary baselines
# muted, zero-copy binary (flatbuffers) orange.
FORMAT_COLORS = {
    "arrow_ipc": "#2ca02c",         # proposed zero-copy data plane (green)
    "contexttable_ipc": "#17becf",  # store snapshot (teal)
    "flatbuffers": "#ff7f0e",       # zero-copy binary (orange)
    "pickle": "#7f7f7f",            # binary baseline (gray)
    "protobuf": "#8c564b",          # binary baseline (brown)
    "base64_tobytes": "#9467bd",    # current FMI encoding (purple)
    "json": "#d62728",             # naive text baseline (red)
}
# Canonical display order (fast -> slow).
FORMAT_ORDER = ["arrow_ipc", "contexttable_ipc", "flatbuffers", "pickle",
                "base64_tobytes", "protobuf", "json"]


def _load_rows(results_csv):
    with open(results_csv) as f:
        return list(csv.DictReader(f))


def _representative_cell(rows):
    """Pick the largest (N, D) cell — the clearest signal."""
    def key(r):
        return (int(r["n"]), int(r["d"]))
    best = max(key(r) for r in rows)
    return [r for r in rows if key(r) == best], best


def _ordered(rows_by_fmt):
    return [f for f in FORMAT_ORDER if f in rows_by_fmt]


def _save(fig, output_dir, name, chart_format, chart_dpi):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.{chart_format}")
    fig.savefig(path, dpi=chart_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def chart_throughput_by_format(rows, output_dir, chart_format, chart_dpi):
    """Horizontal bar of round-trip throughput by format (log x), at the
    representative cell, annotated with speedup vs JSON."""
    cell, (n, d) = _representative_cell(rows)
    by_fmt = {r["format"]: r for r in cell}
    fmts = _ordered(by_fmt)
    vals = [float(by_fmt[f]["throughput_roundtrip_MBps"]) for f in fmts]
    speedups = [by_fmt[f].get("speedup_vs_json") for f in fmts]
    colors = [FORMAT_COLORS.get(f, "#333333") for f in fmts]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = range(len(fmts))
    ax.barh(list(y), vals, color=colors, alpha=0.85)
    ax.set_yticks(list(y))
    ax.set_yticklabels(fmts, fontsize=TICK_SIZE)
    ax.invert_yaxis()  # fastest on top
    ax.set_xscale("log")
    ax.set_xlabel("Round-trip throughput (MB/s, log scale)", fontsize=FONT_SIZE)
    ax.set_title(f"Zero-copy data plane throughput by format  (N={n}, D={d})",
                 fontsize=TITLE_SIZE)
    for i, (v, s) in enumerate(zip(vals, speedups)):
        label = f"{v:,.0f} MB/s"
        if s not in (None, "", "None"):
            label += f"  ({float(s):,.0f}x vs JSON)"
        ax.text(v * 1.05, i, label, va="center", fontsize=TICK_SIZE - 1)
    ax.grid(axis="x", alpha=0.3, which="both")
    ax.set_xlim(right=max(vals) * 12)
    return _save(fig, output_dir, "exp_a_throughput_by_format", chart_format, chart_dpi)


def chart_memory_behavior(rows, output_dir, chart_format, chart_dpi):
    """Deserialize peak allocation by format (log y), corroborating memory_copies.
    Zero-copy formats allocate ~nothing; copy formats allocate ~1x payload; JSON
    balloons from Python object overhead."""
    cell, (n, d) = _representative_cell(rows)
    by_fmt = {r["format"]: r for r in cell}
    fmts = _ordered(by_fmt)
    peaks = [max(float(by_fmt[f]["deserialize_peak_kb"]), 0.1) for f in fmts]
    copies = [by_fmt[f]["memory_copies"] for f in fmts]
    colors = [FORMAT_COLORS.get(f, "#333333") for f in fmts]
    payload_kb = n * d * 4 / 1024

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(fmts))
    ax.bar(list(x), peaks, color=colors, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(fmts, fontsize=TICK_SIZE, rotation=30, ha="right")
    ax.set_ylabel("Deserialize peak allocation (KB, log scale)", fontsize=FONT_SIZE)
    ax.set_title(f"Memory copies on read  (N={n}, D={d}; payload = {payload_kb:,.0f} KB)",
                 fontsize=TITLE_SIZE)
    ax.axhline(payload_kb, color="#555555", ls="--", lw=1, alpha=0.7)
    ax.text(len(fmts) - 0.5, payload_kb * 1.1, "1x payload", fontsize=TICK_SIZE - 1,
            ha="right", color="#555555")
    for i, (p, c) in enumerate(zip(peaks, copies)):
        ax.text(i, p * 1.3, f"{c} cop{'y' if str(c)=='1' else 'ies'}",
                ha="center", fontsize=TICK_SIZE - 1)
    ax.grid(axis="y", alpha=0.3, which="both")
    return _save(fig, output_dir, "exp_a_memory_copies", chart_format, chart_dpi)


def chart_throughput_scaling(rows, output_dir, chart_format, chart_dpi):
    """Round-trip throughput vs embedding dim, one line per format (log y)."""
    fmts_present = {r["format"] for r in rows}
    fmts = [f for f in FORMAT_ORDER if f in fmts_present]
    dims = sorted({int(r["d"]) for r in rows})
    # Aggregate over N by taking the largest N per (format, d) for a clean line.
    fig, ax = plt.subplots(figsize=(10, 6))
    for f in fmts:
        ys = []
        for d in dims:
            cand = [r for r in rows if r["format"] == f and int(r["d"]) == d]
            if not cand:
                ys.append(None)
                continue
            top = max(cand, key=lambda r: int(r["n"]))
            ys.append(float(top["throughput_roundtrip_MBps"]))
        ax.plot(dims, ys, marker="o", label=f, color=FORMAT_COLORS.get(f, "#333333"), lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("Embedding dimension D", fontsize=FONT_SIZE)
    ax.set_ylabel("Round-trip throughput (MB/s, log scale)", fontsize=FONT_SIZE)
    ax.set_title("Throughput vs embedding dimension", fontsize=TITLE_SIZE)
    ax.set_xticks(dims)
    ax.legend(fontsize=LEGEND_SIZE, ncol=2)
    ax.grid(alpha=0.3, which="both")
    return _save(fig, output_dir, "exp_a_throughput_scaling", chart_format, chart_dpi)


def chart_schema_compat(a2_csv, output_dir, chart_format, chart_dpi):
    """A2: per-edge Arrow schema compatibility / zero-copy eligibility matrix."""
    with open(a2_csv) as f:
        edges = list(csv.DictReader(f))
    if not edges:
        return None
    fig, ax = plt.subplots(figsize=(10, max(3, 0.9 * len(edges) + 1)))
    ax.axis("off")
    rows = []
    colors = []
    for e in edges:
        zc = e["zero_copy_eligible"] == "True"
        compat = e["arrow_compatible"] == "True"
        status = "zero-copy" if zc else ("compatible" if compat else "needs serialization")
        rows.append([e["edge"], e["payload_class"], status])
        colors.append("#2ca02c" if zc else ("#ff7f0e" if compat else "#d62728"))
    table = ax.table(
        cellText=rows,
        colLabels=["Operator edge", "Payload class", "Transfer"],
        loc="center", cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE)
    table.scale(1, 1.8)
    for i, c in enumerate(colors):
        cell = table[i + 1, 2]
        cell.set_facecolor(c)
        cell.set_alpha(0.35)
    ax.set_title("A2: Arrow schema compatibility across the operator DAG",
                 fontsize=TITLE_SIZE, pad=20)
    return _save(fig, output_dir, "exp_a2_schema_compat", chart_format, chart_dpi)


def generate_zerocopy_charts(results_dir, output_dir=None, chart_format="svg", chart_dpi=300):
    """Generate all Experiment A / A2 charts from a results directory."""
    output_dir = output_dir or results_dir
    results_csv = os.path.join(results_dir, "exp_a_zerocopy_results.csv")
    a2_csv = os.path.join(results_dir, "exp_a2_schema_compat.csv")
    written = []
    if os.path.exists(results_csv):
        rows = _load_rows(results_csv)
        written.append(chart_throughput_by_format(rows, output_dir, chart_format, chart_dpi))
        written.append(chart_memory_behavior(rows, output_dir, chart_format, chart_dpi))
        if len({r["d"] for r in rows}) > 1:
            written.append(chart_throughput_scaling(rows, output_dir, chart_format, chart_dpi))
    else:
        logger.warning("no results CSV at %s", results_csv)
    if os.path.exists(a2_csv):
        p = chart_schema_compat(a2_csv, output_dir, chart_format, chart_dpi)
        if p:
            written.append(p)
    return written


def main():
    parser = argparse.ArgumentParser(description="Experiment A / A2 charts")
    parser.add_argument("--results", required=True, help="dir with exp_a_zerocopy_results.csv")
    parser.add_argument("--output", default=None, help="output dir (default: same as --results)")
    parser.add_argument("--format", default="svg", choices=["svg", "png"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    written = generate_zerocopy_charts(args.results, args.output, args.format, args.dpi)
    logger.info("generated %d chart(s)", len(written))


if __name__ == "__main__":
    main()