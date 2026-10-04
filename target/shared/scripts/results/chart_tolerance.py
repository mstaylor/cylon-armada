"""Chart 7 of the Cosmic AI data plane spec: the reuse gate's tolerance
operating characteristic, from experiment.armc_tolerance_sweep's score.json.

One panel per template. Each point is one tolerance: reuse rate against verdict
agreement with the query's own fresh answer, with Wilson 95% intervals. The
self-agreement ceiling is drawn as the shaded unreachable region above it.

Usage:
    python -m results.chart_tolerance --score <out-dir>/score.json --output <dir>
"""

import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

FONT_SIZE = 12
TITLE_SIZE = 14
TICK_SIZE = 10
LEGEND_SIZE = 10

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
})

PANELS = (
    ("redshift_analysis", "redshift_analysis (oracle key)"),
    ("outlier_analysis", "outlier_analysis"),
    ("photometry_classification", "photometry_classification"),
)
SCHEME_STYLE = {
    "zpred_only": {"label": "predicted-redshift gate (no labels)", "color": "#7f7f7f",
                   "marker": "s", "facecolor": "white"},
    "production": {"label": "per-template gate (shipped)", "color": "#2ca02c", "marker": "o",
                   "facecolor": "#2ca02c"},
}
CEILING_COLOR = "#d9d9d9"


METRICS = {
    "agreement": {"value": "verdict_agreement", "ceiling": "self_agreement",
                  "ylabel": "verdict agreement with fresh answer", "ylim": (0, 1.05)},
    "kappa": {"value": "kappa", "ceiling": "self_kappa_on_reused",
              "ylabel": "Cohen's kappa against fresh answer", "ylim": (-0.3, 1.05)},
}


def _points(results, scheme, template, part, metric):
    points = []
    for tolerance, groups in results[scheme].items():
        summary = groups.get(f"{template}|{part}")
        if not summary or summary[METRICS[metric]["value"]] is None:
            continue
        value = summary[METRICS[metric]["value"]]
        lo, hi = (summary["verdict_agreement_ci95"] if metric == "agreement"
                  else (value, value))
        points.append((tolerance, summary["reuse_rate"], value, lo, hi))
    return points


def generate_tolerance_chart(score_path, output_dir, chart_format="png", chart_dpi=300,
                             part="full", metric="agreement"):
    with open(score_path) as f:
        score = json.load(f)
    results = score["results"]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 5.5), sharey=True)
    ceiling_handle = None
    for ax, (template, title) in zip(axes, PANELS):
        ceiling = results["production"]["ungated"][f"{template}|{part}"][
            METRICS[metric]["ceiling"]]
        if ceiling is not None:
            ceiling_handle = ax.axhspan(ceiling, 1.05, color=CEILING_COLOR, linewidth=0,
                                        label=f"above self-agreement ceiling (unreachable)")
            top = METRICS[metric]["ylim"][1]
            ax.text(0.5, (ceiling + top) / 2, f"self-agreement ceiling {ceiling:.3f}",
                    ha="center", va="center", fontsize=TICK_SIZE,
                    transform=ax.get_yaxis_transform())
        for scheme, style in SCHEME_STYLE.items():
            points = _points(results, scheme, template, part, metric)
            if not points:
                continue
            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            yerr = [[p[2] - p[3] for p in points], [p[4] - p[2] for p in points]]
            ax.errorbar(xs, ys, yerr=yerr, fmt=style["marker"], color=style["color"],
                        markerfacecolor=style["facecolor"], markeredgecolor=style["color"],
                        markersize=8, capsize=5, linestyle="none", label=style["label"])
            if scheme == "production":
                coincident = {}
                for tolerance, x, y, _, _ in points:
                    coincident.setdefault((round(x, 3), round(y, 3)), []).append(tolerance)
                for (x, y), tolerances in coincident.items():
                    text = (", ".join(tolerances) if len(tolerances) <= 2
                            else f"{tolerances[0]} to {tolerances[-1]}")
                    right_edge = x > 0.8
                    ax.annotate(text, (x, y), textcoords="offset points",
                                xytext=(-8, 10 + 12 * (len(tolerances) > 1)) if right_edge
                                else (6, -14),
                                ha="right" if right_edge else "left", fontsize=TICK_SIZE - 1)
        ax.set_title(title, fontsize=FONT_SIZE)
        ax.set_xlabel("reuse rate", fontsize=FONT_SIZE)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(*METRICS[metric]["ylim"])
        ax.tick_params(labelsize=TICK_SIZE)
    axes[0].set_ylabel(METRICS[metric]["ylabel"], fontsize=FONT_SIZE)
    fig.suptitle("Reuse gate tolerance operating characteristic "
                 f"({score['population']} galaxies, {part} population)", fontsize=TITLE_SIZE)
    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] != (ceiling_handle.get_label()
                                                                  if ceiling_handle else ""))
    handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3,
               frameon=True, fontsize=LEGEND_SIZE)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"chart7_tolerance_operating_characteristic_{metric}_"
                                    f"{part}.{chart_format}")
    fig.savefig(path, dpi=chart_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--score", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", default="png", choices=["png", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--part", nargs="+", default=["full", "report"])
    parser.add_argument("--metric", nargs="+", default=list(METRICS), choices=list(METRICS))
    args = parser.parse_args()
    for part in args.part:
        for metric in args.metric:
            print(generate_tolerance_chart(args.score, args.output, args.format, args.dpi,
                                           part, metric))


if __name__ == "__main__":
    main()
