#!/usr/bin/env python3
"""
Cylon-Armada Experiment Results Pipeline

Orchestrates: download → aggregate → charts → notebook

Usage:
    # Full pipeline from YAML config
    python -m results.pipeline --config configs/experiment_config.yaml

    # From local data only (skip S3)
    python -m results.pipeline --local-dir results/smoke_test/

    # Run individual steps
    python -m results.pipeline --config config.yaml --step aggregate
    python -m results.pipeline --config config.yaml --step charts

    # Quick single-experiment mode
    python -m results.pipeline --platform lambda --instance 1024MB \
        --nodes 1,4,8 --task-counts 4,8,16 --local-dir results/
"""

import argparse
import logging
import os
import sys

from .config import PipelineConfig
from .results_downloader import download_experiment_results
from .results_aggregator import aggregate_all, aggregate_local_dir, save_aggregated_csv
from .chart_generator import generate_all_charts
from .notebook_generator import generate_notebook

logger = logging.getLogger(__name__)

STEPS = ["download", "aggregate", "charts", "notebook"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cylon-Armada experiment results pipeline",
    )

    # Config file mode
    parser.add_argument("--config", type=str, help="YAML config file path")

    # Single-experiment CLI mode
    parser.add_argument("--platform", type=str, default="local",
                        help="Platform name (lambda, fargate, ecs, rivanna, local)")
    parser.add_argument("--instance", type=str, default="local",
                        help="Instance label")
    parser.add_argument("--nodes", type=str, default="1",
                        help="Comma-separated node counts (1,2,4,8)")
    parser.add_argument("--task-counts", type=str, default="4,8,16",
                        help="Comma-separated task counts (4,8,16)")

    # Data source
    parser.add_argument("--bucket", type=str, help="S3 bucket name")
    parser.add_argument("--s3-prefix", type=str, help="S3 prefix pattern")
    parser.add_argument("--local-dir", type=str, help="Local directory with summary CSVs")

    # Output
    parser.add_argument("--download-dir", type=str, default="./data/raw",
                        help="Download directory")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--chart-format", type=str, default="svg",
                        choices=["svg", "png"], help="Chart format")
    parser.add_argument("--chart-dpi", type=int, default=300, help="Chart DPI")

    # Notebook
    parser.add_argument("--notebook-name", type=str,
                        default="context_reuse_results",
                        help="Notebook filename (without .ipynb)")

    # Experiment family: which chart/notebook set to produce.
    parser.add_argument("--experiment", type=str, default="reuse",
                        choices=["reuse", "zerocopy", "collectives"],
                        help="reuse = cost/reuse charts (download/aggregate/charts/notebook); "
                             "zerocopy = Experiment A/A2 charts (charts/notebook on the results dir)")

    # Steps
    parser.add_argument("--step", type=str, action="append", choices=STEPS,
                        help="Run specific step(s). Default: all steps.")

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")

    return parser


def run_pipeline(config: PipelineConfig, steps: list, local_dir: str = None) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    aggregated_csv = os.path.join(config.output_dir, "aggregated_results.csv")

    # Step 1: Download
    if "download" in steps:
        logger.info("=== Step: Download ===")
        download_experiment_results(config)

    # Step 2: Aggregate
    if "aggregate" in steps:
        logger.info("=== Step: Aggregate ===")
        if local_dir and not config.experiments:
            # Quick mode: aggregate a single local directory
            df = aggregate_local_dir(local_dir, platform="local")
        else:
            df = aggregate_all(config.experiments, global_local_dir=local_dir)

        if df.empty:
            logger.error("No data aggregated. Check your config and data paths.")
            return
        save_aggregated_csv(df, aggregated_csv)
        logger.info("Aggregated %d experiment configurations", len(df))

    # Step 3: Charts
    if "charts" in steps:
        logger.info("=== Step: Charts ===")
        import pandas as pd
        if not os.path.exists(aggregated_csv):
            logger.error("Aggregated CSV not found: %s. Run 'aggregate' step first.",
                         aggregated_csv)
            return
        df = pd.read_csv(aggregated_csv)
        generate_all_charts(df, config)
        logger.info("Charts saved to %s", config.output_dir)

    # Step 4: Notebook
    if "notebook" in steps:
        logger.info("=== Step: Notebook ===")
        if not os.path.exists(aggregated_csv):
            logger.error("Aggregated CSV not found: %s. Run 'aggregate' step first.",
                         aggregated_csv)
            return
        notebook_path = os.path.join(
            config.output_dir, f"{config.notebook_name}.ipynb"
        )
        generate_notebook(
            aggregated_csv_path=aggregated_csv,
            output_path=notebook_path,
            output_chart_dir=config.output_dir,
        )
        logger.info("Notebook saved to %s", notebook_path)


def aggregate_zerocopy_runs(results_dir: str) -> bool:
    """Aggregate per-run sweep CSVs into the canonical median CSV with a std column.

    A multi-run cloud sweep writes `<experiment>_run{n}_..._exp_a_zerocopy_results.csv`
    (and `_exp_a2_schema_compat.csv` / `_exp_a_zerocopy_env.csv`) per run. This reduces
    them to `exp_a_zerocopy_results.csv`: for each (format, n, d) cell, every numeric
    column is the median across runs, plus `throughput_roundtrip_MBps_std` (sample std
    across runs) for error bars. `speedup_vs_json` is recomputed from the medians. The
    canonical A2/env CSVs are copied from run 1. Returns True if it aggregated, False if
    no per-run files were found (single-run outputs already write the canonical CSV).
    """
    import csv
    import glob
    import shutil
    import statistics

    # Any prefixed per-run/per-invocation results CSV, excluding the canonical
    # aggregate itself. Catches the multi-execution sweep (`..._run1_...`) and a
    # single warm-container invocation (`warm_measure_...`).
    all_csvs = sorted(glob.glob(os.path.join(results_dir, "*_exp_a_zerocopy_results.csv")))
    run_csvs = [p for p in all_csvs if os.path.basename(p) != "exp_a_zerocopy_results.csv"]
    if not run_csvs:
        return False

    agg, header = {}, None
    for path in run_csvs:
        with open(path) as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            for r in reader:
                agg.setdefault((r["format"], r["n"], r["d"]), []).append(r)

    keep = {"format", "n", "d", "memory_copies", "reuse", "reps"}  # identity/constant cols
    tput = "throughput_roundtrip_MBps"
    out_header = list(header) + ([f"{tput}_std"] if f"{tput}_std" not in header else [])
    rows = []
    for (fmt, n, d), cells in agg.items():
        m = dict(cells[0])
        # Central value = MEAN across runs (matches the scaling spreadsheet's
        # =AVERAGE(runs); "Cylon AWS ECS Scaling Round 2.xlsx", Lambda sheets).
        for col in header:
            if col in keep:
                continue
            vals = []
            for c in cells:
                try:
                    vals.append(float(c[col]))
                except (ValueError, TypeError):
                    vals = None
                    break
            if vals:
                m[col] = statistics.mean(vals)
        # Error bar = SAMPLE std (n-1) across the runs, matching Excel STDEV(runs)
        # in that spreadsheet. statistics.stdev is the sample std. With a single
        # run there is no cross-run spread; fall back to the within-run std the
        # benchmark emits (never clobber it with 0).
        tvals = [float(c[tput]) for c in cells if c.get(tput) not in (None, "", "None")]
        if len(tvals) >= 2:
            m[f"{tput}_std"] = round(statistics.stdev(tvals), 4)
        else:
            try:
                m[f"{tput}_std"] = round(float(cells[0].get(f"{tput}_std", 0.0)), 4)
            except (ValueError, TypeError):
                m[f"{tput}_std"] = 0.0
        rows.append(m)

    # speedup_vs_json is a throughput ratio: format throughput / json throughput
    # (json is the slow baseline, so faster formats are >1). Not json/format.
    json_tp = {(r["n"], r["d"]): float(r[tput]) for r in rows if r["format"] == "json"}
    for r in rows:
        base = json_tp.get((r["n"], r["d"]))
        r["speedup_vs_json"] = round(float(r[tput]) / base, 3) if base and base > 0 else ""
    rows.sort(key=lambda r: (int(r["d"]), int(r["n"])))

    with open(os.path.join(results_dir, "exp_a_zerocopy_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_header)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in out_header})

    # Canonical A2 / env from run 1 (schema-compat + provenance are per-run identical).
    stem = os.path.basename(run_csvs[0]).replace("_exp_a_zerocopy_results.csv", "")
    for suffix, canonical in (("_exp_a2_schema_compat.csv", "exp_a2_schema_compat.csv"),
                              ("_exp_a_zerocopy_env.csv", "exp_a_zerocopy_env.csv")):
        src = os.path.join(results_dir, stem + suffix)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(results_dir, canonical))

    logger.info("Aggregated %d run(s) → exp_a_zerocopy_results.csv (%d cells, +std)",
                len(run_csvs), len(rows))
    return True


def aggregate_collectives_runs(results_dir: str) -> bool:
    """Aggregate per-run Experiment B collective CSVs into the canonical CSV with std.

    A warmed multi-run local launch (or cloud sweep) writes
    `run{n}_exp_b_collectives_results.csv` per run. This reduces them to
    `exp_b_collectives_results.csv`: for each (channel, collective, msg_size, N) cell,
    every numeric metric is the MEAN across runs, plus `latency_p50_ms_std` and
    `throughput_MBps_std` (sample std, n-1) for the error bars. Returns True if it
    aggregated, False if no per-run files were found (a single run already writes the
    canonical CSV).
    """
    import csv
    import glob
    import statistics

    all_csvs = sorted(glob.glob(os.path.join(results_dir, "*_exp_b_collectives_results.csv")))
    run_csvs = [p for p in all_csvs if os.path.basename(p) != "exp_b_collectives_results.csv"]
    if not run_csvs:
        return False

    agg, header = {}, None
    for path in run_csvs:
        with open(path) as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            for r in reader:
                agg.setdefault((r["channel"], r["collective"], r["msg_size"], r["N"]), []).append(r)

    # Identity / constant columns carried through unchanged (not averaged). run_id is
    # meaningless post-aggregation, so keep it fixed rather than averaging it to a
    # fractional value.
    keep = {"channel", "collective", "msg_size", "payload_bytes", "reps",
            "unsupported", "rank", "world_size", "N", "run_id"}
    # Metrics that get a cross-run sample-std column for error bars.
    std_cols = ["latency_p50_ms", "throughput_MBps"]

    out_header = list(header)
    for sc in std_cols:
        if f"{sc}_std" not in out_header:
            out_header.append(f"{sc}_std")

    rows = []
    for key, cells in agg.items():
        m = dict(cells[0])
        # Central value = MEAN across runs (the scaling-spreadsheet convention).
        for col in header:
            if col in keep:
                continue
            vals = []
            for c in cells:
                try:
                    vals.append(float(c[col]))
                except (ValueError, TypeError):
                    vals = None
                    break
            if vals:
                m[col] = round(statistics.mean(vals), 6)
        # Error bar = SAMPLE std (n-1) across runs on the warmed containers.
        for sc in std_cols:
            svals = []
            for c in cells:
                try:
                    svals.append(float(c[sc]))
                except (ValueError, TypeError):
                    svals = None
                    break
            m[f"{sc}_std"] = round(statistics.stdev(svals), 6) if (svals and len(svals) >= 2) else 0.0
        rows.append(m)

    rows.sort(key=lambda r: (r["channel"], r["collective"], int(r["N"]), int(r["msg_size"])))

    with open(os.path.join(results_dir, "exp_b_collectives_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_header)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in out_header})

    logger.info("Aggregated %d run(s) → exp_b_collectives_results.csv (%d cells, +std)",
                len(run_csvs), len(rows))
    return True


def run_collectives_pipeline(results_dir: str, steps: list, chart_format: str = "svg",
                             chart_dpi: int = 300) -> None:
    """Experiment B collective visuals pipeline: aggregate -> charts on a results dir.

    Per-run CSVs are aggregated to the canonical CSV (with cross-run std for error
    bars); a single run already writes it directly. Charts land in
    `results_dir/charts`.
    """
    aggregate_collectives_runs(results_dir)

    results_csv = os.path.join(results_dir, "exp_b_collectives_results.csv")
    if not os.path.exists(results_csv):
        logger.error("No exp_b_collectives_results.csv in %s — run exp_b_collectives.py first.", results_dir)
        return

    if "charts" in steps:
        try:
            from .chart_collectives import generate_collectives_charts
        except ImportError:
            logger.warning("chart_collectives not available yet (SP4 Task 5); aggregation only.")
            return
        charts_dir = os.path.join(results_dir, "charts")
        logger.info("=== Step: Collective charts ===")
        written = generate_collectives_charts(results_dir, charts_dir, chart_format, chart_dpi)
        logger.info("Wrote %d chart(s) to %s", len(written), charts_dir)


def run_zerocopy_pipeline(results_dir: str, steps: list, chart_format: str = "svg",
                          chart_dpi: int = 300, notebook_name: str = "exp_a_zerocopy_charts") -> None:
    """Experiment A / A2 visuals pipeline: aggregate -> charts -> notebook on a results dir.

    A multi-run sweep's per-run CSVs are aggregated to the canonical median CSV (with a
    cross-run std column for error bars); single-run outputs already write it directly.
    Charts land in `results_dir/charts`; the notebook is written to `results_dir` with
    relative paths so it runs from there.
    """
    from .chart_zerocopy import generate_zerocopy_charts
    from .notebook_generator import generate_zerocopy_notebook

    aggregate_zerocopy_runs(results_dir)

    results_csv = os.path.join(results_dir, "exp_a_zerocopy_results.csv")
    if not os.path.exists(results_csv):
        logger.error("No exp_a_zerocopy_results.csv in %s — run exp_a_zerocopy.py first.", results_dir)
        return
    charts_dir = os.path.join(results_dir, "charts")

    if "charts" in steps:
        logger.info("=== Step: Zero-copy charts ===")
        written = generate_zerocopy_charts(results_dir, charts_dir, chart_format, chart_dpi)
        logger.info("Wrote %d chart(s) to %s", len(written), charts_dir)

    if "notebook" in steps:
        logger.info("=== Step: Zero-copy notebook ===")
        notebook_path = os.path.join(results_dir, f"{notebook_name}.ipynb")
        generate_zerocopy_notebook(
            results_csv="exp_a_zerocopy_results.csv",   # relative: run from results_dir
            a2_csv="exp_a2_schema_compat.csv",
            output_path=notebook_path,
            output_chart_dir="charts",
        )
        logger.info("Notebook saved to %s", notebook_path)


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    steps = args.step if args.step else STEPS

    # Zero-copy family (Experiment A / A2): simpler charts->notebook flow on the
    # results dir; bypasses the cost/reuse config + aggregate machinery.
    if args.experiment == "zerocopy":
        if not args.local_dir:
            parser.error("--local-dir (the exp_a_zerocopy results dir) is required for --experiment zerocopy")
        run_zerocopy_pipeline(
            results_dir=args.local_dir,
            steps=steps,
            chart_format=args.chart_format,
            chart_dpi=args.chart_dpi,
            notebook_name=(args.notebook_name if args.notebook_name != "context_reuse_results"
                           else "exp_a_zerocopy_charts"),
        )
        return

    # Experiment B collective benchmark: aggregate per-run CSVs (+cross-run std) then
    # chart, on the results dir. Same lightweight flow as zerocopy.
    if args.experiment == "collectives":
        if not args.local_dir:
            parser.error("--local-dir (the exp_b_collectives results dir) is required for --experiment collectives")
        run_collectives_pipeline(
            results_dir=args.local_dir,
            steps=steps,
            chart_format=args.chart_format,
            chart_dpi=args.chart_dpi,
        )
        return

    # Build config
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    elif args.local_dir:
        config = PipelineConfig.from_args(args)
    else:
        parser.error("Either --config or --local-dir is required")

    # Override output settings from CLI
    config.output_dir = args.output_dir
    config.chart_format = args.chart_format
    config.chart_dpi = args.chart_dpi
    config.notebook_name = args.notebook_name

    # Determine steps
    steps = args.step if args.step else STEPS

    # If local-dir is provided, set it on all experiments
    local_dir = args.local_dir
    if local_dir:
        for exp in config.experiments:
            if not exp.local_data_dir:
                exp.local_data_dir = local_dir

    run_pipeline(config, steps, local_dir=local_dir)


if __name__ == "__main__":
    main()