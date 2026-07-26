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
                        choices=["reuse", "zerocopy"],
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


def run_zerocopy_pipeline(results_dir: str, steps: list, chart_format: str = "svg",
                          chart_dpi: int = 300, notebook_name: str = "exp_a_zerocopy_charts") -> None:
    """Experiment A / A2 visuals pipeline: charts -> notebook on a results dir.

    exp_a_zerocopy.py already writes final CSVs (medians computed in-run), so
    there is no download/aggregate — the charts and notebook read the CSVs in
    `results_dir` directly. Charts land in `results_dir/charts`; the notebook is
    written to `results_dir` with relative paths so it runs from there.
    """
    from .chart_zerocopy import generate_zerocopy_charts
    from .notebook_generator import generate_zerocopy_notebook

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