"""
Results aggregator for cylon-armada experiment summary files.

Follows Cylon's results pipeline pattern:
  1. Each summary CSV = one run of one experiment configuration
  2. Multiple runs of the same config → cross-run mean/std
  3. Output: aggregated CSV with {metric}_mean / {metric}_std columns

File discovery:
  - Summary CSVs: {experiment_name}_summary.csv
  - Grouped by experiment_name for cross-run statistics

Aggregation methodology:
  - Per-run: one summary CSV row per experiment execution
  - Cross-run: mean and std (N-1) across runs with matching experiment name
  - Timing columns (ms) are converted to seconds after aggregation
"""

import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import METRIC_COLUMNS, MS_TO_S_COLUMNS

logger = logging.getLogger(__name__)


def parse_summary_csv(filepath: str) -> Optional[Dict]:
    """Parse a single summary CSV file.

    Expected format (from ExperimentBenchmark._write_summary_csv):
        experiment_name,total_ms,total_cost,baseline_cost,savings_pct,
        reuse_rate,cache_hits,llm_calls,task_count,similarity_threshold,
        embedding_dimensions,backend,baseline

    Returns a dict with all columns, or None if unparseable.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", filepath, e)
        return None

    if df.empty:
        return None

    return df.iloc[0].to_dict()


def _extract_config_from_name(experiment_name: str) -> Dict:
    """Parse experiment parameters from the naming convention.

    Names like: reuse_t4_th0.8_d256_redis_NUMPY
                baseline_t4_d256_cylon_NUMPY
    """
    result = {"experiment_name": experiment_name}
    result["baseline"] = experiment_name.startswith("baseline_")
    result["system"] = "llamaindex" if experiment_name.startswith("llamaindex_") else "cylon"

    m = re.search(r"_t(\d+)_", experiment_name)
    if m:
        result["task_count"] = int(m.group(1))

    m = re.search(r"_th([\d.]+)_", experiment_name)
    if m:
        result["similarity_threshold"] = float(m.group(1))

    m = re.search(r"_d(\d+)_", experiment_name)
    if m:
        result["embedding_dimensions"] = int(m.group(1))

    # Suffix after dimensions: context_backend_SIMD_BACKEND
    m = re.search(r"_d\d+_(.*)", experiment_name)
    if m:
        suffix = m.group(1)
        for simd in ["CYTHON_BATCH", "PYCYLON", "NUMPY"]:
            if suffix.endswith(simd):
                result["backend"] = simd
                ctx = suffix[: -(len(simd) + 1)]
                if ctx:
                    result["context_backend"] = ctx
                break

    return result


def discover_summary_files(local_dir: str) -> List[str]:
    """Find all summary CSV files in a directory tree.

    Returns list of file paths sorted by name.
    """
    results = []
    if not os.path.isdir(local_dir):
        logger.warning("Directory not found: %s", local_dir)
        return results

    for root, _, filenames in os.walk(local_dir):
        for fname in filenames:
            if fname.endswith("_summary.csv"):
                results.append(os.path.join(root, fname))

    results.sort()
    return results


def group_files_by_config(files: List[str]) -> Dict[str, List[str]]:
    """Group summary CSV files by experiment name.

    Files with the same experiment_name across different run directories
    are grouped for cross-run aggregation.

    Returns {experiment_name: [file_paths]}.
    """
    groups: Dict[str, List[str]] = {}

    for filepath in files:
        fname = os.path.basename(filepath)
        exp_name = fname.replace("_summary.csv", "")
        if exp_name not in groups:
            groups[exp_name] = []
        groups[exp_name].append(filepath)

    return groups


def aggregate_runs(
    files: List[str],
    platform: str = "local",
    node_count: int = 1,
) -> Optional[Dict]:
    """Aggregate multiple runs of the same experiment configuration.

    Each file is one run's summary CSV. Computes mean and std (N-1)
    across all runs for each metric column.

    Returns a dict with:
      - Parameter columns (from experiment name parsing + CSV data)
      - {metric}_mean and {metric}_std for each metric
      - Metadata: platform, node_count, num_runs
    """
    run_data = []
    config_info = None

    for filepath in files:
        row = parse_summary_csv(filepath)
        if row is None:
            continue
        run_data.append(row)
        if config_info is None:
            exp_name = row.get(
                "experiment_name",
                os.path.basename(filepath).replace("_summary.csv", ""),
            )
            config_info = _extract_config_from_name(exp_name)
            # Also grab direct params if present in CSV
            for col in ["task_count", "similarity_threshold",
                         "embedding_dimensions", "backend",
                         "context_backend", "baseline", "system"]:
                if col in row and col not in config_info:
                    config_info[col] = row[col]

    if not run_data:
        return None

    runs_df = pd.DataFrame(run_data)
    result = {
        "platform": platform,
        "node_count": node_count,
        "num_runs": len(run_data),
    }

    if config_info:
        result.update(config_info)

    # Compute mean/std for each metric
    for col in METRIC_COLUMNS:
        if col in runs_df.columns and runs_df[col].notna().any():
            values = runs_df[col].dropna().astype(float)
            divisor = 1000.0 if col in MS_TO_S_COLUMNS else 1.0
            result[f"{col}_mean"] = values.mean() / divisor
            result[f"{col}_std"] = values.std() / divisor if len(values) > 1 else 0.0
        else:
            result[f"{col}_mean"] = np.nan
            result[f"{col}_std"] = np.nan

    result["has_cost_data"] = (
        pd.notna(result.get("total_cost_mean"))
        and result.get("total_cost_mean", 0) > 0
    )
    result["has_reuse_data"] = (
        pd.notna(result.get("reuse_rate_mean"))
        and result.get("reuse_rate_mean", 0) > 0
    )

    return result


def aggregate_all(
    experiments: list,
    global_local_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate results for all configured experiments.

    Discovers summary CSVs, groups by experiment name, computes
    cross-run mean/std, returns one row per unique configuration.
    """
    all_results = []

    for exp in experiments:
        local_dir = global_local_dir or exp.local_data_dir
        if not local_dir:
            logger.warning("No data source for %s, skipping", exp.label)
            continue

        logger.info("Aggregating: %s", exp.label)

        files = discover_summary_files(local_dir)
        if not files:
            logger.warning("  No summary CSVs found in %s", local_dir)
            continue

        groups = group_files_by_config(files)
        logger.info(
            "  Found %d experiment configs, %d total files",
            len(groups), len(files),
        )

        for exp_name, group_files in groups.items():
            for nc in exp.node_counts:
                result = aggregate_runs(
                    files=group_files,
                    platform=exp.platform,
                    node_count=nc,
                )
                if result:
                    result["instance_label"] = exp.instance_label
                    result["instance_detail"] = exp.instance_detail
                    all_results.append(result)

    if not all_results:
        logger.warning("No results aggregated")
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def aggregate_local_dir(
    local_dir: str,
    platform: str = "local",
    node_count: int = 1,
) -> pd.DataFrame:
    """Quick aggregation of a single local directory.

    Convenience function for local development — discovers all summary
    CSVs, groups by experiment name, computes mean/std.
    """
    files = discover_summary_files(local_dir)
    if not files:
        logger.warning("No summary CSVs found in %s", local_dir)
        return pd.DataFrame()

    groups = group_files_by_config(files)
    results = []

    for exp_name, group_files in groups.items():
        result = aggregate_runs(group_files, platform=platform, node_count=node_count)
        if result:
            results.append(result)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def save_aggregated_csv(df: pd.DataFrame, output_path: str) -> None:
    """Save aggregated results to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved aggregated results to %s (%d rows)", output_path, len(df))