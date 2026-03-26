"""Generate LLM analysis tasks from real astronomical inference results.

Takes AstroMAE inference output (predictions, magnitudes, true redshifts)
and generates semantically-clustered LLM tasks for context reuse experiments.

Tasks are designed to naturally cluster — similar galaxies produce similar
analysis prompts, creating the semantic overlap that context reuse exploits.

Configuration:
    Templates and survey types can be provided via:
    1. Config file (JSON) passed to load_config()
    2. Direct parameter override on generate_tasks_from_results()
    3. Falls back to built-in defaults if neither is provided
"""

import json
import logging
import os
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# SDSS photometric band names
BANDS = ("u", "g", "r", "i", "z")

# Built-in defaults — used only when no config file or parameter is provided
_DEFAULT_TEMPLATES = {
    "redshift_analysis": (
        "Analyze the photometric redshift prediction z={z_pred:.3f} "
        "(true z={z_true:.3f}) for a galaxy with SDSS magnitudes "
        "{band_str}. Assess the prediction accuracy and classify "
        "the likely galaxy morphological type based on the color profile."
    ),
    "color_classification": (
        "Given SDSS color indices {color_str} and predicted redshift "
        "z={z_pred:.3f}, classify this galaxy's morphological type "
        "and assess whether the colors are consistent with the "
        "predicted redshift."
    ),
    "outlier_analysis": (
        "The AstroMAE model predicted z={z_pred:.3f} for a galaxy "
        "with true spectroscopic redshift z={z_true:.3f} "
        "(residual={residual:.4f}). The galaxy has magnitudes {band_str}. "
        "Analyze whether this prediction error is significant and "
        "identify possible causes."
    ),
    "batch_summary": (
        "Summarize the inference results for a batch of {n} galaxies: "
        "mean predicted redshift z={mean_z:.3f}, MAE={mae:.4f}, "
        "bias={bias:.4f}, precision(NMAD)={nmad:.4f}. "
        "Assess whether this accuracy meets the requirements for "
        "{survey_type} surveys."
    ),
    "cost_analysis": (
        "For a serverless inference run processing {n} galaxy images "
        "in {time_s:.1f} seconds at {throughput:.1f} Gbps throughput, "
        "analyze the cost-effectiveness compared to traditional HPC. "
        "The batch used {batch_size} samples with magnitudes ranging "
        "{mag_range}."
    ),
}

_DEFAULT_SURVEY_TYPES = [
    "large-scale structure",
    "cosmological distance measurement",
    "galaxy cluster identification",
    "weak gravitational lensing",
    "baryon acoustic oscillation",
]


def load_config(config_path):
    """Load task generator configuration from a JSON file.

    Expected format:
        {
            "templates": {
                "redshift_analysis": "...",
                "color_classification": "...",
                ...
            },
            "survey_types": ["...", "..."],
            "bands": ["u", "g", "r", "i", "z"]
        }

    All fields are optional — missing fields fall back to defaults.

    Args:
        config_path: Path to JSON config file.

    Returns:
        dict with 'templates', 'survey_types', and 'bands' keys.
    """
    with open(config_path) as f:
        data = json.load(f)

    return {
        "templates": {**_DEFAULT_TEMPLATES, **data.get("templates", {})},
        "survey_types": data.get("survey_types", _DEFAULT_SURVEY_TYPES),
        "bands": tuple(data.get("bands", BANDS)),
    }


def _resolve_config(templates=None, survey_types=None, config_path=None):
    """Resolve configuration with precedence:
    1. Direct parameters (templates, survey_types)
    2. Config file (config_path)
    3. Environment variable COSMIC_AI_CONFIG pointing to config file
    4. Built-in defaults

    Returns:
        (templates_dict, survey_types_list, bands_tuple)
    """
    file_config = None

    # Check env var for config path
    if config_path is None:
        config_path = os.environ.get("COSMIC_AI_CONFIG")

    if config_path and os.path.exists(config_path):
        file_config = load_config(config_path)
        logger.info("Loaded task generator config from %s", config_path)

    resolved_templates = (
        templates
        or (file_config["templates"] if file_config else None)
        or _DEFAULT_TEMPLATES
    )
    resolved_survey_types = (
        survey_types
        or (file_config["survey_types"] if file_config else None)
        or _DEFAULT_SURVEY_TYPES
    )
    resolved_bands = (
        (file_config["bands"] if file_config else None)
        or BANDS
    )

    return resolved_templates, resolved_survey_types, resolved_bands


def _format_bands(magnitudes, bands=BANDS):
    """Format magnitude values as 'u=22.31, g=21.08, ...'."""
    return ", ".join(
        f"{band}={mag:.2f}" for band, mag in zip(bands, magnitudes)
    )


def _format_colors(magnitudes, bands=BANDS):
    """Format color indices as 'u-g=1.23, g-r=0.63, ...'."""
    colors = []
    for i in range(len(bands) - 1):
        diff = magnitudes[i] - magnitudes[i + 1]
        colors.append(f"{bands[i]}-{bands[i+1]}={diff:.2f}")
    return ", ".join(colors)


def generate_tasks_from_results(
    predictions,
    true_redshifts,
    magnitudes,
    metrics=None,
    max_tasks=None,
    seed=None,
    templates=None,
    survey_types=None,
    config_path=None,
):
    """Generate LLM analysis tasks from inference results.

    Args:
        predictions: Array of predicted redshifts (N,).
        true_redshifts: Array of true redshifts (N,).
        magnitudes: Array of magnitude values (N, 5).
        metrics: Optional inference metrics dict (for batch/cost tasks).
        max_tasks: Maximum number of tasks to generate. If None, generates
            one task per sample plus batch-level tasks.
        seed: Random seed for reproducible task selection.
        templates: Optional dict of custom templates (overrides config file).
        survey_types: Optional list of survey type strings (overrides config file).
        config_path: Optional path to JSON config file.

    Returns:
        List of task description strings.
    """
    if seed is not None:
        random.seed(seed)

    resolved_templates, resolved_survey_types, resolved_bands = _resolve_config(
        templates=templates,
        survey_types=survey_types,
        config_path=config_path,
    )

    predictions = np.asarray(predictions)
    true_redshifts = np.asarray(true_redshifts)
    magnitudes = np.asarray(magnitudes)
    n_samples = len(predictions)

    tasks = []

    # Per-sample tasks — these naturally cluster by galaxy similarity
    residuals = np.abs(predictions - true_redshifts)
    indices = list(range(n_samples))

    if max_tasks and max_tasks < n_samples:
        # Select a mix: some normal, some outliers for diversity
        n_outliers = max(1, max_tasks // 4)
        n_normal = max_tasks - n_outliers

        outlier_indices = np.argsort(residuals)[-n_outliers:].tolist()
        normal_pool = [i for i in indices if i not in outlier_indices]
        normal_indices = random.sample(normal_pool, min(n_normal, len(normal_pool)))
        selected = normal_indices + outlier_indices
    else:
        selected = indices

    outlier_threshold = np.percentile(residuals, 90)

    for idx in selected:
        mags = magnitudes[idx]
        z_pred = float(predictions[idx])
        z_true = float(true_redshifts[idx])
        residual = float(residuals[idx])
        band_str = _format_bands(mags, resolved_bands)
        color_str = _format_colors(mags, resolved_bands)

        # Choose template based on residual — outliers get outlier_analysis
        if residual > outlier_threshold:
            template = resolved_templates.get("outlier_analysis", "")
            tasks.append(template.format(
                z_pred=z_pred, z_true=z_true, residual=residual,
                band_str=band_str,
            ))
        elif idx % 2 == 0:
            template = resolved_templates.get("redshift_analysis", "")
            tasks.append(template.format(
                z_pred=z_pred, z_true=z_true, band_str=band_str,
            ))
        else:
            template = resolved_templates.get("color_classification", "")
            tasks.append(template.format(
                z_pred=z_pred, color_str=color_str,
            ))

    # Batch-level tasks — summaries that cluster with each other
    if metrics:
        delta_z = (predictions - true_redshifts) / (1 + true_redshifts)
        mae = float(np.mean(np.abs(predictions - true_redshifts)))
        bias = float(np.mean(delta_z))
        nmad = float(1.48 * np.median(np.abs(delta_z - np.median(delta_z))))

        for survey_type in random.sample(
            resolved_survey_types, min(2, len(resolved_survey_types))
        ):
            template = resolved_templates.get("batch_summary", "")
            tasks.append(template.format(
                n=n_samples, mean_z=float(np.mean(predictions)),
                mae=mae, bias=bias, nmad=nmad,
                survey_type=survey_type,
            ))

        template = resolved_templates.get("cost_analysis", "")
        mag_min = float(np.min(magnitudes))
        mag_max = float(np.max(magnitudes))
        tasks.append(template.format(
            n=n_samples,
            time_s=metrics.get("total_time_s", 0),
            throughput=metrics.get("throughput_bps", 0) / 1e9,
            batch_size=metrics.get("batch_size", 512),
            mag_range=f"{mag_min:.1f}-{mag_max:.1f}",
        ))

    logger.info(
        "Generated %d tasks from %d inference results (%d per-sample, %d batch-level)",
        len(tasks), n_samples, len(selected), len(tasks) - len(selected),
    )

    return tasks


def generate_tasks_from_data(
    data_path,
    model_path,
    batch_size=512,
    device="cpu",
    max_tasks=None,
    seed=None,
    templates=None,
    survey_types=None,
    config_path=None,
):
    """End-to-end: load data, run inference, generate tasks.

    Convenience function that chains inference → task generation.

    Args:
        data_path: Path to SDSS .pt data partition.
        model_path: Path to pre-trained AstroMAE model checkpoint.
        batch_size: Inference batch size.
        device: 'cpu' or 'cuda'.
        max_tasks: Maximum tasks to generate.
        seed: Random seed.
        templates: Optional dict of custom templates (overrides config file).
        survey_types: Optional list of survey type strings (overrides config file).
        config_path: Optional path to JSON config file.

    Returns:
        (tasks, inference_results) tuple.
    """
    from .inference import load_data, load_model, run_inference

    dataset = load_data(data_path, device=device)
    model = load_model(model_path, device=device)
    results = run_inference(model, dataset, batch_size=batch_size, device=device)

    tasks = generate_tasks_from_results(
        predictions=results["predictions"],
        true_redshifts=results["true_redshifts"],
        magnitudes=results["magnitudes"],
        metrics=results["metrics"],
        max_tasks=max_tasks,
        seed=seed,
        templates=templates,
        survey_types=survey_types,
        config_path=config_path,
    )

    return tasks, results