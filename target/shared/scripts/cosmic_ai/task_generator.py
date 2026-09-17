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
        "(true z={z_true:.3f}) for a galaxy whose standardized ugriz "
        "photometry features are {band_str}. Assess the prediction "
        "accuracy and what it implies for this galaxy's distance estimate."
    ),
    "photometry_classification": (
        "Given standardized ugriz photometry features {band_str} and "
        "predicted redshift z={z_pred:.3f}, assess whether the "
        "photometry is consistent with the predicted redshift and "
        "flag any features that look anomalous."
    ),
    "outlier_analysis": (
        "The AstroMAE model predicted z={z_pred:.3f} for a galaxy "
        "with true spectroscopic redshift z={z_true:.3f} "
        "(residual={residual:.4f}). Its standardized ugriz photometry "
        "features are {band_str}. Analyze whether this prediction error "
        "is significant and identify possible causes."
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
        "The batch used {batch_size} samples with standardized photometry "
        "features ranging {feature_range}."
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
                "photometry_classification": "...",
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


def _format_bands(photometry, bands=BANDS):
    """Format per-band photometry features as 'u=0.37, g=1.21, ...'.

    The values AstroMAE consumes are standardized per band (zero mean, unit
    variance), not apparent magnitudes, so they are never labelled as
    magnitudes in a prompt. Colour indices are deliberately not derived from
    them either: a difference of two standardized values is not a colour.
    """
    return ", ".join(
        f"{band}={value:.2f}" for band, value in zip(bands, photometry)
    )


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
    outlier_threshold=None,
    index_offset=0,
):
    """Generate LLM analysis tasks from inference results.

    Args:
        predictions: Array of predicted redshifts (N,).
        true_redshifts: Array of true redshifts (N,).
        magnitudes: Array of per-band photometry features (N, 5). These are the
            standardized values AstroMAE consumes, not apparent magnitudes.
        metrics: Optional inference metrics dict (for batch/cost tasks).
        max_tasks: Maximum number of tasks to generate. If None, generates
            one task per sample plus batch-level tasks.
        seed: Random seed for reproducible task selection.
        outlier_threshold: Residual above which a galaxy gets the outlier
            template. ``None`` (default) derives it from THIS call's residuals,
            which makes the prompt for a given galaxy depend on which other
            galaxies were passed alongside it — so a sharded run gives the same
            galaxy different prompts at different world sizes. Pass a fixed
            value to make prompts a property of the galaxy alone, which any
            experiment that varies the shard count must do or its workload
            moves with the independent variable.
        index_offset: Position of this array's first row in the whole
            population. Template choice alternates on index parity, and that
            index is otherwise local, so galaxy 19 is odd when it starts at 0
            and even when it starts a shard — the same corpus-size dependence
            as the threshold, by a different route. Pass the shard's global
            start whenever the population is sharded.
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

    if outlier_threshold is None:
        outlier_threshold = np.percentile(residuals, 90)

    for idx in selected:
        mags = magnitudes[idx]
        z_pred = float(predictions[idx])
        z_true = float(true_redshifts[idx])
        residual = float(residuals[idx])
        band_str = _format_bands(mags, resolved_bands)

        # Choose template based on residual — outliers get outlier_analysis
        if residual > outlier_threshold:
            template = resolved_templates.get("outlier_analysis", "")
            tasks.append(template.format(
                z_pred=z_pred, z_true=z_true, residual=residual,
                band_str=band_str,
            ))
        elif (index_offset + idx) % 2 == 0:
            template = resolved_templates.get("redshift_analysis", "")
            tasks.append(template.format(
                z_pred=z_pred, z_true=z_true, band_str=band_str,
            ))
        else:
            template = resolved_templates.get("photometry_classification", "")
            tasks.append(template.format(
                z_pred=z_pred, band_str=band_str,
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
        feature_min = float(np.min(magnitudes))
        feature_max = float(np.max(magnitudes))
        tasks.append(template.format(
            n=n_samples,
            time_s=metrics.get("total_time_s", 0),
            throughput=metrics.get("throughput_bps", 0) / 1e9,
            batch_size=metrics.get("batch_size", 512),
            feature_range=f"{feature_min:.1f}-{feature_max:.1f}",
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
    outlier_threshold=None,
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
        outlier_threshold: Residual above which a galaxy gets the outlier
            template. ``None`` (default) derives it from THIS call's residuals,
            which makes the prompt for a given galaxy depend on which other
            galaxies were passed alongside it — so a sharded run gives the same
            galaxy different prompts at different world sizes. Pass a fixed
            value to make prompts a property of the galaxy alone, which any
            experiment that varies the shard count must do or its workload
            moves with the independent variable.
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
        outlier_threshold=outlier_threshold,
    )

    return tasks, results