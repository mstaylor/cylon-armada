"""Tests for cosmic-ai task generator."""

import os
import sys
import json
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'shared', 'scripts'))

from cosmic_ai.task_generator import (
    generate_tasks_from_results,
    load_config,
    _resolve_config,
    _format_bands,
    _format_colors,
    _DEFAULT_TEMPLATES,
    _DEFAULT_SURVEY_TYPES,
    BANDS,
)


@pytest.fixture
def sample_data():
    """Generate sample inference data."""
    np.random.seed(42)
    n = 20
    return {
        "predictions": np.random.uniform(0.0, 1.5, n),
        "true_redshifts": np.random.uniform(0.0, 1.5, n),
        "magnitudes": np.random.uniform(17.0, 25.0, (n, 5)),
        "metrics": {
            "total_time_s": 2.5,
            "num_samples": n,
            "batch_size": 512,
            "throughput_bps": 1.3e9,
        },
    }


class TestFormatHelpers:
    def test_format_bands(self):
        mags = [22.31, 21.08, 20.45, 19.82, 19.24]
        result = _format_bands(mags)
        assert "u=22.31" in result
        assert "z=19.24" in result
        assert result.count(",") == 4

    def test_format_colors(self):
        mags = [22.0, 21.0, 20.0, 19.0, 18.0]
        result = _format_colors(mags)
        assert "u-g=1.00" in result
        assert "i-z=1.00" in result
        assert result.count(",") == 3

    def test_format_bands_custom_bands(self):
        mags = [1.0, 2.0, 3.0]
        result = _format_bands(mags, bands=("a", "b", "c"))
        assert "a=1.00" in result
        assert "c=3.00" in result


class TestGenerateTasksFromResults:
    def test_generates_tasks(self, sample_data):
        tasks = generate_tasks_from_results(**sample_data)
        assert len(tasks) > 0
        assert all(isinstance(t, str) for t in tasks)

    def test_max_tasks_limit(self, sample_data):
        tasks = generate_tasks_from_results(**sample_data, max_tasks=5)
        # Per-sample tasks limited + batch-level tasks added
        assert len(tasks) <= 10  # 5 per-sample + some batch

    def test_seed_reproducibility(self, sample_data):
        tasks1 = generate_tasks_from_results(**sample_data, max_tasks=8, seed=42)
        tasks2 = generate_tasks_from_results(**sample_data, max_tasks=8, seed=42)
        assert tasks1 == tasks2

    def test_different_seeds_different_results(self, sample_data):
        tasks1 = generate_tasks_from_results(**sample_data, max_tasks=8, seed=42)
        tasks2 = generate_tasks_from_results(**sample_data, max_tasks=8, seed=99)
        assert tasks1 != tasks2

    def test_no_metrics_skips_batch_tasks(self, sample_data):
        sample_data["metrics"] = None
        tasks = generate_tasks_from_results(**sample_data)
        # No batch_summary or cost_analysis tasks without metrics
        for t in tasks:
            assert "batch of" not in t
            assert "serverless inference run" not in t

    def test_includes_outlier_tasks(self, sample_data):
        # Make one prediction a clear outlier
        sample_data["predictions"][0] = 5.0
        sample_data["true_redshifts"][0] = 0.1
        tasks = generate_tasks_from_results(**sample_data)
        outlier_tasks = [t for t in tasks if "prediction error" in t or "residual" in t]
        assert len(outlier_tasks) > 0

    def test_custom_templates(self, sample_data):
        custom = {
            "redshift_analysis": "CUSTOM: z={z_pred:.3f} for {band_str}",
            "color_classification": "CUSTOM: colors {color_str}",
            "outlier_analysis": "CUSTOM: outlier z={z_pred:.3f}",
            "batch_summary": "CUSTOM: batch {n}",
            "cost_analysis": "CUSTOM: cost {n}",
        }
        tasks = generate_tasks_from_results(**sample_data, templates=custom)
        assert any(t.startswith("CUSTOM:") for t in tasks)


class TestConfigResolution:
    def test_defaults(self):
        templates, survey_types, bands = _resolve_config()
        assert templates == _DEFAULT_TEMPLATES
        assert survey_types == _DEFAULT_SURVEY_TYPES
        assert bands == BANDS

    def test_direct_override(self):
        custom = {"redshift_analysis": "custom template"}
        templates, _, _ = _resolve_config(templates=custom)
        assert templates["redshift_analysis"] == "custom template"

    def test_config_file(self):
        config_data = {
            "templates": {"redshift_analysis": "file template"},
            "survey_types": ["custom survey"],
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            templates, survey_types, _ = _resolve_config(config_path=f.name)

        assert templates["redshift_analysis"] == "file template"
        # Other templates should still have defaults
        assert "color_classification" in templates
        assert survey_types == ["custom survey"]
        os.unlink(f.name)

    def test_param_overrides_file(self):
        config_data = {"templates": {"redshift_analysis": "file template"}}
        direct = {"redshift_analysis": "direct template"}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            templates, _, _ = _resolve_config(templates=direct, config_path=f.name)

        # Direct parameter should win
        assert templates["redshift_analysis"] == "direct template"
        os.unlink(f.name)

    def test_env_var_config_path(self, monkeypatch):
        config_data = {"survey_types": ["env survey"]}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            monkeypatch.setenv("COSMIC_AI_CONFIG", f.name)
            _, survey_types, _ = _resolve_config()

        assert survey_types == ["env survey"]
        os.unlink(f.name)