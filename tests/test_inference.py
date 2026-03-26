"""Tests for cosmic-ai inference module."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'shared', 'scripts'))

from cosmic_ai.inference import compute_metrics


class TestComputeMetrics:
    def test_perfect_predictions(self):
        true_z = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        pred_z = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        metrics = compute_metrics(pred_z, true_z)

        assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["mse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["bias"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)

    def test_known_error(self):
        true_z = np.array([0.1, 0.2, 0.3])
        pred_z = np.array([0.2, 0.3, 0.4])

        metrics = compute_metrics(pred_z, true_z)

        assert metrics["mae"] == pytest.approx(0.1, abs=1e-6)
        assert metrics["mse"] == pytest.approx(0.01, abs=1e-6)
        assert metrics["bias"] > 0  # Predictions are systematically high

    def test_returns_all_keys(self):
        true_z = np.random.uniform(0, 1, 50)
        pred_z = true_z + np.random.normal(0, 0.05, 50)

        metrics = compute_metrics(pred_z, true_z)

        assert "mae" in metrics
        assert "mse" in metrics
        assert "bias" in metrics
        assert "precision_nmad" in metrics
        assert "r2" in metrics

    def test_nmad_robust_to_outliers(self):
        """NMAD should be more robust than standard deviation to outliers."""
        true_z = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        # One massive outlier
        pred_z = np.array([0.1, 0.2, 0.3, 0.4, 5.0])

        metrics = compute_metrics(pred_z, true_z)

        # NMAD should be much smaller than MAE due to median-based robustness
        assert metrics["precision_nmad"] < metrics["mae"]

    def test_r2_negative_for_bad_predictions(self):
        """R2 should be negative when predictions are worse than mean."""
        true_z = np.array([0.1, 0.2, 0.3])
        pred_z = np.array([1.0, 2.0, 3.0])  # Terrible predictions

        metrics = compute_metrics(pred_z, true_z)
        assert metrics["r2"] < 0