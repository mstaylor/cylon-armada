"""Tests for ONNX export and model parallelism partitioning."""

import os
import sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'experiments'))

from cosmic_ai.export_onnx import estimate_memory


class MockModel(nn.Module):
    """Simplified mock of AstroMAE for testing memory estimation."""
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(100, 768)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.randn(1, 197, 768))
        self.blocks = nn.ModuleList([nn.Linear(768, 768) for _ in range(4)])
        self.fc_norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, 768)
        self.vit_block = nn.Sequential(nn.Linear(768, 1096), nn.ReLU(), nn.Linear(1096, 1096))
        self.inception_model = nn.Sequential(nn.Linear(100, 2120))
        self.concat_block = nn.Sequential(nn.Linear(3216, 1024), nn.ReLU(), nn.Linear(1024, 1))

    def forward(self, x):
        return self.concat_block(torch.randn(1, 3216))


class TestMemoryEstimation:
    def test_returns_all_stages(self):
        model = MockModel()
        report = estimate_memory(model)

        assert "total_parameters" in report
        assert "total_param_mb" in report
        assert "stages" in report
        assert "stage_0_vit" in report["stages"]
        assert "stage_1_inception" in report["stages"]
        assert "stage_2_fusion" in report["stages"]

    def test_stage_percentages_sum_to_100(self):
        model = MockModel()
        report = estimate_memory(model)

        total_pct = sum(info["param_pct"] for info in report["stages"].values())
        assert total_pct == pytest.approx(100.0, abs=1.0)

    def test_recommended_lambda_memory_is_valid(self):
        model = MockModel()
        report = estimate_memory(model)

        for stage_name, mem_mb in report["recommended_lambda_memory"].items():
            assert mem_mb >= 256
            assert mem_mb % 256 == 0

    def test_vit_stage_is_largest(self):
        model = MockModel()
        report = estimate_memory(model)

        vit = report["stages"]["stage_0_vit"]["parameters"]
        inception = report["stages"]["stage_1_inception"]["parameters"]
        fusion = report["stages"]["stage_2_fusion"]["parameters"]

        assert vit > inception
        assert vit > fusion

    def test_total_params_matches_sum(self):
        model = MockModel()
        report = estimate_memory(model)

        stage_sum = sum(info["parameters"] for info in report["stages"].values())
        assert stage_sum == report["total_parameters"]

    def test_estimated_peak_mb_is_positive(self):
        model = MockModel()
        report = estimate_memory(model)

        for info in report["stages"].values():
            assert info["estimated_peak_mb"] >= 0

    def test_empty_stage_handled(self):
        """Stages with no matching children should report 0 params."""
        model = nn.Linear(10, 10)  # No named children matching stage names
        report = estimate_memory(model)

        for info in report["stages"].values():
            assert info["parameters"] == 0