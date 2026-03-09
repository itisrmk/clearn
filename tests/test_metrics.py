"""Tests for RetentionReport (clearn/metrics.py)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytest
from clearn.metrics import RetentionReport, _get_recommendation, compute_retention


class TestRetentionReport:
    def test_repr_contains_box_drawing(self):
        report = RetentionReport(
            task_scores={"task_a": 94.2, "task_b": 100.0},
            plasticity_score=0.87,
            stability_score=0.94,
            recommendation="stable \u2014 no action needed",
        )
        text = repr(report)
        assert "\u251c\u2500\u2500" in text  # ├──
        assert "\u2514\u2500\u2500" in text  # └──
        assert "task_a" in text
        assert "94.2%" in text

    def test_repr_contains_all_fields(self):
        report = RetentionReport(
            task_scores={"t1": 80.0},
            plasticity_score=0.8,
            stability_score=0.9,
            recommendation="stable",
        )
        text = repr(report)
        assert "plasticity_score" in text
        assert "stability_score" in text
        assert "recommendation" in text
        assert "RetentionReport" in text


class TestRecommendation:
    def test_stable(self):
        assert "stable" in _get_recommendation(0.95)

    def test_mild_forgetting(self):
        assert "mild" in _get_recommendation(0.75)

    def test_significant_forgetting(self):
        assert "significant" in _get_recommendation(0.55)

    def test_severe_forgetting(self):
        assert "severe" in _get_recommendation(0.3)

    def test_boundary_90(self):
        assert "stable" in _get_recommendation(0.9)

    def test_boundary_70(self):
        assert "mild" in _get_recommendation(0.7)

    def test_boundary_50(self):
        assert "significant" in _get_recommendation(0.5)


class TestComputeRetention:
    def test_perfect_accuracy(self):
        """Model that always predicts correctly should get 1.0."""
        # Create a trivially correct model
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        # Data where input = one-hot, target = argmax
        X = torch.eye(2).repeat(10, 1)  # [1,0] and [0,1] repeated
        y = torch.tensor([0, 1] * 10)
        loader = DataLoader(TensorDataset(X, y), batch_size=4)
        acc = compute_retention(model, loader, torch.device("cpu"))
        assert acc == pytest.approx(1.0)

    def test_empty_dataloader(self):
        model = nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(torch.zeros(0, 2), torch.zeros(0, dtype=torch.long)),
            batch_size=1,
        )
        acc = compute_retention(model, loader, torch.device("cpu"))
        assert acc == 0.0
