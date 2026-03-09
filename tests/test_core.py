"""Tests for ContinualModel (clearn/core.py)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytest
import clearn
from clearn.core import ContinualModel
from clearn.metrics import RetentionReport
from clearn.strategies.ewc import EWC


class TestWrap:
    def test_wrap_creates_continual_model(self, tiny_mlp):
        cl = clearn.wrap(tiny_mlp)
        assert isinstance(cl, ContinualModel)

    def test_wrap_with_string_strategy(self, tiny_mlp):
        cl = clearn.wrap(tiny_mlp, strategy="ewc")
        assert isinstance(cl.strategy, EWC)

    def test_wrap_with_strategy_instance(self, tiny_mlp):
        strategy = EWC(tiny_mlp, lambda_=1000)
        cl = clearn.wrap(tiny_mlp, strategy=strategy)
        assert cl.strategy is strategy

    def test_wrap_invalid_strategy_raises(self, tiny_mlp):
        with pytest.raises(ValueError, match="Unknown strategy"):
            clearn.wrap(tiny_mlp, strategy="nonexistent")

    def test_wrap_default_strategy_is_ewc(self, tiny_mlp):
        cl = clearn.wrap(tiny_mlp)
        assert isinstance(cl.strategy, EWC)


class TestFit:
    def test_fit_basic(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt)  # Should not raise

    def test_fit_auto_task_id(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt)
        assert cl._task_history == ["task_0"]

    def test_fit_custom_task_id(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt, task_id="my_task")
        assert "my_task" in cl._task_history

    def test_fit_returns_self(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        result = cl.fit(dummy_dataloader, opt)
        assert result is cl

    def test_fit_multiple_tasks(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt, task_id="t1")
        cl.fit(dummy_dataloader, opt, task_id="t2")
        cl.fit(dummy_dataloader, opt, task_id="t3")
        assert len(cl._task_history) == 3


class TestDiff:
    def test_diff_before_fit_raises(self, tiny_mlp):
        cl = clearn.wrap(tiny_mlp)
        with pytest.raises(RuntimeError, match="Call fit\\(\\) first"):
            cl.diff()

    def test_diff_returns_retention_report(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt)
        report = cl.diff()
        assert isinstance(report, RetentionReport)

    def test_diff_after_one_task(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt)
        report = cl.diff()
        assert report.stability_score == 1.0  # No past tasks
        assert report.plasticity_score >= 0.0

    def test_diff_has_all_task_scores(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt, task_id="a")
        cl.fit(dummy_dataloader, opt, task_id="b")
        report = cl.diff()
        assert "a" in report.task_scores
        assert "b" in report.task_scores


class TestSaveLoad:
    def test_save_and_load(self, tiny_mlp, dummy_dataloader, tmp_path):
        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt, task_id="saved_task")

        save_dir = str(tmp_path / "ckpt")
        cl.save(save_dir)

        model2 = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
        cl2 = clearn.load(save_dir, model=model2)

        assert cl2._task_history == ["saved_task"]
        # Verify weights match
        for p1, p2 in zip(tiny_mlp.parameters(), model2.parameters()):
            assert torch.allclose(p1.cpu(), p2.cpu())

    def test_load_without_model_raises(self, tiny_mlp, dummy_dataloader, tmp_path):
        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt)
        cl.save(str(tmp_path / "ckpt"))

        with pytest.raises(ValueError, match="Must provide the model"):
            clearn.load(str(tmp_path / "ckpt"))


class TestDelegation:
    def test_forward_delegation(self, tiny_mlp):
        cl = clearn.wrap(tiny_mlp)
        x = torch.randn(5, 784)
        out = cl(x)
        assert out.shape == (5, 10)

    def test_eval_delegation(self, tiny_mlp):
        cl = clearn.wrap(tiny_mlp)
        cl.eval()
        assert not tiny_mlp.training

    def test_train_delegation(self, tiny_mlp):
        cl = clearn.wrap(tiny_mlp)
        cl.eval()
        cl.train()
        assert tiny_mlp.training
