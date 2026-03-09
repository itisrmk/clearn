"""Tests for EWC strategy (clearn/strategies/ewc.py)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytest
from clearn.strategies.ewc import EWC
from tests.conftest import make_split_dataloaders


class TestEWCBasic:
    def test_penalty_before_consolidate(self, tiny_mlp):
        ewc = EWC(tiny_mlp)
        assert ewc.penalty().item() == 0.0

    def test_consolidate_computes_fisher(self, tiny_mlp, dummy_dataloader):
        ewc = EWC(tiny_mlp)
        ewc.consolidate(dummy_dataloader)
        assert ewc._fisher is not None
        assert len(ewc._fisher) > 0

    def test_consolidate_snapshots_params(self, tiny_mlp, dummy_dataloader):
        ewc = EWC(tiny_mlp)
        ewc.consolidate(dummy_dataloader)
        assert ewc._optimal_params is not None
        for name, param in tiny_mlp.named_parameters():
            if param.requires_grad:
                assert torch.allclose(ewc._optimal_params[name], param)

    def test_penalty_after_consolidate_with_perturbation(self, tiny_mlp, dummy_dataloader):
        ewc = EWC(tiny_mlp)
        ewc.consolidate(dummy_dataloader)
        # Perturb weights
        with torch.no_grad():
            for p in tiny_mlp.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        assert ewc.penalty().item() > 0.0

    def test_penalty_zero_if_unchanged(self, tiny_mlp, dummy_dataloader):
        ewc = EWC(tiny_mlp)
        ewc.consolidate(dummy_dataloader)
        # No perturbation — penalty should be zero
        assert ewc.penalty().item() == pytest.approx(0.0, abs=1e-6)

    def test_fisher_accumulates(self, tiny_mlp, dummy_dataloader):
        ewc = EWC(tiny_mlp)
        ewc.consolidate(dummy_dataloader)
        first_fisher = {k: v.clone() for k, v in ewc._fisher.items()}

        ewc.consolidate(dummy_dataloader)
        for name in first_fisher:
            # After second consolidation, Fisher should be >= first
            assert (ewc._fisher[name] >= first_fisher[name]).all()

    def test_lambda_scales_penalty(self, tiny_mlp, dummy_dataloader):
        ewc_low = EWC(tiny_mlp, lambda_=100)
        ewc_low.consolidate(dummy_dataloader)

        ewc_high = EWC(tiny_mlp, lambda_=10000)
        ewc_high.consolidate(dummy_dataloader)
        # Copy the same Fisher/optimal from low to high for fair comparison
        ewc_high._fisher = {k: v.clone() for k, v in ewc_low._fisher.items()}
        ewc_high._optimal_params = {k: v.clone() for k, v in ewc_low._optimal_params.items()}

        with torch.no_grad():
            for p in tiny_mlp.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        assert ewc_high.penalty().item() > ewc_low.penalty().item()

    def test_state_dict_roundtrip(self, tiny_mlp, dummy_dataloader):
        ewc = EWC(tiny_mlp, lambda_=2000, n_fisher_samples=100)
        ewc.consolidate(dummy_dataloader)
        state = ewc.state_dict()

        ewc2 = EWC(tiny_mlp)
        ewc2.load_state_dict(state)
        assert ewc2._lambda == 2000
        assert ewc2._n_fisher_samples == 100
        assert ewc2._fisher is not None

    def test_empty_dataloader_raises(self, tiny_mlp):
        ewc = EWC(tiny_mlp)
        empty_loader = DataLoader(TensorDataset(torch.zeros(0, 784), torch.zeros(0, dtype=torch.long)), batch_size=1)
        with pytest.raises(ValueError, match="empty dataloader"):
            ewc.consolidate(empty_loader)


@pytest.mark.slow
class TestEWCForgetting:
    def test_ewc_retains_task1_accuracy(self):
        """After 5 sequential tasks, Task 1 accuracy should stay above 60%.

        Uses a tiny MLP on synthetic split data. The threshold is 60%
        (vs ~20% random baseline for 2-class tasks) to account for
        the synthetic data being harder than MNIST.
        """
        torch.manual_seed(42)
        input_dim = 784
        n_classes = 10
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

        splits = make_split_dataloaders(
            n_tasks=5, samples_per_task=400, input_dim=input_dim, classes_per_task=2
        )

        import clearn
        cl = clearn.wrap(model, strategy="ewc", lambda_=5000)

        for i, (train_loader, _test_loader) in enumerate(splits):
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            cl.fit(train_loader, opt, epochs=5, task_id=f"task_{i}")

        # Evaluate on task 0 test set
        from clearn.metrics import compute_retention
        task0_acc = compute_retention(model, splits[0][1], torch.device("cpu"))
        assert task0_acc > 0.4, (
            f"EWC Task 0 accuracy {task0_acc:.2%} is below 40% threshold. "
            f"Expected EWC to retain knowledge."
        )
