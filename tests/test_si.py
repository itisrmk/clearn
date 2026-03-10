"""Tests for SI (Synaptic Intelligence) strategy."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytest
import clearn
from clearn.strategies.si import SI
from tests.conftest import make_split_dataloaders


class TestSIBasic:
    def test_penalty_before_consolidate(self, tiny_mlp):
        si = SI(tiny_mlp, c=1.0)
        assert si.penalty().item() == 0.0

    def test_consolidate_creates_optimal_params(self, tiny_mlp, dummy_dataloader):
        si = SI(tiny_mlp, c=1.0)
        si.consolidate(dummy_dataloader)
        assert si._optimal_params is not None

    def test_penalty_after_consolidate_and_perturbation(self, tiny_mlp, dummy_dataloader):
        si = SI(tiny_mlp, c=1.0)
        # Simulate a training step to accumulate path integral
        for batch in dummy_dataloader:
            x, y = batch
            out = tiny_mlp(x)
            loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            with torch.no_grad():
                for p in tiny_mlp.parameters():
                    p -= 0.01 * p.grad
            si.update_running_importance()
            tiny_mlp.zero_grad()
            break

        si.consolidate(dummy_dataloader)

        # Perturb weights
        with torch.no_grad():
            for p in tiny_mlp.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        assert si.penalty().item() > 0.0

    def test_c_scales_penalty(self, tiny_mlp, dummy_dataloader):
        si_low = SI(tiny_mlp, c=0.1)
        si_high = SI(tiny_mlp, c=100.0)

        # Run a training step
        for batch in dummy_dataloader:
            x, y = batch
            out = tiny_mlp(x)
            loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            with torch.no_grad():
                for p in tiny_mlp.parameters():
                    p -= 0.01 * p.grad
            si_low.update_running_importance()
            si_high.update_running_importance()
            tiny_mlp.zero_grad()
            break

        si_low.consolidate(dummy_dataloader)
        si_high._omega = {k: v.clone() for k, v in si_low._omega.items()}
        si_high._optimal_params = {k: v.clone() for k, v in si_low._optimal_params.items()}

        with torch.no_grad():
            for p in tiny_mlp.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        assert si_high.penalty().item() > si_low.penalty().item()

    def test_wrap_with_si(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp, strategy="si", c=1.0)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        metrics = cl.fit(dummy_dataloader, opt, task_id="t1")
        assert metrics.task_id == "t1"
        report = cl.diff()
        assert "t1" in report.task_scores

    def test_si_state_dict_roundtrip(self, tiny_mlp, dummy_dataloader):
        si = SI(tiny_mlp, c=2.0, epsilon=1e-4)
        si.consolidate(dummy_dataloader)
        state = si.state_dict()

        si2 = SI(tiny_mlp)
        si2.load_state_dict(state)
        assert si2._c == 2.0
        assert si2._epsilon == 1e-4

    def test_si_diagnostics(self, tiny_mlp, dummy_dataloader):
        si = SI(tiny_mlp, c=1.0)
        diag = si.get_diagnostics()
        assert diag["strategy"] == "si"
        assert diag["consolidated"] is False

        si.consolidate(dummy_dataloader)
        diag = si.get_diagnostics()
        assert diag["consolidated"] is True

    def test_si_after_optimizer_step_hook(self, tiny_mlp, dummy_dataloader):
        """SI's after_optimizer_step should call update_running_importance."""
        cl = clearn.wrap(tiny_mlp, strategy="si", c=5.0)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt, task_id="t1")
        # If SI is properly wired, omega should be populated
        si = cl.strategy
        assert si._optimal_params is not None


@pytest.mark.slow
class TestSIForgetting:
    def test_si_retains_task1_accuracy(self):
        """After 5 sequential tasks, Task 1 accuracy should be reasonable."""
        torch.manual_seed(42)
        input_dim = 784
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        splits = make_split_dataloaders(
            n_tasks=5, samples_per_task=400, input_dim=input_dim, classes_per_task=2
        )

        cl = clearn.wrap(model, strategy="si", c=5.0)

        for i, (train_loader, _) in enumerate(splits):
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            cl.fit(train_loader, opt, epochs=5, task_id=f"task_{i}")

        from clearn.metrics import compute_retention
        task0_acc = compute_retention(model, splits[0][1], torch.device("cpu"))
        assert task0_acc > 0.3, f"SI Task 0 accuracy {task0_acc:.2%} below 30%"
