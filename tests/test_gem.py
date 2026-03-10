"""Tests for GEM (Gradient Episodic Memory) strategy."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytest
import clearn
from clearn.strategies.gem import GEM
from tests.conftest import make_split_dataloaders


class TestGEMBasic:
    def test_penalty_is_zero(self, tiny_mlp):
        gem = GEM(tiny_mlp)
        assert gem.penalty().item() == 0.0

    def test_consolidate_stores_memory(self, tiny_mlp, dummy_dataloader):
        gem = GEM(tiny_mlp, memory_size=50)
        gem.set_task_id("t1")
        gem.consolidate(dummy_dataloader)
        assert "t1" in gem._memories
        assert gem._memories["t1"][0].size(0) <= 50

    def test_memory_size_limit(self, tiny_mlp):
        X = torch.randn(200, 784)
        y = torch.randint(0, 10, (200,))
        loader = DataLoader(TensorDataset(X, y), batch_size=32)
        gem = GEM(tiny_mlp, memory_size=50)
        gem.set_task_id("t1")
        gem.consolidate(loader)
        assert gem._memories["t1"][0].size(0) == 50

    def test_gradient_projection_does_not_crash(self, tiny_mlp, dummy_dataloader):
        gem = GEM(tiny_mlp, memory_size=50)
        gem.set_task_id("t1")
        gem.consolidate(dummy_dataloader)

        # Run a forward/backward pass and then project
        X = torch.randn(10, 784)
        y = torch.randint(0, 10, (10,))
        out = tiny_mlp(X)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        gem.project_gradients()  # Should not crash

    def test_wrap_with_gem(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp, strategy="gem", memory_size=50)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        metrics = cl.fit(dummy_dataloader, opt, task_id="t1")
        assert metrics.task_id == "t1"

    def test_wrap_with_agem_alias(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp, strategy="agem")
        assert isinstance(cl.strategy, GEM)

    def test_gem_state_dict_roundtrip(self, tiny_mlp, dummy_dataloader):
        gem = GEM(tiny_mlp, memory_size=30)
        gem.set_task_id("t1")
        gem.consolidate(dummy_dataloader)

        state = gem.state_dict()
        gem2 = GEM(tiny_mlp)
        gem2.load_state_dict(state)
        assert gem2._memory_size == 30
        assert "t1" in gem2._memories

    def test_gem_diagnostics(self, tiny_mlp, dummy_dataloader):
        gem = GEM(tiny_mlp, memory_size=50)
        gem.set_task_id("t1")
        gem.consolidate(dummy_dataloader)
        diag = gem.get_diagnostics()
        assert diag["strategy"] == "gem"
        assert diag["tasks_in_memory"] == 1
        assert diag["total_memory_samples"] > 0

    def test_gem_multiple_tasks(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp, strategy="gem", memory_size=30)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt, task_id="t1")
        cl.fit(dummy_dataloader, opt, task_id="t2")
        diag = cl.diagnostics()
        assert diag["tasks_in_memory"] == 2


@pytest.mark.slow
class TestGEMForgetting:
    def test_gem_retains_task1_accuracy(self):
        """After 5 sequential tasks, Task 1 should have some retention."""
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

        cl = clearn.wrap(model, strategy="gem", memory_size=100)

        for i, (train_loader, _) in enumerate(splits):
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            cl.fit(train_loader, opt, epochs=5, task_id=f"task_{i}")

        from clearn.metrics import compute_retention
        task0_acc = compute_retention(model, splits[0][1], torch.device("cpu"))
        assert task0_acc > 0.25, f"GEM Task 0 accuracy {task0_acc:.2%} below 25%"
