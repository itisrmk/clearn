"""Tests for DER++ strategy (clearn/strategies/der.py)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytest
from clearn.strategies.der import DER
from tests.conftest import make_split_dataloaders


class TestDERBasic:
    def test_buffer_starts_empty(self, tiny_mlp):
        der = DER(tiny_mlp, buffer_size=50)
        assert len(der._buffer_inputs) == 0

    def test_update_buffer_adds_samples(self, tiny_mlp):
        der = DER(tiny_mlp, buffer_size=50)
        inputs = torch.randn(10, 784)
        targets = torch.randint(0, 10, (10,))
        logits = torch.randn(10, 10)
        der.update_buffer(inputs, targets, logits)
        assert len(der._buffer_inputs) == 10

    def test_reservoir_sampling_caps_at_buffer_size(self, tiny_mlp):
        der = DER(tiny_mlp, buffer_size=20)
        for _ in range(10):
            inputs = torch.randn(10, 784)
            targets = torch.randint(0, 10, (10,))
            logits = torch.randn(10, 10)
            der.update_buffer(inputs, targets, logits)
        assert len(der._buffer_inputs) == 20

    def test_replay_loss_empty_buffer(self, tiny_mlp):
        der = DER(tiny_mlp, buffer_size=50)
        loss = der.get_replay_loss(tiny_mlp, nn.CrossEntropyLoss())
        assert loss.item() == 0.0

    def test_replay_loss_nonempty_buffer(self, tiny_mlp):
        der = DER(tiny_mlp, buffer_size=50)
        inputs = torch.randn(20, 784)
        targets = torch.randint(0, 10, (20,))
        with torch.no_grad():
            logits = tiny_mlp(inputs)
        der.update_buffer(inputs, targets, logits)
        loss = der.get_replay_loss(tiny_mlp, nn.CrossEntropyLoss())
        assert loss.item() >= 0.0  # Should be non-negative

    def test_penalty_is_zero(self, tiny_mlp):
        der = DER(tiny_mlp)
        assert der.penalty().item() == 0.0

    def test_consolidate_does_not_crash(self, tiny_mlp, dummy_dataloader):
        der = DER(tiny_mlp, buffer_size=50)
        der.consolidate(dummy_dataloader)  # Should not raise

    def test_update_buffer_without_logits_is_noop(self, tiny_mlp):
        der = DER(tiny_mlp, buffer_size=50)
        inputs = torch.randn(10, 784)
        targets = torch.randint(0, 10, (10,))
        der.update_buffer(inputs, targets)  # No logits
        assert len(der._buffer_inputs) == 0

    def test_state_dict_roundtrip(self, tiny_mlp):
        der = DER(tiny_mlp, buffer_size=30, alpha=0.2, beta=0.3)
        inputs = torch.randn(10, 784)
        targets = torch.randint(0, 10, (10,))
        logits = torch.randn(10, 10)
        der.update_buffer(inputs, targets, logits)

        state = der.state_dict()
        der2 = DER(tiny_mlp)
        der2.load_state_dict(state)
        assert der2._buffer_size == 30
        assert der2._alpha == 0.2
        assert der2._beta == 0.3
        assert len(der2._buffer_inputs) == 10


@pytest.mark.slow
class TestDERForgetting:
    def test_der_retains_task1_accuracy(self):
        """After 5 sequential tasks, Task 1 accuracy should be reasonable."""
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
        cl = clearn.wrap(model, strategy="der", buffer_size=200, alpha=0.5, beta=1.0)

        for i, (train_loader, _test_loader) in enumerate(splits):
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            cl.fit(train_loader, opt, epochs=5, task_id=f"task_{i}")

        from clearn.metrics import compute_retention
        task0_acc = compute_retention(model, splits[0][1], torch.device("cpu"))
        assert task0_acc > 0.3, (
            f"DER Task 0 accuracy {task0_acc:.2%} is below 30% threshold."
        )
