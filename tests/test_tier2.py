"""Tests for Tier 2 improvements: TrainingMetrics, diagnostics, callbacks."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytest
import clearn
from clearn.metrics import TrainingMetrics
from clearn.callbacks import ContinualCallback, EarlyStoppingCallback


class TestTrainingMetrics:
    def test_fit_returns_metrics(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        metrics = cl.fit(dummy_dataloader, opt, task_id="t1", epochs=3)
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.task_id == "t1"
        assert metrics.epochs == 3
        assert len(metrics.epoch_losses) == 3
        assert len(metrics.epoch_accuracies) == 3

    def test_metrics_loss_decreases(self, tiny_mlp):
        """With enough epochs, loss should generally decrease."""
        # Create a trivially learnable dataset
        torch.manual_seed(42)
        centroids = torch.randn(2, 784) * 3
        X = torch.cat([centroids[0].unsqueeze(0).expand(100, -1) + torch.randn(100, 784) * 0.1,
                       centroids[1].unsqueeze(0).expand(100, -1) + torch.randn(100, 784) * 0.1])
        y = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(100, dtype=torch.long)])
        loader = DataLoader(TensorDataset(X, y), batch_size=32)

        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.Adam(tiny_mlp.parameters(), lr=0.01)
        metrics = cl.fit(loader, opt, epochs=10, task_id="learn")
        # Last epoch loss should be less than first
        assert metrics.epoch_losses[-1] < metrics.epoch_losses[0]

    def test_metrics_repr(self):
        m = TrainingMetrics(
            task_id="test", epochs=5, final_loss=0.5, final_accuracy=0.9, wall_time=1.5
        )
        text = repr(m)
        assert "test" in text
        assert "0.5000" in text
        assert "90.00%" in text

    def test_metrics_wall_time(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        metrics = cl.fit(dummy_dataloader, opt)
        assert metrics.wall_time >= 0


class TestDiagnostics:
    def test_ewc_diagnostics(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp, strategy="ewc")
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt)
        diag = cl.diagnostics()
        assert diag["strategy"] == "ewc"
        assert diag["consolidated"] is True
        assert "fisher_mean" in diag
        assert diag["tasks_trained"] == 1
        assert "lambda" in diag

    def test_der_diagnostics(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp, strategy="der")
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt)
        diag = cl.diagnostics()
        assert diag["strategy"] == "der++"
        assert diag["buffer_used"] > 0
        assert "buffer_class_distribution" in diag
        assert diag["tasks_trained"] == 1

    def test_ewc_diagnostics_before_training(self, tiny_mlp):
        cl = clearn.wrap(tiny_mlp, strategy="ewc")
        diag = cl.diagnostics()
        assert diag["consolidated"] is False
        assert diag["tasks_trained"] == 0

    def test_diagnostics_includes_task_history(self, tiny_mlp, dummy_dataloader):
        cl = clearn.wrap(tiny_mlp, strategy="ewc")
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt, task_id="a")
        cl.fit(dummy_dataloader, opt, task_id="b")
        diag = cl.diagnostics()
        assert diag["task_history"] == ["a", "b"]


class TestCallbacks:
    def test_on_task_start_called(self, tiny_mlp, dummy_dataloader):
        calls = []

        class TrackCallback(ContinualCallback):
            def on_task_start(self, model, task_id):
                calls.append(("start", task_id))

        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt, task_id="t1", callbacks=[TrackCallback()])
        assert ("start", "t1") in calls

    def test_on_task_end_called(self, tiny_mlp, dummy_dataloader):
        calls = []

        class TrackCallback(ContinualCallback):
            def on_task_end(self, model, task_id, metrics):
                calls.append(("end", task_id, metrics))

        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt, task_id="t1", callbacks=[TrackCallback()])
        assert len(calls) == 1
        assert calls[0][0] == "end"
        assert isinstance(calls[0][2], TrainingMetrics)

    def test_on_batch_end_called(self, tiny_mlp, dummy_dataloader):
        batch_count = []

        class BatchCounter(ContinualCallback):
            def on_batch_end(self, model, loss):
                batch_count.append(loss)

        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt, callbacks=[BatchCounter()])
        # 100 samples / 32 batch_size = ~4 batches
        assert len(batch_count) >= 3

    def test_multiple_callbacks(self, tiny_mlp, dummy_dataloader):
        starts = []
        ends = []

        class CB1(ContinualCallback):
            def on_task_start(self, model, task_id):
                starts.append(task_id)

        class CB2(ContinualCallback):
            def on_task_end(self, model, task_id, metrics):
                ends.append(task_id)

        cl = clearn.wrap(tiny_mlp)
        opt = torch.optim.SGD(tiny_mlp.parameters(), lr=0.01)
        cl.fit(dummy_dataloader, opt, task_id="x", callbacks=[CB1(), CB2()])
        assert starts == ["x"]
        assert ends == ["x"]

    def test_early_stopping_callback(self):
        cb = EarlyStoppingCallback(patience=3)
        assert not cb.stopped
        cb.on_task_start(None, "t1")
        cb.on_batch_end(None, 1.0)
        cb.on_batch_end(None, 1.0)
        cb.on_batch_end(None, 1.0)
        assert not cb.stopped  # 3 = patience, not yet
        cb.on_batch_end(None, 1.0)
        assert cb.stopped  # 4 > patience
