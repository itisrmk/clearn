"""Shared fixtures for clearn tests."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytest


@pytest.fixture
def device() -> torch.device:
    """Return the appropriate torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tiny_mlp(device: torch.device) -> nn.Module:
    """A 2-layer MLP for fast testing (784 -> 128 -> 10)."""
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)
    return model


@pytest.fixture
def dummy_dataloader() -> DataLoader:
    """Random tensor dataloader for smoke tests. 100 samples, 10 classes."""
    X = torch.randn(100, 784)
    y = torch.randint(0, 10, (100,))
    return DataLoader(TensorDataset(X, y), batch_size=32)


@pytest.fixture
def small_dataloader() -> DataLoader:
    """Smaller random dataloader (20 samples) for unit tests."""
    X = torch.randn(20, 784)
    y = torch.randint(0, 10, (20,))
    return DataLoader(TensorDataset(X, y), batch_size=10)


def make_split_dataloaders(
    n_tasks: int = 5,
    samples_per_task: int = 200,
    input_dim: int = 784,
    classes_per_task: int = 2,
) -> list[tuple[DataLoader, DataLoader]]:
    """Create synthetic split dataloaders for sequential task testing.

    Each task has distinct class labels to simulate task boundaries.

    Args:
        n_tasks: Number of sequential tasks.
        samples_per_task: Samples per task.
        input_dim: Input feature dimension.
        classes_per_task: Number of classes per task.

    Returns:
        List of (train_loader, test_loader) tuples.
    """
    loaders = []
    # Create class centroids so each class has a learnable pattern
    total_classes = n_tasks * classes_per_task
    centroids = torch.randn(total_classes, input_dim) * 3.0

    for t in range(n_tasks):
        base_class = t * classes_per_task

        def _make_data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
            xs, ys = [], []
            per_class = n // classes_per_task
            for c in range(classes_per_task):
                label = base_class + c
                x = centroids[label].unsqueeze(0) + torch.randn(per_class, input_dim) * 0.5
                xs.append(x)
                ys.append(torch.full((per_class,), label, dtype=torch.long))
            return torch.cat(xs), torch.cat(ys)

        X_train, y_train = _make_data(samples_per_task)
        X_test, y_test = _make_data(samples_per_task // 4)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
        loaders.append((train_loader, test_loader))

    return loaders
