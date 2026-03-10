"""clearn — Continual learning for PyTorch models. Wrap once. Train forever."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from clearn.callbacks import ContinualCallback, EarlyStoppingCallback
from clearn.core import ContinualModel
from clearn.metrics import TrainingMetrics

__version__ = "0.3.0"


def wrap(
    model: nn.Module, strategy: str = "ewc", **kwargs: Any
) -> ContinualModel:
    """Wrap a PyTorch model with continual learning protection.

    Args:
        model: Any PyTorch nn.Module.
        strategy: Strategy name ("ewc", "der") or a BaseStrategy instance.
        **kwargs: Strategy-specific keyword arguments.

    Returns:
        A ContinualModel wrapping the provided model.
    """
    return ContinualModel.wrap(model, strategy, **kwargs)


def load(path: str, model: nn.Module | None = None) -> ContinualModel:
    """Load a saved ContinualModel from disk.

    Args:
        path: Directory path containing the checkpoint.
        model: The model architecture to load weights into.

    Returns:
        A restored ContinualModel.
    """
    return ContinualModel.load(path, model)


try:
    from clearn.integrations import from_pretrained
except ImportError:
    from clearn.integrations import from_pretrained  # fallback stub

__all__ = [
    "wrap",
    "load",
    "from_pretrained",
    "ContinualModel",
    "ContinualCallback",
    "EarlyStoppingCallback",
    "TrainingMetrics",
    "__version__",
]
