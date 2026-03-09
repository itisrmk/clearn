"""Shared helpers for clearn."""

from __future__ import annotations

import torch


def get_device(model: torch.nn.Module) -> torch.device:
    """Get the device of a model's parameters.

    Args:
        model: A PyTorch module.

    Returns:
        The device of the first parameter, or CPU if model has no parameters.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def generate_task_id(task_history: list[str]) -> str:
    """Generate the next auto-incremented task ID.

    Args:
        task_history: List of existing task IDs.

    Returns:
        A string like "task_0", "task_1", etc.
    """
    return f"task_{len(task_history)}"
