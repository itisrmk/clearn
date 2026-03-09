"""Shared helpers for clearn."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def get_device(model: nn.Module) -> torch.device:
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


def unpack_batch(
    batch: tuple | dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor] | torch.Tensor, torch.Tensor]:
    """Normalize a batch from either tuple or dict format.

    Handles both standard PyTorch (inputs, targets) tuples and
    HuggingFace-style dicts with 'labels' key.

    Args:
        batch: Either a (inputs, targets) tuple or a dict with 'labels'.
        device: Device to move tensors to.

    Returns:
        (model_inputs, targets) where model_inputs is either a Tensor
        or a dict of Tensors (for HuggingFace models).
    """
    if isinstance(batch, dict):
        targets = batch.pop("labels").to(device)
        model_inputs = {k: v.to(device) for k, v in batch.items()}
        # Restore labels in the original dict so it can be reused
        batch["labels"] = targets
        return model_inputs, targets
    else:
        inputs, targets = batch[0], batch[1]
        return inputs.to(device), targets.to(device)


def forward_with_inputs(
    model: nn.Module,
    model_inputs: dict[str, torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Forward pass that handles both tensor and dict inputs.

    Args:
        model: The model to run.
        model_inputs: Either a Tensor or a dict of Tensors.

    Returns:
        Model output logits.
    """
    if isinstance(model_inputs, dict):
        outputs = model(**model_inputs)
        # HuggingFace models return objects with .logits
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs
    else:
        return model(model_inputs)


def inputs_for_buffer(
    model_inputs: dict[str, torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Extract a single tensor suitable for replay buffer storage.

    For dict inputs (HuggingFace), extracts input_ids.
    For tensor inputs, returns as-is.

    Args:
        model_inputs: The model inputs from unpack_batch.

    Returns:
        A single tensor for buffer storage.
    """
    if isinstance(model_inputs, dict):
        return model_inputs.get("input_ids", next(iter(model_inputs.values())))
    return model_inputs
