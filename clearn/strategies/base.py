"""Abstract base class for continual learning strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseStrategy(ABC):
    """Base class for all continual learning strategies.

    All strategies must implement `consolidate()` and `penalty()`.
    Replay-based strategies should also override `update_buffer()`
    and `get_replay_loss()`.

    Args:
        model: The PyTorch model to protect from forgetting.
    """

    def __init__(self, model: nn.Module, **kwargs: Any) -> None:
        self.model = model

    @abstractmethod
    def consolidate(self, dataloader: DataLoader) -> None:
        """Lock in knowledge after completing a task.

        Called automatically by `ContinualModel.fit()` after training.

        Args:
            dataloader: The training data for the task that just completed.
        """

    @abstractmethod
    def penalty(self) -> torch.Tensor:
        """Compute the regularization penalty for the current parameter state.

        Returns:
            A scalar tensor to be added to the task loss during training.
        """

    def update_buffer(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> None:
        """Store samples in a replay buffer.

        Override for replay-based strategies. Default is a no-op.

        Args:
            inputs: Input tensor from the current batch.
            targets: Target tensor from the current batch.
            logits: Model output logits (used by DER++ for logit matching).
        """

    def get_replay_loss(
        self, model: nn.Module, loss_fn: nn.Module
    ) -> torch.Tensor:
        """Compute the replay loss from buffered samples.

        Override for replay-based strategies. Default returns zero.

        Args:
            model: The model to evaluate replay samples on.
            loss_fn: The loss function to use for replay.

        Returns:
            A scalar tensor representing the replay loss.
        """
        return torch.tensor(0.0, device=next(model.parameters()).device)

    def state_dict(self) -> dict[str, Any]:
        """Return strategy-specific state for serialization.

        Returns:
            A dictionary of state to be saved.
        """
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore strategy state from a saved dictionary.

        Args:
            state: The state dictionary to restore from.
        """
