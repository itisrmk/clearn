"""Dark Experience Replay++ (Buzzega et al., NeurIPS 2020).

Maintains a small memory buffer of past (input, logit, target) tuples.
During new task training, replays buffer samples and matches their original
output logits (not just labels). This preserves soft knowledge.
"""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from clearn.strategies.base import BaseStrategy


class DER(BaseStrategy):
    """Dark Experience Replay++ strategy.

    Args:
        model: The PyTorch model to protect.
        buffer_size: Maximum number of samples to store. Default: 200.
        alpha: Weight for cross-entropy replay loss. Default: 0.1.
        beta: Weight for logit-matching (KL divergence) replay loss. Default: 0.5.
        temperature: Temperature for KL divergence logit matching. Default: 2.0.
        buffer_device: Device to store buffer on ('cpu', 'cuda', etc.).
            Default: 'cpu'. Set to 'cuda' to avoid CPU→GPU transfers each batch.
    """

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 200,
        alpha: float = 0.1,
        beta: float = 0.5,
        temperature: float = 2.0,
        buffer_device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self._buffer_size = buffer_size
        self._alpha = alpha
        self._beta = beta
        self._temperature = temperature
        self._buffer_device = torch.device(buffer_device)
        self._buffer_inputs: list[torch.Tensor] = []
        self._buffer_logits: list[torch.Tensor] = []
        self._buffer_targets: list[torch.Tensor] = []
        self._seen_count: int = 0

    def update_buffer(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> None:
        """Add samples to the replay buffer using reservoir sampling.

        Uses Vitter's Algorithm R for O(1) insertion with uniform
        replacement probability.

        Args:
            inputs: Input tensor from the current batch.
            targets: Target tensor from the current batch.
            logits: Model output logits for logit matching.
        """
        if logits is None:
            return

        for i in range(inputs.size(0)):
            self._seen_count += 1
            inp = inputs[i].detach().to(self._buffer_device)
            tgt = targets[i].detach().to(self._buffer_device)
            lgt = logits[i].detach().to(self._buffer_device)

            if len(self._buffer_inputs) < self._buffer_size:
                self._buffer_inputs.append(inp)
                self._buffer_targets.append(tgt)
                self._buffer_logits.append(lgt)
            else:
                j = random.randint(0, self._seen_count - 1)
                if j < self._buffer_size:
                    self._buffer_inputs[j] = inp
                    self._buffer_targets[j] = tgt
                    self._buffer_logits[j] = lgt

    def get_replay_loss(
        self, model: nn.Module, loss_fn: nn.Module
    ) -> torch.Tensor:
        """Compute DER++ replay loss from buffered samples.

        Combines cross-entropy on replayed targets (alpha-weighted) and
        KL divergence on original logits with temperature scaling
        (beta-weighted).

        Args:
            model: The model to evaluate replay samples on.
            loss_fn: The loss function for cross-entropy replay.

        Returns:
            Scalar tensor representing the combined replay loss.
        """
        if not self._buffer_inputs:
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device)

        # Sample a mini-batch from the buffer
        n = min(32, len(self._buffer_inputs))
        indices = random.sample(range(len(self._buffer_inputs)), n)

        device = next(model.parameters()).device
        buf_inputs = torch.stack([self._buffer_inputs[i] for i in indices]).to(device)
        buf_targets = torch.stack([self._buffer_targets[i] for i in indices]).to(device)
        buf_logits = torch.stack([self._buffer_logits[i] for i in indices]).to(device)

        # Forward pass on buffered samples
        current_logits = model(buf_inputs)

        # DER++ losses
        replay_ce = loss_fn(current_logits, buf_targets)

        # KL divergence with temperature scaling for logit matching
        T = self._temperature
        log_p = F.log_softmax(current_logits / T, dim=1)
        q = F.softmax(buf_logits / T, dim=1)
        replay_kl = F.kl_div(log_p, q, reduction="batchmean") * (T * T)

        return self._alpha * replay_ce + self._beta * replay_kl

    def consolidate(self, dataloader: DataLoader) -> None:
        """Top up the buffer if not full after training.

        For DER++, the buffer is primarily populated during training via
        `update_buffer()`. This pass ensures coverage if training was short.

        Args:
            dataloader: Training data for the completed task.
        """
        if len(self._buffer_inputs) >= self._buffer_size:
            return

        self.model.eval()
        device = next(self.model.parameters()).device

        for inputs, targets in dataloader:
            if len(self._buffer_inputs) >= self._buffer_size:
                break
            inputs = inputs.to(device)
            with torch.no_grad():
                logits = self.model(inputs)
            self.update_buffer(inputs, targets, logits)

        self.model.train()

    def penalty(self) -> torch.Tensor:
        """DER++ uses replay, not parameter-space regularization.

        Returns:
            Zero tensor — DER++ does not penalize parameter changes.
        """
        device = next(self.model.parameters()).device
        return torch.tensor(0.0, device=device)

    def get_diagnostics(self) -> dict[str, Any]:
        """Return DER++ diagnostic information.

        Returns:
            Dictionary with buffer statistics and config.
        """
        diag: dict[str, Any] = {
            "strategy": "der++",
            "buffer_size": self._buffer_size,
            "buffer_used": len(self._buffer_inputs),
            "buffer_utilization": len(self._buffer_inputs) / max(self._buffer_size, 1),
            "total_samples_seen": self._seen_count,
            "alpha": self._alpha,
            "beta": self._beta,
            "temperature": self._temperature,
            "buffer_device": str(self._buffer_device),
        }
        if self._buffer_targets:
            targets = torch.stack(self._buffer_targets)
            unique, counts = targets.unique(return_counts=True)
            diag["buffer_class_distribution"] = {
                int(c): int(n) for c, n in zip(unique, counts)
            }
        return diag

    def state_dict(self) -> dict[str, Any]:
        """Serialize buffer contents and hyperparameters."""
        return {
            "buffer_inputs": [t.cpu() for t in self._buffer_inputs],
            "buffer_logits": [t.cpu() for t in self._buffer_logits],
            "buffer_targets": [t.cpu() for t in self._buffer_targets],
            "buffer_size": self._buffer_size,
            "alpha": self._alpha,
            "beta": self._beta,
            "temperature": self._temperature,
            "seen_count": self._seen_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore buffer and hyperparameters from saved state."""
        self._buffer_inputs = [
            t.to(self._buffer_device) for t in state["buffer_inputs"]
        ]
        self._buffer_logits = [
            t.to(self._buffer_device) for t in state["buffer_logits"]
        ]
        self._buffer_targets = [
            t.to(self._buffer_device) for t in state["buffer_targets"]
        ]
        self._buffer_size = state["buffer_size"]
        self._alpha = state["alpha"]
        self._beta = state["beta"]
        self._temperature = state.get("temperature", 2.0)
        self._seen_count = state["seen_count"]
