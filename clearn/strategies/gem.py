"""Gradient Episodic Memory (Lopez-Paz & Ranzato, NeurIPS 2017).

Stores a small episodic memory per task. During training on new tasks,
projects the gradient to avoid increasing the loss on any previous task's
memory. Uses a simple projection (A-GEM variant) for efficiency.
"""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from clearn.strategies.base import BaseStrategy


class GEM(BaseStrategy):
    """Gradient Episodic Memory (A-GEM variant) strategy.

    Uses the A-GEM (Averaged GEM) approach for efficiency: instead of
    solving a QP over all task constraints, samples a reference gradient
    from the episodic memory and projects the current gradient if it
    violates the constraint.

    Args:
        model: The PyTorch model to protect.
        memory_size: Number of samples to store per task. Default: 256.
    """

    def __init__(
        self,
        model: nn.Module,
        memory_size: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self._memory_size = memory_size
        # Episodic memories: task_id -> (inputs, targets)
        self._memories: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._current_task_id: str | None = None

    def set_task_id(self, task_id: str) -> None:
        """Set the current task ID for memory allocation.

        Called automatically by ContinualModel.fit().
        """
        self._current_task_id = task_id

    def consolidate(self, dataloader: DataLoader) -> None:
        """Store episodic memory for the completed task.

        Randomly selects ``memory_size`` samples from the dataloader
        to use as constraints for future gradient projections.

        Args:
            dataloader: Training data for the completed task.
        """
        all_inputs, all_targets = [], []
        for batch in dataloader:
            if isinstance(batch, dict):
                inputs = batch.get("input_ids", next(iter(batch.values())))
                targets = batch["labels"]
            else:
                inputs, targets = batch[0], batch[1]
            all_inputs.append(inputs.cpu())
            all_targets.append(targets.cpu())

        if not all_inputs:
            return

        all_inputs = torch.cat(all_inputs)
        all_targets = torch.cat(all_targets)

        # Subsample if needed
        n = all_inputs.size(0)
        if n > self._memory_size:
            indices = random.sample(range(n), self._memory_size)
            all_inputs = all_inputs[indices]
            all_targets = all_targets[indices]

        task_id = self._current_task_id or f"task_{len(self._memories)}"
        self._memories[task_id] = (all_inputs, all_targets)

    def penalty(self) -> torch.Tensor:
        """GEM uses gradient projection, not parameter-space penalty.

        Returns:
            Zero tensor.
        """
        device = next(self.model.parameters()).device
        return torch.tensor(0.0, device=device)

    def project_gradients(self) -> None:
        """Project current gradients to satisfy episodic memory constraints.

        Implements A-GEM: compute a reference gradient from sampled
        episodic memory, and if the current gradient violates the
        constraint (negative inner product), project it onto the
        constraint surface.
        """
        if not self._memories:
            return

        device = next(self.model.parameters()).device

        # Get current gradient
        current_grad = self._flatten_grads()
        if current_grad is None:
            return

        # Compute reference gradient from episodic memory
        ref_grad = self._compute_reference_gradient(device)
        if ref_grad is None:
            return

        # Check constraint: dot(g, g_ref) >= 0
        dot_product = torch.dot(current_grad, ref_grad)
        if dot_product < 0:
            # Project: g_proj = g - (g . g_ref / g_ref . g_ref) * g_ref
            ref_norm_sq = torch.dot(ref_grad, ref_grad)
            if ref_norm_sq > 1e-12:
                proj = current_grad - (dot_product / ref_norm_sq) * ref_grad
                self._unflatten_grads(proj)

    def _flatten_grads(self) -> torch.Tensor | None:
        """Flatten all parameter gradients into a single vector."""
        grads = []
        for param in self.model.parameters():
            if param.requires_grad:
                if param.grad is None:
                    return None
                grads.append(param.grad.detach().flatten())
        return torch.cat(grads) if grads else None

    def _unflatten_grads(self, flat_grad: torch.Tensor) -> None:
        """Write a flat gradient vector back into parameter .grad fields."""
        offset = 0
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                numel = param.numel()
                param.grad.copy_(
                    flat_grad[offset : offset + numel].view_as(param)
                )
                offset += numel

    def _compute_reference_gradient(
        self, device: torch.device
    ) -> torch.Tensor | None:
        """Compute an averaged reference gradient from episodic memories."""
        from clearn.utils import forward_with_inputs

        self.model.zero_grad()

        total_loss = torch.tensor(0.0, device=device)
        n_memories = 0

        for _task_id, (mem_inputs, mem_targets) in self._memories.items():
            # Sample a mini-batch from memory
            n = min(32, mem_inputs.size(0))
            indices = random.sample(range(mem_inputs.size(0)), n)
            batch_in = mem_inputs[indices].to(device)
            batch_tgt = mem_targets[indices].to(device)

            outputs = forward_with_inputs(self.model, batch_in)
            loss = F.cross_entropy(outputs, batch_tgt)
            total_loss = total_loss + loss
            n_memories += 1

        if n_memories == 0:
            return None

        avg_loss = total_loss / n_memories
        avg_loss.backward()

        ref_grad = self._flatten_grads()

        # Zero grads so they don't interfere with current training
        self.model.zero_grad()
        return ref_grad

    def before_optimizer_step(self) -> None:
        """Project gradients before optimizer step."""
        self.project_gradients()

    def after_optimizer_step(self) -> None:
        """No-op for GEM — projection happens before optimizer step."""

    def get_diagnostics(self) -> dict[str, Any]:
        """Return GEM diagnostic information."""
        diag: dict[str, Any] = {
            "strategy": "gem",
            "memory_size_per_task": self._memory_size,
            "tasks_in_memory": len(self._memories),
            "total_memory_samples": sum(
                m[0].size(0) for m in self._memories.values()
            ),
        }
        for task_id, (inp, _) in self._memories.items():
            diag[f"memory_{task_id}_size"] = inp.size(0)
        return diag

    def state_dict(self) -> dict[str, Any]:
        """Serialize GEM state."""
        return {
            "memories": {
                k: (v[0].cpu(), v[1].cpu())
                for k, v in self._memories.items()
            },
            "memory_size": self._memory_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore GEM state from a saved dictionary."""
        self._memories = state["memories"]
        self._memory_size = state["memory_size"]
