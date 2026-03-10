"""Synaptic Intelligence (Zenke et al., ICML 2017).

Tracks per-parameter path integrals online during training to estimate
parameter importance. After each task, accumulated importance is used
to penalize changes to important weights — similar to EWC but computed
online rather than via a separate Fisher pass.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from clearn.strategies.base import BaseStrategy


class SI(BaseStrategy):
    """Synaptic Intelligence strategy.

    Args:
        model: The PyTorch model to protect.
        c: Regularization strength (analogous to EWC's lambda).
            Default: 1.0.
        epsilon: Small constant for numerical stability in importance
            normalization. Default: 1e-3.
    """

    def __init__(
        self,
        model: nn.Module,
        c: float = 1.0,
        epsilon: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self._c = c
        self._epsilon = epsilon

        # Running path integral (accumulated across batches within a task)
        self._W: dict[str, torch.Tensor] = {}
        # Accumulated importance (accumulated across tasks)
        self._omega: dict[str, torch.Tensor] = {}
        # Parameter snapshot at the start of the current task
        self._prev_params: dict[str, torch.Tensor] = {}
        # Consolidated optimal params (for penalty computation)
        self._optimal_params: dict[str, torch.Tensor] | None = None

        # Initialize running accumulators
        self._init_tracking()

    def _init_tracking(self) -> None:
        """Initialize parameter snapshots and running integrals."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._W[name] = torch.zeros_like(param)
                self._prev_params[name] = param.detach().clone()

    def update_running_importance(self) -> None:
        """Update the running path integral after an optimizer step.

        Call this after ``optimizer.step()`` to accumulate the
        per-parameter path integral: ``W += -grad * delta``.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._prev_params:
                if param.grad is not None:
                    delta = param.detach() - self._prev_params[name]
                    self._W[name] += (-param.grad.detach() * delta)
                self._prev_params[name] = param.detach().clone()

    def after_optimizer_step(self) -> None:
        """Update the running path integral after an optimizer step."""
        self.update_running_importance()

    def consolidate(self, dataloader: DataLoader) -> None:
        """Finalize importance for the completed task.

        Computes per-parameter importance as the path integral
        normalized by the squared parameter change + epsilon.

        Args:
            dataloader: Training data (unused by SI, present for API
                compatibility).
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._W:
                # Importance = W / (delta^2 + epsilon)
                if self._optimal_params is not None:
                    delta = param.detach() - self._optimal_params[name]
                else:
                    delta = param.detach() - self._prev_params.get(
                        name, param.detach()
                    )

                importance = self._W[name] / (delta ** 2 + self._epsilon)
                importance = torch.clamp(importance, min=0.0)

                if name in self._omega:
                    self._omega[name] += importance
                else:
                    self._omega[name] = importance

        # Snapshot current parameters as optimal
        self._optimal_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Reset running integrals for next task
        for name in self._W:
            self._W[name].zero_()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._prev_params[name] = param.detach().clone()

    def penalty(self) -> torch.Tensor:
        """Compute the SI penalty: c * sum(omega * (theta - theta*)^2).

        Returns:
            Scalar tensor. Zero if no prior consolidation.
        """
        if self._optimal_params is None or not self._omega:
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)

        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._omega:
                diff = param - self._optimal_params[name]
                penalty = penalty + (self._omega[name] * diff ** 2).sum()

        return self._c * penalty

    def get_diagnostics(self) -> dict[str, Any]:
        """Return SI diagnostic information."""
        diag: dict[str, Any] = {
            "strategy": "si",
            "c": self._c,
            "epsilon": self._epsilon,
            "consolidated": self._optimal_params is not None,
        }
        if self._omega:
            all_omega = torch.cat([o.flatten() for o in self._omega.values()])
            diag["omega_mean"] = float(all_omega.mean())
            diag["omega_std"] = float(all_omega.std())
            diag["omega_max"] = float(all_omega.max())
            diag["n_protected_params"] = len(self._omega)
            diag["current_penalty"] = float(self.penalty().item())
        return diag

    def state_dict(self) -> dict[str, Any]:
        """Serialize SI state."""
        return {
            "omega": self._omega,
            "optimal_params": self._optimal_params,
            "c": self._c,
            "epsilon": self._epsilon,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore SI state from a saved dictionary."""
        self._omega = state["omega"]
        self._optimal_params = state["optimal_params"]
        self._c = state["c"]
        self._epsilon = state["epsilon"]
