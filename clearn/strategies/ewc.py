"""Elastic Weight Consolidation (Kirkpatrick et al., 2017).

After each task, computes the Fisher Information Matrix to identify which
weights matter most. During future training, penalizes large changes to
those important weights. Pure regularization — no data replay needed.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from clearn.strategies.base import BaseStrategy


class EWC(BaseStrategy):
    """Elastic Weight Consolidation strategy.

    Args:
        model: The PyTorch model to protect.
        lambda_: Regularization strength. Higher = less forgetting,
            less plasticity. Default: 5000.
        n_fisher_samples: Number of samples used to estimate the Fisher
            Information Matrix. Default: 200.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_: float = 5000,
        n_fisher_samples: int = 200,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self._lambda = lambda_
        self._n_fisher_samples = n_fisher_samples
        self._fisher: dict[str, torch.Tensor] | None = None
        self._optimal_params: dict[str, torch.Tensor] | None = None

    def consolidate(self, dataloader: DataLoader) -> None:
        """Compute Fisher Information Matrix and snapshot parameters.

        Uses the empirical Fisher: diagonal of E[grad(log p)^2] estimated
        over `n_fisher_samples` from the dataloader. Accumulates across
        tasks (online EWC).

        Args:
            dataloader: Training data for the completed task.

        Raises:
            ValueError: If the dataloader is empty.
        """
        from clearn.utils import forward_with_inputs, unpack_batch

        self.model.eval()

        fisher: dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        device = next(self.model.parameters()).device
        n_samples = 0
        for batch in dataloader:
            if n_samples >= self._n_fisher_samples:
                break

            model_inputs, targets = unpack_batch(batch, device)

            # Get batch size from targets (works for both formats)
            batch_size = targets.size(0)
            remaining = self._n_fisher_samples - n_samples
            if batch_size > remaining:
                targets = targets[:remaining]
                if isinstance(model_inputs, dict):
                    model_inputs = {k: v[:remaining] for k, v in model_inputs.items()}
                else:
                    model_inputs = model_inputs[:remaining]
                batch_size = remaining

            self.model.zero_grad()
            outputs = forward_with_inputs(self.model, model_inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += (param.grad.detach() ** 2) * batch_size

            n_samples += batch_size

        if n_samples == 0:
            raise ValueError(
                "Cannot consolidate on empty dataloader. "
                "Provide a dataloader with at least one batch."
            )

        # Average and clamp for numerical stability
        for name in fisher:
            fisher[name] /= n_samples
            fisher[name] = torch.clamp(fisher[name], min=1e-8, max=1e4)

        # Online EWC: accumulate Fisher across tasks
        if self._fisher is not None:
            for name in fisher:
                fisher[name] = self._fisher[name] + fisher[name]

        self._fisher = fisher

        # Snapshot current parameters
        self._optimal_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.train()

    def penalty(self) -> torch.Tensor:
        """Compute the EWC penalty: lambda/2 * sum(F * (theta - theta*)^2).

        Returns:
            Scalar tensor. Zero if no prior consolidation.
        """
        if self._fisher is None or self._optimal_params is None:
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)

        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._fisher:
                diff = param - self._optimal_params[name]
                penalty = penalty + (self._fisher[name] * diff ** 2).sum()

        return (self._lambda / 2) * penalty

    def state_dict(self) -> dict[str, Any]:
        """Serialize EWC state (Fisher matrix, optimal params, hyperparams)."""
        return {
            "fisher": self._fisher,
            "optimal_params": self._optimal_params,
            "lambda_": self._lambda,
            "n_fisher_samples": self._n_fisher_samples,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore EWC state from a saved dictionary."""
        self._fisher = state["fisher"]
        self._optimal_params = state["optimal_params"]
        self._lambda = state["lambda_"]
        self._n_fisher_samples = state["n_fisher_samples"]
