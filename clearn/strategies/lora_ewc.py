"""LoRA-EWC — Parameter-efficient continual learning.

Combines LoRA (Low-Rank Adaptation) for efficient fine-tuning with EWC
regularization applied only to the LoRA adapter weights. This gives you
parameter-efficient training AND forgetting protection in one strategy.

Requires: pip install clearn-ai[hf]  (needs peft library)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from clearn.strategies.base import BaseStrategy

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError:
    LoraConfig = None
    get_peft_model = None
    PeftModel = None


def _check_peft_installed() -> None:
    """Raise a clear error if peft is not installed."""
    if LoraConfig is None:
        raise ImportError(
            "LoRA-EWC strategy requires 'peft'. "
            "Install with: pip install clearn-ai[hf]"
        )


class LoRAEWC(BaseStrategy):
    """LoRA + EWC hybrid strategy for parameter-efficient continual learning.

    Applies LoRA adapters to the model for efficient fine-tuning, then
    uses EWC regularization on only the LoRA parameters to prevent
    forgetting. The base model weights stay frozen.

    Args:
        model: The PyTorch model to adapt. LoRA adapters will be added
            automatically if the model is not already a PeftModel.
        lambda_: EWC regularization strength on LoRA weights. Default: 5000.
        n_fisher_samples: Samples for Fisher estimation. Default: 200.
        lora_r: LoRA rank. Default: 8.
        lora_alpha: LoRA alpha scaling. Default: 16.
        lora_dropout: Dropout for LoRA layers. Default: 0.1.
        target_modules: Which modules to apply LoRA to. Default: None
            (auto-detect linear layers).
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_: float = 5000,
        n_fisher_samples: int = 200,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        _check_peft_installed()

        # Apply LoRA if not already a PeftModel
        if PeftModel is not None and not isinstance(model, PeftModel):
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            model = get_peft_model(model, lora_config)

        super().__init__(model, **kwargs)

        self._lambda = lambda_
        self._n_fisher_samples = n_fisher_samples
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self._lora_dropout = lora_dropout
        self._fisher: dict[str, torch.Tensor] | None = None
        self._optimal_params: dict[str, torch.Tensor] | None = None

    def _lora_params(self) -> list[tuple[str, nn.Parameter]]:
        """Get only the LoRA adapter parameters (trainable params)."""
        return [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]

    def consolidate(self, dataloader: DataLoader) -> None:
        """Compute Fisher Information Matrix over LoRA parameters only.

        Much faster than full EWC since only adapter weights are tracked.

        Args:
            dataloader: Training data for the completed task.

        Raises:
            ValueError: If the dataloader is empty.
        """
        from clearn.utils import forward_with_inputs, unpack_batch

        self.model.eval()

        fisher: dict[str, torch.Tensor] = {}
        for name, param in self._lora_params():
            fisher[name] = torch.zeros_like(param)

        device = next(self.model.parameters()).device
        n_samples = 0

        for batch in dataloader:
            if n_samples >= self._n_fisher_samples:
                break

            model_inputs, targets = unpack_batch(batch, device)

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

            for name, param in self._lora_params():
                if param.grad is not None:
                    fisher[name] += (param.grad.detach() ** 2) * batch_size

            n_samples += batch_size

        if n_samples == 0:
            raise ValueError(
                "Cannot consolidate on empty dataloader. "
                "Provide a dataloader with at least one batch."
            )

        # Average and clamp
        for name in fisher:
            fisher[name] /= n_samples
            fisher[name] = torch.clamp(fisher[name], min=1e-8, max=1e4)

        # Online EWC: accumulate
        if self._fisher is not None:
            for name in fisher:
                if name in self._fisher:
                    fisher[name] = self._fisher[name] + fisher[name]

        self._fisher = fisher

        # Snapshot LoRA parameters
        self._optimal_params = {
            name: param.detach().clone()
            for name, param in self._lora_params()
        }

        self.model.train()

    def penalty(self) -> torch.Tensor:
        """EWC penalty computed over LoRA parameters only.

        Returns:
            Scalar tensor. Zero if no prior consolidation.
        """
        if self._fisher is None or self._optimal_params is None:
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)

        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self._lora_params():
            if name in self._fisher:
                diff = param - self._optimal_params[name]
                penalty = penalty + (self._fisher[name] * diff ** 2).sum()

        return (self._lambda / 2) * penalty

    def get_diagnostics(self) -> dict[str, Any]:
        """Return LoRA-EWC diagnostic information.

        Returns:
            Dictionary with Fisher stats, LoRA config, and penalty info.
        """
        diag: dict[str, Any] = {
            "strategy": "lora-ewc",
            "lambda": self._lambda,
            "n_fisher_samples": self._n_fisher_samples,
            "lora_r": self._lora_r,
            "lora_alpha": self._lora_alpha,
            "lora_dropout": self._lora_dropout,
            "consolidated": self._fisher is not None,
            "n_lora_params": len(self._lora_params()),
            "total_lora_parameters": sum(
                p.numel() for _, p in self._lora_params()
            ),
        }
        if self._fisher is not None:
            all_fisher = torch.cat([f.flatten() for f in self._fisher.values()])
            diag["fisher_mean"] = float(all_fisher.mean())
            diag["fisher_std"] = float(all_fisher.std())
            diag["fisher_max"] = float(all_fisher.max())
            diag["current_penalty"] = float(self.penalty().item())
        return diag

    def state_dict(self) -> dict[str, Any]:
        """Serialize LoRA-EWC state."""
        return {
            "fisher": self._fisher,
            "optimal_params": self._optimal_params,
            "lambda_": self._lambda,
            "n_fisher_samples": self._n_fisher_samples,
            "lora_r": self._lora_r,
            "lora_alpha": self._lora_alpha,
            "lora_dropout": self._lora_dropout,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore LoRA-EWC state."""
        self._fisher = state["fisher"]
        self._optimal_params = state["optimal_params"]
        self._lambda = state["lambda_"]
        self._n_fisher_samples = state["n_fisher_samples"]
        self._lora_r = state.get("lora_r", self._lora_r)
        self._lora_alpha = state.get("lora_alpha", self._lora_alpha)
        self._lora_dropout = state.get("lora_dropout", self._lora_dropout)
