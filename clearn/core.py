"""ContinualModel — the main wrapper class for continual learning."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from clearn.metrics import RetentionReport, _get_recommendation, compute_retention
from clearn.strategies import BaseStrategy, get_strategy
from clearn.utils import generate_task_id, get_device


def _make_eval_subset(
    dataloader: DataLoader, n: int = 500
) -> DataLoader:
    """Extract a fixed-size subset from a dataloader for evaluation.

    Args:
        dataloader: The source dataloader.
        n: Maximum number of samples to keep.

    Returns:
        A new DataLoader with at most `n` samples.
    """
    all_inputs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    count = 0

    for inputs, targets in dataloader:
        remaining = n - count
        if remaining <= 0:
            break
        if inputs.size(0) > remaining:
            inputs = inputs[:remaining]
            targets = targets[:remaining]
        all_inputs.append(inputs.cpu())
        all_targets.append(targets.cpu())
        count += inputs.size(0)

    if not all_inputs:
        return dataloader

    dataset = TensorDataset(torch.cat(all_inputs), torch.cat(all_targets))
    return DataLoader(dataset, batch_size=dataloader.batch_size or 64)


class ContinualModel:
    """Wraps any PyTorch model with continual learning protection.

    Args:
        model: Any PyTorch nn.Module.
        strategy: Strategy name ("ewc", "der") or a BaseStrategy instance.
        **strategy_kwargs: Keyword arguments passed to the strategy constructor.

    Example:
        >>> model = clearn.wrap(your_model, strategy="ewc", lambda_=5000)
        >>> model.fit(dataloader, optimizer)
        >>> print(model.diff())
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: str | BaseStrategy = "ewc",
        **strategy_kwargs: Any,
    ) -> None:
        self.model = model
        self.strategy = get_strategy(strategy, model, **strategy_kwargs)
        self._strategy_name = (
            strategy if isinstance(strategy, str) else type(strategy).__name__.lower()
        )
        self._task_history: list[str] = []
        self._eval_cache: dict[str, float] = {}
        self._task_dataloaders: dict[str, DataLoader] = {}
        self._device = get_device(model)

    @classmethod
    def wrap(
        cls,
        model: nn.Module,
        strategy: str | BaseStrategy = "ewc",
        **kwargs: Any,
    ) -> ContinualModel:
        """Wrap a PyTorch model with continual learning protection.

        This is the primary public API entry point.

        Args:
            model: Any PyTorch nn.Module.
            strategy: Strategy name ("ewc", "der") or a BaseStrategy instance.
            **kwargs: Strategy-specific keyword arguments.

        Returns:
            A ContinualModel wrapping the provided model.
        """
        return cls(model, strategy, **kwargs)

    def fit(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 1,
        task_id: str | None = None,
        loss_fn: nn.Module | None = None,
    ) -> ContinualModel:
        """Train the model on a new task with forgetting protection.

        Automatically calls `consolidate()` after training to lock in
        the learned knowledge.

        Args:
            dataloader: Training data for this task.
            optimizer: The optimizer to use.
            epochs: Number of training epochs. Default: 1.
            task_id: Optional name for this task. Auto-generated if None.
            loss_fn: Loss function. Defaults to CrossEntropyLoss.

        Returns:
            self, for method chaining.
        """
        if task_id is None:
            task_id = generate_task_id(self._task_history)

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        self.model.train()

        for _epoch in range(epochs):
            for inputs, targets in dataloader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                task_loss = loss_fn(outputs, targets)

                penalty = self.strategy.penalty()
                replay_loss = self.strategy.get_replay_loss(self.model, loss_fn)

                loss = task_loss + penalty + replay_loss

                loss.backward()
                optimizer.step()

                # Update replay buffer (no-op for non-replay strategies)
                self.strategy.update_buffer(
                    inputs.detach(), targets.detach(), logits=outputs.detach()
                )

        # Consolidate — lock in knowledge from this task
        self.strategy.consolidate(dataloader)

        # Store eval subset for diff()
        self._task_dataloaders[task_id] = _make_eval_subset(dataloader)

        # Cache current task accuracy
        self._eval_cache[task_id] = compute_retention(
            self.model, self._task_dataloaders[task_id], self._device
        )

        self._task_history.append(task_id)
        return self

    def diff(self) -> RetentionReport:
        """Compute a retention report across all trained tasks.

        Re-evaluates the model on stored evaluation data for each task
        and produces a human-readable report.

        Returns:
            A RetentionReport with per-task scores and recommendations.

        Raises:
            RuntimeError: If called before any `fit()`.
        """
        if not self._task_history:
            raise RuntimeError(
                "Cannot compute diff before any training. Call fit() first."
            )

        task_scores: dict[str, float] = {}
        for task_id in self._task_history:
            if task_id in self._task_dataloaders:
                score = compute_retention(
                    self.model, self._task_dataloaders[task_id], self._device
                )
                task_scores[task_id] = score * 100.0

        # Plasticity: how well the latest task was learned
        latest = self._task_history[-1]
        plasticity = task_scores.get(latest, 0.0) / 100.0

        # Stability: average retention across past tasks (excluding current)
        past_scores = [
            task_scores[tid] / 100.0
            for tid in self._task_history[:-1]
            if tid in task_scores
        ]
        stability = sum(past_scores) / len(past_scores) if past_scores else 1.0

        recommendation = _get_recommendation(stability)

        return RetentionReport(
            task_scores=task_scores,
            plasticity_score=plasticity,
            stability_score=stability,
            recommendation=recommendation,
        )

    def save(self, path: str) -> None:
        """Save the full model and strategy state to disk.

        Args:
            path: Directory path to save the checkpoint to.
        """
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "strategy_name": self._strategy_name,
            "strategy_state": self.strategy.state_dict(),
            "task_history": self._task_history,
            "eval_cache": self._eval_cache,
            "version": "0.1.0",
        }
        torch.save(checkpoint, os.path.join(path, "checkpoint.pt"))

    @classmethod
    def load(
        cls,
        path: str,
        model: nn.Module | None = None,
    ) -> ContinualModel:
        """Load a saved ContinualModel from disk.

        Args:
            path: Directory path containing the checkpoint.
            model: The model architecture to load weights into.
                Must match the architecture used during save.

        Returns:
            A restored ContinualModel.

        Raises:
            ValueError: If model is not provided.
        """
        if model is None:
            raise ValueError(
                "Must provide the model architecture to load into. "
                "Usage: clearn.load('path', model=your_model)"
            )

        checkpoint_path = os.path.join(path, "checkpoint.pt")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])

        strategy_name = checkpoint["strategy_name"]
        instance = cls(model, strategy=strategy_name)
        instance.strategy.load_state_dict(checkpoint["strategy_state"])
        instance._task_history = checkpoint["task_history"]
        instance._eval_cache = checkpoint["eval_cache"]

        return instance

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying model.

        This allows calling model.eval(), model(x), etc. directly
        on the ContinualModel wrapper.
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            model = object.__getattribute__(self, "model")
            return getattr(model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the underlying model."""
        return self.model(*args, **kwargs)
