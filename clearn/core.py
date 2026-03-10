"""ContinualModel — the main wrapper class for continual learning."""

from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from clearn.metrics import (
    RetentionReport,
    TrainingMetrics,
    _get_recommendation,
    compute_retention,
)
from clearn.strategies import BaseStrategy, get_strategy
from clearn.utils import (
    forward_with_inputs,
    generate_task_id,
    get_device,
    inputs_for_buffer,
    unpack_batch,
)


def _make_eval_subset(
    dataloader: DataLoader, n: int = 500
) -> DataLoader:
    """Extract a fixed-size subset from a dataloader for evaluation.

    Handles both standard (tensor, tensor) and HuggingFace dict batches.

    Args:
        dataloader: The source dataloader.
        n: Maximum number of samples to keep.

    Returns:
        A new DataLoader with at most `n` samples.
    """
    all_inputs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    count = 0
    is_dict = False

    for batch in dataloader:
        if isinstance(batch, dict):
            is_dict = True
            inputs = batch.get("input_ids", next(iter(batch.values())))
            targets = batch["labels"]
        else:
            inputs, targets = batch[0], batch[1]

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
        grad_clip: float | None = None,
        callbacks: list[Any] | None = None,
        use_amp: bool = False,
    ) -> TrainingMetrics:
        """Train the model on a new task with forgetting protection.

        Automatically calls ``consolidate()`` after training to lock in
        the learned knowledge.

        Args:
            dataloader: Training data for this task.
            optimizer: The optimizer to use.
            epochs: Number of training epochs. Default: 1.
            task_id: Optional name for this task. Auto-generated if None.
            loss_fn: Loss function. Defaults to CrossEntropyLoss.
            grad_clip: Max gradient norm for clipping. None disables clipping.
            callbacks: Optional list of ``ContinualCallback`` instances.
            use_amp: Enable automatic mixed precision (requires CUDA).

        Returns:
            A TrainingMetrics object with per-epoch loss/accuracy and timing.
        """
        if task_id is None:
            task_id = generate_task_id(self._task_history)

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        if callbacks is None:
            callbacks = []

        # Let strategy know the current task ID (used by GEM)
        if hasattr(self.strategy, "set_task_id"):
            self.strategy.set_task_id(task_id)

        # Fire on_task_start callbacks
        for cb in callbacks:
            cb.on_task_start(self, task_id)

        # Set up AMP scaler
        scaler = torch.amp.GradScaler(enabled=use_amp)
        amp_device_type = "cuda" if self._device.type == "cuda" else "cpu"

        self.model.train()

        start_time = time.monotonic()
        epoch_losses: list[float] = []
        epoch_accuracies: list[float] = []

        for _epoch in range(epochs):
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            for batch in dataloader:
                model_inputs, targets = unpack_batch(batch, self._device)

                optimizer.zero_grad()

                with torch.amp.autocast(
                    device_type=amp_device_type, enabled=use_amp
                ):
                    outputs = forward_with_inputs(self.model, model_inputs)
                    task_loss = loss_fn(outputs, targets)

                    penalty = self.strategy.penalty()
                    replay_loss = self.strategy.get_replay_loss(
                        self.model, loss_fn
                    )

                    loss = task_loss + penalty + replay_loss

                scaler.scale(loss).backward()

                # Hook for strategies that modify gradients (e.g. GEM)
                if use_amp:
                    scaler.unscale_(optimizer)
                self.strategy.before_optimizer_step()

                if grad_clip is not None:
                    if not use_amp:
                        # Already unscaled above if use_amp
                        pass
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()

                # Hook for strategies that track params online (e.g. SI)
                self.strategy.after_optimizer_step()

                # Update replay buffer (no-op for non-replay strategies)
                buf_inputs = inputs_for_buffer(model_inputs)
                self.strategy.update_buffer(
                    buf_inputs.detach(), targets.detach(), logits=outputs.detach()
                )

                # Track metrics
                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size
                preds = outputs.argmax(dim=1)
                running_correct += (preds == targets).sum().item()
                running_total += batch_size

                # Fire on_batch_end callbacks
                for cb in callbacks:
                    cb.on_batch_end(self, loss.item())

            avg_loss = running_loss / max(running_total, 1)
            avg_acc = running_correct / max(running_total, 1)
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(avg_acc)

        wall_time = time.monotonic() - start_time

        # Consolidate — lock in knowledge from this task
        self.strategy.consolidate(dataloader)

        # Store eval subset for diff()
        self._task_dataloaders[task_id] = _make_eval_subset(dataloader)

        # Cache current task accuracy
        self._eval_cache[task_id] = compute_retention(
            self.model, self._task_dataloaders[task_id], self._device
        )

        self._task_history.append(task_id)

        metrics = TrainingMetrics(
            task_id=task_id,
            epochs=epochs,
            epoch_losses=epoch_losses,
            epoch_accuracies=epoch_accuracies,
            final_loss=epoch_losses[-1] if epoch_losses else 0.0,
            final_accuracy=epoch_accuracies[-1] if epoch_accuracies else 0.0,
            wall_time=wall_time,
        )

        # Fire on_task_end callbacks
        for cb in callbacks:
            cb.on_task_end(self, task_id, metrics)

        return metrics

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

    def diagnostics(self) -> dict[str, Any]:
        """Return diagnostic information about the current strategy state.

        Returns:
            A dictionary with strategy-specific diagnostic metrics.
        """
        diag = self.strategy.get_diagnostics()
        diag["tasks_trained"] = len(self._task_history)
        diag["task_history"] = list(self._task_history)
        return diag

    def save(self, path: str) -> None:
        """Save the full model and strategy state to disk.

        Serializes model weights, strategy state, task history, and
        evaluation data so that ``diff()`` works after ``load()``.

        Args:
            path: Directory path to save the checkpoint to.
        """
        os.makedirs(path, exist_ok=True)

        # Serialize eval dataloaders as raw tensor pairs
        eval_data: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for task_id, dl in self._task_dataloaders.items():
            all_x, all_y = [], []
            for batch in dl:
                x, y = batch[0], batch[1]
                all_x.append(x.cpu())
                all_y.append(y.cpu())
            if all_x:
                eval_data[task_id] = (torch.cat(all_x), torch.cat(all_y))

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "strategy_name": self._strategy_name,
            "strategy_state": self.strategy.state_dict(),
            "task_history": self._task_history,
            "eval_cache": self._eval_cache,
            "eval_data": eval_data,
            "version": "0.2.1",
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

        # Restore eval dataloaders so diff() works after load()
        eval_data = checkpoint.get("eval_data", {})
        for task_id, (X, y) in eval_data.items():
            dataset = TensorDataset(X, y)
            instance._task_dataloaders[task_id] = DataLoader(
                dataset, batch_size=64
            )

        return instance

    def save_pretrained(self, path: str) -> None:
        """Save model using HuggingFace's save_pretrained + clearn state.

        For HuggingFace models, saves using the native ``save_pretrained``
        format alongside the clearn checkpoint. This makes models compatible
        with ``push_to_hub``.

        Args:
            path: Directory path to save to.
        """
        os.makedirs(path, exist_ok=True)

        # Save HF model natively if it supports save_pretrained
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)

        # Always save clearn checkpoint alongside
        self.save(path)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload clearn continual learning model",
        private: bool = False,
        token: str | None = None,
    ) -> str:
        """Push the model and clearn state to the HuggingFace Hub.

        Saves the model using HuggingFace's native format alongside the
        clearn checkpoint, then uploads to the Hub.

        Args:
            repo_id: The HuggingFace Hub repository ID (e.g. "user/model-name").
            commit_message: Commit message for the upload.
            private: Whether the repository should be private. Default: False.
            token: HuggingFace API token. Uses cached token if None.

        Returns:
            The URL of the uploaded model on the Hub.

        Raises:
            ImportError: If ``huggingface_hub`` is not installed.
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "push_to_hub requires 'huggingface_hub'. "
                "Install with: pip install huggingface_hub"
            )

        import tempfile

        api = HfApi(token=token)

        # Create or get repo
        api.create_repo(repo_id, private=private, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)

            # Upload all files
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                commit_message=commit_message,
            )

        return f"https://huggingface.co/{repo_id}"

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
