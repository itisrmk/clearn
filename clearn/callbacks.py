"""Callback system for ContinualModel training hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clearn.core import ContinualModel
    from clearn.metrics import TrainingMetrics


class ContinualCallback:
    """Base class for training callbacks.

    Override any of the hook methods to add custom behavior
    during continual learning training.

    Example::

        class PrintCallback(ContinualCallback):
            def on_task_start(self, model, task_id):
                print(f"Starting {task_id}")

            def on_task_end(self, model, task_id, metrics):
                print(f"Finished {task_id}: {metrics.final_accuracy:.2%}")

        model.fit(loader, opt, callbacks=[PrintCallback()])
    """

    def on_task_start(
        self, model: ContinualModel, task_id: str
    ) -> None:
        """Called before training begins for a task.

        Args:
            model: The ContinualModel being trained.
            task_id: The identifier for the upcoming task.
        """

    def on_batch_end(
        self, model: ContinualModel, loss: float
    ) -> None:
        """Called after each training batch.

        Args:
            model: The ContinualModel being trained.
            loss: The total loss for this batch.
        """

    def on_task_end(
        self,
        model: ContinualModel,
        task_id: str,
        metrics: TrainingMetrics,
    ) -> None:
        """Called after a task finishes (including consolidation).

        Args:
            model: The ContinualModel being trained.
            task_id: The identifier for the completed task.
            metrics: The training metrics for this task.
        """


class EarlyStoppingCallback(ContinualCallback):
    """Stop training early if loss plateaus within a task.

    Note: This only monitors loss within a single task's training.
    It does not stop across tasks.

    Args:
        patience: Number of batches with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 50, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss = float("inf")
        self._wait = 0
        self.stopped = False

    def on_task_start(
        self, model: ContinualModel, task_id: str
    ) -> None:
        self._best_loss = float("inf")
        self._wait = 0
        self.stopped = False

    def on_batch_end(
        self, model: ContinualModel, loss: float
    ) -> None:
        if loss < self._best_loss - self.min_delta:
            self._best_loss = loss
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.stopped = True
