"""Retention report and evaluation metrics for continual learning."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class RetentionReport:
    """Human-readable retention report — like git diff for model knowledge.

    Attributes:
        task_scores: Per-task retention percentages (0-100).
        plasticity_score: How well the current task was learned (0.0-1.0).
        stability_score: Average retention across all past tasks (0.0-1.0).
        recommendation: Actionable advice based on stability.
    """

    task_scores: dict[str, float] = field(default_factory=dict)
    plasticity_score: float = 0.0
    stability_score: float = 0.0
    recommendation: str = ""

    def __repr__(self) -> str:
        lines = ["RetentionReport"]

        items = list(self.task_scores.items())
        total_entries = len(items) + 3  # tasks + plasticity + stability + rec

        for i, (task_id, score) in enumerate(items):
            connector = "\u2502   " if i < len(items) - 1 or total_entries > len(items) else "\u2514\u2500\u2500 "
            connector = "\u251c\u2500\u2500 "
            suffix = ""
            if score >= 99.9:
                suffix = " (current task)"
            else:
                drop = 100.0 - score
                suffix = f"  (-{drop:.1f}%)"
            lines.append(f"{connector}{task_id}: {score:.1f}% retained{suffix}")

        lines.append(f"\u251c\u2500\u2500 plasticity_score: {self.plasticity_score:.2f}")
        lines.append(f"\u251c\u2500\u2500 stability_score: {self.stability_score:.2f}")
        lines.append(f"\u2514\u2500\u2500 recommendation: \"{self.recommendation}\"")

        return "\n".join(lines)


def _get_recommendation(stability_score: float) -> str:
    """Generate a recommendation based on stability score.

    Args:
        stability_score: Average retention across past tasks (0.0-1.0).

    Returns:
        Human-readable advice string.
    """
    if stability_score >= 0.9:
        return "stable \u2014 no action needed"
    elif stability_score >= 0.7:
        return "mild forgetting \u2014 consider increasing lambda or buffer_size"
    elif stability_score >= 0.5:
        return "significant forgetting \u2014 increase regularization strength"
    else:
        return "severe forgetting \u2014 consider a replay-based strategy (der) or reduce learning rate"


@torch.no_grad()
def compute_retention(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model accuracy on a dataloader.

    Handles both standard (tensor, tensor) and HuggingFace dict batches.

    Args:
        model: The model to evaluate.
        dataloader: Data to evaluate on.
        device: Device to run evaluation on.

    Returns:
        Accuracy as a float between 0.0 and 1.0.
    """
    from clearn.utils import forward_with_inputs, unpack_batch

    model.eval()
    correct = 0
    total = 0

    for batch in dataloader:
        model_inputs, targets = unpack_batch(batch, device)
        outputs = forward_with_inputs(model, model_inputs)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    model.train()

    if total == 0:
        return 0.0
    return correct / total
