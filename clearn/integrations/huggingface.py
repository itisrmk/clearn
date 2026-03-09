"""HuggingFace Transformers integration for clearn.

Provides `from_pretrained()` to load any HuggingFace model and wrap it
with continual learning, and `ContinualTrainer` for Trainer-compatible
training with automatic forgetting protection.

Requires: pip install clearn-ai[hf]
"""

from __future__ import annotations

from typing import Any

try:
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        Trainer,
        TrainingArguments,
    )
except ImportError:
    raise ImportError(
        "HuggingFace integration requires 'transformers'. "
        "Install with: pip install clearn-ai[hf]"
    )

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from clearn.core import ContinualModel

# Task -> AutoModel mapping
_TASK_MODEL_MAP = {
    "classification": AutoModelForSequenceClassification,
    "sequence-classification": AutoModelForSequenceClassification,
    "token-classification": AutoModelForTokenClassification,
    "ner": AutoModelForTokenClassification,
    "causal-lm": AutoModelForCausalLM,
    "text-generation": AutoModelForCausalLM,
    "seq2seq-lm": AutoModelForSeq2SeqLM,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}


def from_pretrained(
    model_name: str,
    strategy: str = "ewc",
    task: str = "classification",
    num_labels: int = 2,
    **kwargs: Any,
) -> ContinualModel:
    """Load a HuggingFace model and wrap it with continual learning.

    Args:
        model_name: HuggingFace model identifier (e.g. "bert-base-uncased").
        strategy: Continual learning strategy name. Default: "ewc".
            Use "lora-ewc" for parameter-efficient continual learning.
        task: The model task type. Options:
            - "classification" / "sequence-classification"
            - "token-classification" / "ner"
            - "causal-lm" / "text-generation"
            - "seq2seq-lm" / "summarization" / "translation"
            Default: "classification".
        num_labels: Number of output labels (for classification tasks).
            Default: 2.
        **kwargs: Additional keyword arguments passed to the strategy.
            For "lora-ewc": lora_r, lora_alpha, lora_dropout, target_modules.

    Returns:
        A ContinualModel wrapping the HuggingFace model.

    Raises:
        ValueError: If the task type is not recognized.

    Example:
        >>> model = clearn.from_pretrained("bert-base-uncased", strategy="ewc")
        >>> model = clearn.from_pretrained("gpt2", task="causal-lm", strategy="lora-ewc")
    """
    task_lower = task.lower()
    if task_lower not in _TASK_MODEL_MAP:
        available = ", ".join(sorted(set(_TASK_MODEL_MAP.keys())))
        raise ValueError(
            f"Unknown task '{task}'. Available tasks: {available}"
        )

    auto_cls = _TASK_MODEL_MAP[task_lower]

    # Build model kwargs
    model_kwargs: dict[str, Any] = {}
    if task_lower in ("classification", "sequence-classification", "token-classification", "ner"):
        model_kwargs["num_labels"] = num_labels

    model = auto_cls.from_pretrained(model_name, **model_kwargs)

    return ContinualModel.wrap(model, strategy=strategy, **kwargs)


class ContinualTrainer:
    """HuggingFace Trainer wrapper with automatic continual learning.

    Wraps the standard HuggingFace Trainer to automatically apply
    forgetting protection during training and call consolidate()
    after each task.

    Args:
        model: A ContinualModel (from clearn.wrap or clearn.from_pretrained).
        args: HuggingFace TrainingArguments.
        train_dataset: Training dataset for the current task.
        eval_dataset: Evaluation dataset (optional).
        task_id: Name for this training task.
        **trainer_kwargs: Additional kwargs passed to HuggingFace Trainer.

    Example:
        >>> cl_model = clearn.from_pretrained("bert-base-uncased", strategy="lora-ewc")
        >>> trainer = ContinualTrainer(
        ...     model=cl_model,
        ...     args=TrainingArguments(output_dir="./out", num_train_epochs=3),
        ...     train_dataset=dataset,
        ...     task_id="sentiment_v1",
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: ContinualModel,
        args: TrainingArguments,
        train_dataset: Any,
        eval_dataset: Any | None = None,
        task_id: str | None = None,
        **trainer_kwargs: Any,
    ) -> None:
        if not isinstance(model, ContinualModel):
            raise TypeError(
                "ContinualTrainer requires a ContinualModel. "
                "Use clearn.wrap() or clearn.from_pretrained() first."
            )

        self.cl_model = model
        self.task_id = task_id
        self._args = args
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset

        # Build the inner HF Trainer with the unwrapped model
        # and a custom loss that adds the strategy penalty
        self._trainer = _ContinualHFTrainer(
            model=model.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            cl_strategy=model.strategy,
            **trainer_kwargs,
        )

    def train(self, **kwargs: Any) -> Any:
        """Train the model with continual learning protection.

        After training completes, automatically consolidates knowledge
        and updates the ContinualModel's task history.

        Returns:
            The HuggingFace TrainerOutput.
        """
        from clearn.utils import generate_task_id

        result = self._trainer.train(**kwargs)

        # Post-training: consolidate
        train_loader = self._trainer.get_train_dataloader()
        self.cl_model.strategy.consolidate(train_loader)

        # Update task history
        task_id = self.task_id or generate_task_id(self.cl_model._task_history)
        self.cl_model._task_history.append(task_id)

        # Store eval subset for diff()
        from clearn.core import _make_eval_subset
        from clearn.metrics import compute_retention
        from clearn.utils import get_device

        eval_loader = _make_eval_subset(train_loader)
        self.cl_model._task_dataloaders[task_id] = eval_loader
        device = get_device(self.cl_model.model)
        self.cl_model._eval_cache[task_id] = compute_retention(
            self.cl_model.model, eval_loader, device
        )

        return result

    def evaluate(self, **kwargs: Any) -> dict:
        """Run evaluation using the inner HF Trainer."""
        return self._trainer.evaluate(**kwargs)


class _ContinualHFTrainer(Trainer):
    """Internal Trainer subclass that adds strategy penalty to the loss."""

    def __init__(self, *args: Any, cl_strategy: Any = None, **kwargs: Any) -> None:
        self._cl_strategy = cl_strategy
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute task loss + strategy penalty + replay loss."""
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        if self._cl_strategy is not None:
            loss = loss + self._cl_strategy.penalty()
            loss = loss + self._cl_strategy.get_replay_loss(
                model, nn.CrossEntropyLoss()
            )

        if return_outputs:
            return loss, outputs
        return loss
