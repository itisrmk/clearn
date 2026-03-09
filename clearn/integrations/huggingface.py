"""HuggingFace Transformers integration for clearn.

Requires: pip install clearn-ai[hf]
"""

from __future__ import annotations

from typing import Any

try:
    from transformers import AutoModelForSequenceClassification
except ImportError:
    raise ImportError(
        "HuggingFace integration requires 'transformers'. "
        "Install with: pip install clearn-ai[hf]"
    )

from clearn.core import ContinualModel


def from_pretrained(
    model_name: str,
    strategy: str = "ewc",
    num_labels: int = 2,
    **kwargs: Any,
) -> ContinualModel:
    """Load a HuggingFace model and wrap it with continual learning.

    Args:
        model_name: HuggingFace model identifier (e.g. "bert-base-uncased").
        strategy: Continual learning strategy name. Default: "ewc".
        num_labels: Number of output labels. Default: 2.
        **kwargs: Additional keyword arguments passed to the strategy.

    Returns:
        A ContinualModel wrapping the HuggingFace model.

    Example:
        >>> model = clearn.from_pretrained("bert-base-uncased", strategy="ewc")
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return ContinualModel.wrap(model, strategy=strategy, **kwargs)
