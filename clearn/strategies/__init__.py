"""Continual learning strategies."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from clearn.strategies.base import BaseStrategy
from clearn.strategies.der import DER
from clearn.strategies.ewc import EWC

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "ewc": EWC,
    "der": DER,
}


def get_strategy(
    name_or_instance: str | BaseStrategy,
    model: nn.Module,
    **kwargs: Any,
) -> BaseStrategy:
    """Resolve a strategy from a string name or return an existing instance.

    Args:
        name_or_instance: Strategy name (e.g. "ewc") or a BaseStrategy instance.
        model: The PyTorch model to protect.
        **kwargs: Strategy-specific keyword arguments.

    Returns:
        A configured BaseStrategy instance.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    if isinstance(name_or_instance, BaseStrategy):
        return name_or_instance

    name = name_or_instance.lower()
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown strategy '{name_or_instance}'. "
            f"Available strategies: {available}"
        )

    return STRATEGY_REGISTRY[name](model, **kwargs)


__all__ = ["BaseStrategy", "DER", "EWC", "get_strategy", "STRATEGY_REGISTRY"]
