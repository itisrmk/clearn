"""Continual learning strategies."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from clearn.strategies.base import BaseStrategy
from clearn.strategies.der import DER
from clearn.strategies.ewc import EWC
from clearn.strategies.gem import GEM
from clearn.strategies.si import SI

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "ewc": EWC,
    "der": DER,
    "si": SI,
    "gem": GEM,
    "agem": GEM,
}

# Lazy registration for strategies with optional deps
_LAZY_STRATEGIES: dict[str, tuple[str, str]] = {
    "lora-ewc": ("clearn.strategies.lora_ewc", "LoRAEWC"),
    "lora_ewc": ("clearn.strategies.lora_ewc", "LoRAEWC"),
}


def get_strategy(
    name_or_instance: str | BaseStrategy,
    model: nn.Module,
    **kwargs: Any,
) -> BaseStrategy:
    """Resolve a strategy from a string name or return an existing instance.

    Args:
        name_or_instance: Strategy name (e.g. "ewc", "lora-ewc") or a
            BaseStrategy instance.
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

    # Check eager registry first
    if name in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[name](model, **kwargs)

    # Check lazy registry (optional deps)
    if name in _LAZY_STRATEGIES:
        module_path, class_name = _LAZY_STRATEGIES[name]
        import importlib
        module = importlib.import_module(module_path)
        strategy_cls = getattr(module, class_name)
        return strategy_cls(model, **kwargs)

    all_available = sorted(set(list(STRATEGY_REGISTRY.keys()) + list(_LAZY_STRATEGIES.keys())))
    available = ", ".join(all_available)
    raise ValueError(
        f"Unknown strategy '{name_or_instance}'. "
        f"Available strategies: {available}"
    )


__all__ = ["BaseStrategy", "DER", "EWC", "GEM", "SI", "get_strategy", "STRATEGY_REGISTRY"]
