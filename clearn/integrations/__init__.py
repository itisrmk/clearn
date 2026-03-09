"""Optional integrations for clearn."""

from __future__ import annotations

try:
    from clearn.integrations.huggingface import from_pretrained

    __all__ = ["from_pretrained"]
except ImportError:
    def from_pretrained(*args, **kwargs):
        raise ImportError(
            "HuggingFace integration requires additional dependencies. "
            "Install with: pip install clearn-ai[hf]"
        )

    __all__ = ["from_pretrained"]
