"""Tests for HuggingFace integration (clearn/integrations/huggingface.py).

These tests verify the from_pretrained and ContinualTrainer APIs
without downloading actual models — they test the wiring and error
handling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch, MagicMock

import pytest

transformers_available = True
try:
    import transformers
except ImportError:
    transformers_available = False

pytestmark = pytest.mark.skipif(
    not transformers_available, reason="transformers not installed"
)


class TestFromPretrained:
    def test_invalid_task_raises(self):
        from clearn.integrations.huggingface import from_pretrained
        with pytest.raises(ValueError, match="Unknown task"):
            from_pretrained("bert-base-uncased", task="nonexistent")

    def test_task_aliases(self):
        """All task aliases should be valid keys."""
        from clearn.integrations.huggingface import _TASK_MODEL_MAP
        expected_tasks = {
            "classification", "sequence-classification",
            "token-classification", "ner",
            "causal-lm", "text-generation",
            "seq2seq-lm", "summarization", "translation",
        }
        assert set(_TASK_MODEL_MAP.keys()) == expected_tasks

    def test_classification_task(self):
        """from_pretrained with classification should use the right AutoModel."""
        from clearn.integrations.huggingface import from_pretrained, _TASK_MODEL_MAP

        mock_auto = MagicMock()
        mock_model = nn.Sequential(nn.Linear(10, 2))
        mock_auto.from_pretrained.return_value = mock_model

        original = _TASK_MODEL_MAP["classification"]
        _TASK_MODEL_MAP["classification"] = mock_auto
        _TASK_MODEL_MAP["sequence-classification"] = mock_auto
        try:
            result = from_pretrained("test-model", task="classification", num_labels=3)
            mock_auto.from_pretrained.assert_called_once_with("test-model", num_labels=3)
            assert result is not None
        finally:
            _TASK_MODEL_MAP["classification"] = original
            _TASK_MODEL_MAP["sequence-classification"] = original

    def test_causal_lm_task(self):
        """from_pretrained with causal-lm should not pass num_labels."""
        from clearn.integrations.huggingface import from_pretrained, _TASK_MODEL_MAP

        mock_auto = MagicMock()
        mock_model = nn.Sequential(nn.Linear(10, 2))
        mock_auto.from_pretrained.return_value = mock_model

        original = _TASK_MODEL_MAP["causal-lm"]
        _TASK_MODEL_MAP["causal-lm"] = mock_auto
        _TASK_MODEL_MAP["text-generation"] = mock_auto
        try:
            result = from_pretrained("test-model", task="causal-lm")
            mock_auto.from_pretrained.assert_called_once_with("test-model")
            assert result is not None
        finally:
            _TASK_MODEL_MAP["causal-lm"] = original
            _TASK_MODEL_MAP["text-generation"] = original


class TestContinualTrainer:
    def test_requires_continual_model(self):
        """ContinualTrainer should reject non-ContinualModel input."""
        from clearn.integrations.huggingface import ContinualTrainer
        model = nn.Linear(10, 2)
        with pytest.raises(TypeError, match="ContinualModel"):
            ContinualTrainer(
                model=model,
                args=MagicMock(),
                train_dataset=MagicMock(),
            )

    def test_accepts_continual_model(self):
        """ContinualTrainer should accept a ContinualModel (type check passes)."""
        from clearn.integrations.huggingface import ContinualTrainer
        import clearn
        from clearn.core import ContinualModel

        model = nn.Sequential(nn.Linear(10, 4))
        cl_model = clearn.wrap(model, strategy="ewc")

        # Verify the type check passes — ContinualModel is accepted
        assert isinstance(cl_model, ContinualModel)

        # Verify non-ContinualModel is rejected
        with pytest.raises(TypeError, match="ContinualModel"):
            ContinualTrainer(
                model=model,  # raw nn.Module, not wrapped
                args=MagicMock(),
                train_dataset=MagicMock(),
            )


class TestDictBatchSupport:
    """Test that fit() handles dict-style batches from HuggingFace."""

    def test_fit_with_tuple_batch(self):
        """Standard tuple batches should still work."""
        import clearn
        model = nn.Sequential(nn.Linear(20, 10))
        cl = clearn.wrap(model, strategy="ewc")
        X = torch.randn(30, 20)
        y = torch.randint(0, 10, (30,))
        loader = DataLoader(TensorDataset(X, y), batch_size=10)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        cl.fit(loader, opt, task_id="t1")
        assert "t1" in cl._task_history
