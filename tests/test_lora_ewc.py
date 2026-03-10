"""Tests for LoRA-EWC strategy (clearn/strategies/lora_ewc.py).

These tests require peft to be installed. They are skipped if peft
is not available.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytest

peft_available = True
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    peft_available = False

pytestmark = pytest.mark.skipif(not peft_available, reason="peft not installed")


def _make_lora_model():
    """Create a simple model with LoRA applied."""
    base = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 4))
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["0", "2"], bias="none")
    return get_peft_model(base, config)


@pytest.fixture
def lora_model():
    return _make_lora_model()


@pytest.fixture
def small_loader():
    X = torch.randn(40, 20)
    y = torch.randint(0, 4, (40,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


class TestLoRAEWCBasic:
    def test_create_from_base_model(self):
        """LoRA-EWC should auto-apply LoRA to a base model."""
        from clearn.strategies.lora_ewc import LoRAEWC
        base = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 4))
        strategy = LoRAEWC(base, lora_r=4, target_modules=["0", "2"])
        # Model should now be a PeftModel
        from peft import PeftModel
        assert isinstance(strategy.model, PeftModel)

    def test_create_from_peft_model(self, lora_model):
        """LoRA-EWC should accept an existing PeftModel."""
        from clearn.strategies.lora_ewc import LoRAEWC
        strategy = LoRAEWC(lora_model)
        assert strategy.model is lora_model

    def test_only_lora_params_tracked(self, lora_model):
        """Fisher should only be computed over LoRA parameters."""
        from clearn.strategies.lora_ewc import LoRAEWC
        strategy = LoRAEWC(lora_model)
        lora_params = strategy._lora_params()
        # LoRA params should be a subset of all params
        all_params = list(lora_model.named_parameters())
        assert len(lora_params) < len(all_params)
        # All returned params should require grad
        for name, param in lora_params:
            assert param.requires_grad

    def test_penalty_before_consolidate(self, lora_model):
        from clearn.strategies.lora_ewc import LoRAEWC
        strategy = LoRAEWC(lora_model)
        assert strategy.penalty().item() == 0.0

    def test_consolidate_and_penalty(self, lora_model, small_loader):
        from clearn.strategies.lora_ewc import LoRAEWC
        strategy = LoRAEWC(lora_model)
        strategy.consolidate(small_loader)
        assert strategy._fisher is not None
        # Penalty should be 0 right after consolidation (no weight change)
        assert strategy.penalty().item() == pytest.approx(0.0, abs=1e-5)
        # Perturb LoRA weights
        with torch.no_grad():
            for name, param in strategy._lora_params():
                param.add_(torch.randn_like(param) * 0.1)
        assert strategy.penalty().item() > 0.0

    def test_state_dict_roundtrip(self, lora_model, small_loader):
        from clearn.strategies.lora_ewc import LoRAEWC
        strategy = LoRAEWC(lora_model, lambda_=3000)
        strategy.consolidate(small_loader)
        state = strategy.state_dict()

        strategy2 = LoRAEWC(lora_model)
        strategy2.load_state_dict(state)
        assert strategy2._lambda == 3000
        assert strategy2._fisher is not None


class TestLoRAEWCIntegration:
    def test_wrap_with_lora_ewc_string(self):
        """clearn.wrap with strategy='lora-ewc' should work."""
        import clearn
        base = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 4))
        cl = clearn.wrap(base, strategy="lora-ewc", lora_r=4, target_modules=["0", "2"])
        assert cl is not None
        from clearn.strategies.lora_ewc import LoRAEWC
        assert isinstance(cl.strategy, LoRAEWC)

    def test_fit_with_lora_ewc(self):
        """End-to-end fit with LoRA-EWC."""
        import clearn
        base = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 4))
        cl = clearn.wrap(base, strategy="lora-ewc", lora_r=4, target_modules=["0", "2"])
        X = torch.randn(40, 20)
        y = torch.randint(0, 4, (40,))
        loader = DataLoader(TensorDataset(X, y), batch_size=16)
        # Need optimizer for LoRA params (which are the trainable ones)
        opt = torch.optim.Adam(cl.model.parameters(), lr=0.001)
        cl.fit(loader, opt, task_id="task_0")
        assert "task_0" in cl._task_history

    def test_diff_with_lora_ewc(self):
        """diff() should work after LoRA-EWC training."""
        import clearn
        base = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 4))
        cl = clearn.wrap(base, strategy="lora-ewc", lora_r=4, target_modules=["0", "2"])
        X = torch.randn(60, 20)
        y = torch.randint(0, 4, (60,))
        loader = DataLoader(TensorDataset(X, y), batch_size=16)
        opt = torch.optim.Adam(cl.model.parameters(), lr=0.001)
        cl.fit(loader, opt, task_id="t1")
        cl.fit(loader, opt, task_id="t2")
        report = cl.diff()
        assert "t1" in report.task_scores
        assert "t2" in report.task_scores

    def test_lora_ewc_diagnostics(self):
        """get_diagnostics() should return LoRA config + Fisher stats."""
        import clearn
        base = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 4))
        cl = clearn.wrap(base, strategy="lora-ewc", lora_r=4, target_modules=["0", "2"])

        diag = cl.diagnostics()
        assert diag["strategy"] == "lora-ewc"
        assert diag["lora_r"] == 4
        assert diag["consolidated"] is False

        X = torch.randn(40, 20)
        y = torch.randint(0, 4, (40,))
        loader = DataLoader(TensorDataset(X, y), batch_size=16)
        opt = torch.optim.Adam(cl.model.parameters(), lr=0.001)
        cl.fit(loader, opt, task_id="t1")

        diag = cl.diagnostics()
        assert diag["consolidated"] is True
        assert "fisher_mean" in diag
        assert diag["total_lora_parameters"] > 0
