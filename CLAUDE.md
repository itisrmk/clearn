# clearn — Claude Code Context

## What This Project Is

**clearn** is an open-source continual learning library for PyTorch models.
The name signals both "continual learning" (cl-earn) and "clean" API design.

Tagline: *"Wrap once. Train forever."*

The core problem: when you fine-tune a neural network on new data, it
catastrophically forgets previously learned tasks. clearn wraps any PyTorch
model with continual learning strategies (EWC, DER++, LoRA-EWC) that prevent
this forgetting without requiring full retraining from scratch.

This is infrastructure — the goal is Stripe-quality developer experience for
a research-grade problem. Think: `import clearn; model = clearn.wrap(your_model)`.

---

## Project Status

**Current phase:** v0.1 — building the OSS library before any hosted API.

Priority order:
1. Clean Python library that works (`pip install clearn-ai`)
2. EWC strategy (regularization-based, no replay needed)
3. DER++ strategy (replay-based, best general-purpose)
4. `diff()` retention report (the key differentiator — git diff for model knowledge)
5. HuggingFace integration
6. Benchmark notebook (sequential CIFAR-100 — this is the GitHub stars play)
7. Hosted API (FastAPI wrapper — comes after OSS traction)

---

## Repo Structure

```
clearn/
├── CLAUDE.md                        ← you are here
├── README.md                        ← write for developers, lead with code
├── pyproject.toml                   ← packaging, use hatchling
├── LICENSE                          ← MIT
├── clearn/
│   ├── __init__.py                  ← exposes: wrap, ContinualModel
│   ├── core.py                      ← ContinualModel class (main API)
│   ├── strategies/
│   │   ├── __init__.py              ← exports EWC, DER
│   │   ├── base.py                  ← abstract BaseStrategy
│   │   ├── ewc.py                   ← Elastic Weight Consolidation
│   │   └── der.py                   ← Dark Experience Replay++
│   ├── metrics.py                   ← RetentionReport, diff() logic
│   ├── integrations/
│   │   ├── __init__.py
│   │   └── huggingface.py           ← ContinualModel.from_pretrained()
│   └── utils.py                     ← shared helpers
├── benchmarks/
│   └── cifar100_sequential.ipynb    ← THE viral demo notebook
├── examples/
│   ├── quickstart.py                ← minimal 20-line working example
│   ├── llm_skills.py                ← sequential skill acquisition on LLM
│   └── fraud_detection.py           ← domain drift example (fintech)
└── tests/
    ├── test_core.py
    ├── test_ewc.py
    ├── test_der.py
    └── test_metrics.py
```

---

## The Public API (what users touch — keep this sacred)

This is the API surface. Do not change function signatures without strong reason.
Every design decision should ask: "does this feel like Stripe?"

```python
import clearn

# --- Primary interface ---

# 1. Wrap any PyTorch model
model = clearn.wrap(your_model, strategy="ewc")
model = clearn.wrap(your_model, strategy="der", buffer_size=500)

# 2. Train on sequential tasks
model.fit(dataloader, optimizer)                    # task auto-named
model.fit(dataloader, optimizer, task_id="q1_fraud")

# 3. Inspect what was retained
report = model.diff()
print(report)
# TaskRetentionReport
# ├── q1_fraud: 94.2% retained  (-5.8%)
# ├── q2_fraud: 100% (current task)
# ├── plasticity_score: 0.87
# └── recommendation: "stable — no action needed"

# 4. Save and load
model.save("./checkpoints/my_model")
model = clearn.load("./checkpoints/my_model")

# --- HuggingFace integration (v0.2) ---
model = clearn.from_pretrained("bert-base-uncased", strategy="lora-ewc")
```

---

## Core Classes

### `ContinualModel` (clearn/core.py)

The main wrapper class. Wraps any `nn.Module`.

Key methods:
- `__init__(model, strategy, **strategy_kwargs)` — wraps the model
- `fit(dataloader, optimizer, epochs=1, task_id=None)` — trains with forgetting protection
- `diff()` → `RetentionReport` — knowledge delta since last task
- `save(path)` / `load(path)` — serialize full state including task history
- `@classmethod wrap(model, strategy, **kwargs)` — alias for __init__, cleaner UX

Internal state:
- `self.model` — the underlying nn.Module
- `self.strategy` — the active BaseStrategy instance
- `self._task_history` — list of task_ids in order
- `self._eval_cache` — cached per-task evaluation results for diff()

### `BaseStrategy` (clearn/strategies/base.py)

Abstract base. All strategies implement:
- `consolidate(dataloader)` — called after each task to lock in knowledge
- `penalty()` → `torch.Tensor` — regularization loss added to task loss during training
- `update_buffer(inputs, targets)` — for replay-based strategies (optional, defaults to no-op)

### `EWC` (clearn/strategies/ewc.py)

Elastic Weight Consolidation (Kirkpatrick et al., 2017 DeepMind).

How it works: After each task, compute Fisher Information Matrix to identify
which weights matter most. During future training, penalize large changes to
those important weights. Pure regularization — no data replay needed.

Key params:
- `lambda_: float = 5000` — regularization strength. Higher = less forgetting, less plasticity.
- `n_fisher_samples: int = 200` — samples used to estimate Fisher matrix

### `DER` (clearn/strategies/der.py)

Dark Experience Replay++ (Buzzega et al., NeurIPS 2020).

How it works: Maintains a small memory buffer of past (input, logit) pairs.
During new task training, replays buffer samples and matches their original
output logits (not just labels). This preserves soft knowledge.

Key params:
- `buffer_size: int = 200` — number of past samples to store
- `alpha: float = 0.1` — weight for replay loss term
- `beta: float = 0.5` — weight for logit matching term

### `RetentionReport` (clearn/metrics.py)

Returned by `model.diff()`. The key differentiator — no other CL library ships this.

Attributes:
- `task_scores: dict[str, float]` — per-task retention percentages
- `plasticity_score: float` — how well current task was learned (0-1)
- `stability_score: float` — average retention across all past tasks (0-1)
- `recommendation: str` — human-readable advice ("increase lambda", "stable", etc.)
- `__repr__` — pretty tree-formatted output (see API example above)

---

## Implementation Guidelines

### Code style
- Python 3.10+
- Type hints everywhere, especially on public API
- Docstrings on every public method — Google style
- No external dependencies beyond: `torch`, `numpy`
- HuggingFace integration is optional: `pip install clearn-ai[hf]`

### Error handling
- Raise `ValueError` with clear messages for bad strategy names
- Raise `RuntimeError` if `diff()` called before any `fit()`
- Never silently fail — if Fisher computation fails, tell the user why

### Performance
- Fisher computation is the expensive step in EWC — cache it, never recompute unnecessarily
- DER buffer sampling should be reservoir sampling (uniform random, O(1) insert)
- `consolidate()` runs after training, not during — never slow down the training loop

### Testing philosophy
- Every strategy must pass the "forgetting test": after 5 sequential tasks,
  Task 1 accuracy must stay above 80% (vs ~30% baseline)
- Use a tiny model (2-layer MLP) and MNIST splits for fast unit tests
- The CIFAR-100 benchmark is integration/demo, not CI

---

## Key Design Decisions (don't undo these)

**1. `clearn.wrap()` not `EWCModel()` or `ContinualEWC()`**
Users should not need to know strategy names to get started. The strategy
is an implementation detail. The interface is "wrap your model."

**2. `fit()` not `train()`**
Matches scikit-learn convention. More familiar to ML practitioners.

**3. `diff()` returns an object, not a dict**
The RetentionReport has a beautiful `__repr__`. When someone calls `print(model.diff())`
in a Jupyter notebook, it should look like a polished dashboard, not a raw dict.
This is what makes people screenshot and post on Twitter.

**4. Strategy is passed as a string, not a class**
`clearn.wrap(model, strategy="ewc")` not `clearn.wrap(model, strategy=EWC())`.
Lower barrier to entry. Power users can still pass instantiated strategy objects.

**5. `consolidate()` is called automatically inside `fit()`**
Users should never have to remember to call `consolidate()` manually.
It's an internal implementation detail, not a user responsibility.

---

## Benchmark Notebook Goal (benchmarks/cifar100_sequential.ipynb)

This notebook is the distribution mechanism — it needs to be shareable,
runnable in Google Colab in one click, and produce a dramatic visual.

The experiment:
- Dataset: Split CIFAR-100 into 20 sequential tasks (5 classes each)
- Model: ResNet-18
- Baseline: Standard fine-tuning (SGD, no CL)
- clearn EWC: same model wrapped with clearn
- clearn DER++: same model wrapped with clearn
- Plot: Accuracy on Task 1 as model trains through Tasks 1→20

Expected result to show:
- Baseline: Task 1 accuracy collapses from ~85% → ~8% by Task 20
- clearn EWC: Task 1 stays ~82% through all 20 tasks
- clearn DER++: Task 1 stays ~88% through all 20 tasks

The notebook should end with a `model.diff()` call showing the final retention report.

---

## The Hosted API (future — don't build yet)

After OSS library hits 1k GitHub stars, wrap it in FastAPI.

Endpoints (v1 design, implement later):
```
POST   /v1/models                     create a new continual model
POST   /v1/models/{id}/update         push new training data
GET    /v1/models/{id}/diff           get retention report
GET    /v1/models/{id}/checkpoints    list all saved states
POST   /v1/models/{id}/rollback       restore to earlier checkpoint
```

Stack when ready: FastAPI + Modal.com (serverless GPU) + S3 + Redis job queue.
Auth: API keys only (no OAuth for v1). Rate limit by key.

---

## Founder Context (for Claude Code — do not include in public docs)

- Builder: Rahul Kashyap, PhD candidate at ASU (Information Technology)
- Research: AGNI framework — Adaptive Glucose Neural Intelligence — a continual
  learning framework for blood glucose prediction. The EWC + replay hybrid
  approach from that dissertation is the technical foundation of clearn's
  strategy router.
- Goal: OSS library → GitHub traction → seed raise ($2–4M) → hosted API → Series A
- Comparable raises: Supermemory $3M (memory API, also ASU-connected),
  Mem0 $24M (memory layer for agents)
- Positioning: "The Stripe for continual learning" — clean API, production-grade,
  developer-first
- PyPI package name: `clearn-ai` (import as `clearn`)
- Domain: clearn.ai
- The `diff()` method is the product differentiator — no other CL library ships
  a human-readable retention report. This is what goes viral.

---

## What "Done" Looks Like for v0.1

A developer can:
1. `pip install clearn-ai`
2. Copy the quickstart example
3. Run it on their own PyTorch model
4. See a `RetentionReport` printed in their terminal
5. Open a PR or issue on GitHub

That's v0.1. Ship that. Nothing else matters until that works end-to-end.
