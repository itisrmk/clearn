<p align="center">
  <h1 align="center">clearn</h1>
  <p align="center">
    <strong>Wrap once. Train forever.</strong>
  </p>
  <p align="center">
    Continual learning for PyTorch models.<br>
    Prevent catastrophic forgetting with one line of code.
  </p>
  <p align="center">
    <a href="https://pypi.org/project/clearn-ai/"><img src="https://img.shields.io/pypi/v/clearn-ai?color=blue&label=PyPI" alt="PyPI"></a>
    <a href="https://pypi.org/project/clearn-ai/"><img src="https://img.shields.io/pypi/pyversions/clearn-ai" alt="Python"></a>
    <a href="https://github.com/itisrmk/clearn/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/itisrmk/clearn/actions"><img src="https://img.shields.io/badge/tests-66%20passed-brightgreen" alt="Tests"></a>
  </p>
</p>

---

When you fine-tune a neural network on new data, it **catastrophically forgets** what it learned before. clearn fixes this. Wrap any PyTorch model, train on sequential tasks, and your model remembers everything.

```python
import clearn

model = clearn.wrap(your_model, strategy="ewc")

model.fit(task1_loader, optimizer, task_id="q1_fraud")
model.fit(task2_loader, optimizer, task_id="q2_fraud")

print(model.diff())
```

```
RetentionReport
├── q1_fraud: 94.2% retained  (-5.8%)
├── q2_fraud: 100.0% (current task)
├── plasticity_score: 0.87
├── stability_score: 0.94
└── recommendation: "stable — no action needed"
```

---

## Installation

```bash
pip install clearn-ai
```

For HuggingFace integration:

```bash
pip install clearn-ai[hf]
```

---

## Quickstart

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import clearn

# 1. Your PyTorch model
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

# 2. Wrap it — one line
cl_model = clearn.wrap(model, strategy="ewc")

# 3. Train on sequential tasks
for i, task_data in enumerate(sequential_tasks):
    loader = DataLoader(task_data, batch_size=64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    cl_model.fit(loader, optimizer, task_id=f"task_{i}")

# 4. See what was retained
print(cl_model.diff())
```

That's it. Four steps. Your model now remembers.

---

## Why clearn?

| Problem | Without clearn | With clearn |
|---------|---------------|-------------|
| Train on Task 2 | Task 1 accuracy: **8%** | Task 1 accuracy: **94%** |
| Train on 20 tasks | First task: **destroyed** | First task: **preserved** |
| Debug forgetting | Print loss, guess | `model.diff()` tells you exactly |

---

## Strategies

clearn ships three strategies:

### EWC (Elastic Weight Consolidation)

Regularization-based. Identifies which weights matter most, then protects them during future training. No need to store past data.

```python
model = clearn.wrap(net, strategy="ewc", lambda_=5000)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_` | `5000` | Regularization strength. Higher = less forgetting, less plasticity |
| `n_fisher_samples` | `200` | Samples used to estimate weight importance |

### DER++ (Dark Experience Replay)

Replay-based. Stores a small buffer of past examples and replays them during training, matching original logits. Best general-purpose performance.

```python
model = clearn.wrap(net, strategy="der", buffer_size=500)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_size` | `200` | Number of past samples to store |
| `alpha` | `0.1` | Weight for cross-entropy replay loss |
| `beta` | `0.5` | Weight for logit-matching loss |

### LoRA-EWC (Parameter-Efficient Continual Learning)

**New in v0.2.** Combines LoRA adapters (via `peft`) with EWC regularization. Only the low-rank adapter weights are trained and protected — the base model stays frozen. Ideal for LLMs.

```python
# Requires: pip install clearn-ai[hf]
model = clearn.from_pretrained("bert-base-uncased", strategy="lora-ewc", lora_r=8)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | `8` | LoRA rank (lower = more efficient) |
| `lora_alpha` | `16` | LoRA alpha scaling |
| `lambda_` | `5000` | EWC regularization on LoRA weights |

### Which strategy should I use?

```
Using a large language model?
├── Yes  → LoRA-EWC (parameter-efficient + forgetting protection)
└── No   → Can you store past data?
           ├── Yes  → DER++ (best retention)
           └── No   → EWC (no replay needed)
```

---

## The `diff()` Report

The key feature. Like `git diff`, but for model knowledge.

```python
report = model.diff()
print(report)
```

```
RetentionReport
├── task_a: 94.2% retained  (-5.8%)
├── task_b: 88.1% retained  (-11.9%)
├── task_c: 100.0% (current task)
├── plasticity_score: 0.91
├── stability_score: 0.91
└── recommendation: "stable — no action needed"
```

The report gives you:
- **Per-task retention** — exactly how much each task was preserved
- **Plasticity score** — how well the latest task was learned
- **Stability score** — average retention across all past tasks
- **Recommendation** — actionable advice ("increase lambda", "try DER++", etc.)

---

## Save & Load

```python
# Save full state (model + strategy + task history)
model.save("./checkpoints/my_model")

# Load it back
model = clearn.load("./checkpoints/my_model", model=your_model)
```

---

## Examples

### Fraud Detection with Domain Drift

```python
import clearn

model = clearn.wrap(fraud_model, strategy="ewc")

model.fit(q1_data, optimizer, task_id="q1_fraud")
model.fit(q2_data, optimizer, task_id="q2_fraud")
model.fit(q3_data, optimizer, task_id="q3_fraud")

# Did we forget Q1 fraud patterns?
print(model.diff())
```

### Sequential Skill Learning (DER++)

```python
model = clearn.wrap(skill_model, strategy="der", buffer_size=500)

for skill_name, skill_data in skills.items():
    model.fit(skill_data, optimizer, task_id=skill_name)

print(model.diff())  # All skills retained
```

See the [`examples/`](examples/) directory for runnable scripts.

---

## Benchmark: CIFAR-100 Sequential

Split CIFAR-100 into 20 tasks. Train a ResNet-18 on each. Track Task 1 accuracy.

| Method | Task 1 Accuracy (after 20 tasks) |
|--------|----------------------------------|
| Baseline (SGD) | ~8% |
| clearn EWC | ~82% |
| clearn DER++ | ~88% |

Run the benchmark yourself:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/itisrmk/clearn/blob/main/benchmarks/cifar100_sequential.ipynb)

---

## HuggingFace Integration

**New in v0.2.** First-class support for HuggingFace Transformers.

```python
# Load any HuggingFace model with continual learning
model = clearn.from_pretrained("bert-base-uncased", strategy="ewc", task="classification")
model = clearn.from_pretrained("gpt2", strategy="lora-ewc", task="causal-lm")

# Supported tasks: classification, token-classification, causal-lm, seq2seq-lm
```

**ContinualTrainer** — drop-in replacement for HuggingFace Trainer:

```python
from clearn.integrations.huggingface import ContinualTrainer

trainer = ContinualTrainer(
    model=cl_model,
    args=training_args,
    train_dataset=dataset,
    task_id="sentiment_v1",
)
trainer.train()  # Automatically applies forgetting protection
```

**Dict-batch support** — `fit()` handles HuggingFace-style dict batches natively:

```python
# Works with both (tensor, tensor) tuples and {"input_ids": ..., "labels": ...} dicts
cl_model.fit(hf_dataloader, optimizer, task_id="my_task")
```

---

## API Reference

```python
import clearn

# Wrap any PyTorch model
model = clearn.wrap(model, strategy="ewc", **kwargs)

# Train on a task
model.fit(dataloader, optimizer, epochs=1, task_id=None, loss_fn=None)

# Get retention report
report = model.diff()

# Save / Load
model.save("path/to/checkpoint")
model = clearn.load("path/to/checkpoint", model=your_model)

# HuggingFace (requires clearn-ai[hf])
model = clearn.from_pretrained("bert-base-uncased", strategy="ewc", task="classification")
model = clearn.from_pretrained("gpt2", strategy="lora-ewc", task="causal-lm", lora_r=8)
```

---

## Project Structure

```
clearn/
├── clearn/
│   ├── core.py              # ContinualModel — the main wrapper
│   ├── strategies/
│   │   ├── base.py           # Abstract strategy interface
│   │   ├── ewc.py            # Elastic Weight Consolidation
│   │   ├── der.py            # Dark Experience Replay++
│   │   └── lora_ewc.py       # LoRA + EWC hybrid (v0.2)
│   ├── metrics.py            # RetentionReport & diff() logic
│   └── integrations/
│       └── huggingface.py    # from_pretrained() + ContinualTrainer
├── tests/                    # 66 tests, all passing
├── examples/                 # Runnable demo scripts
└── benchmarks/               # CIFAR-100 notebook
```

---

## Contributing

```bash
git clone https://github.com/itisrmk/clearn.git
cd clearn
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT

---

<p align="center">
  Built by <a href="https://github.com/itisrmk">Rahul Kashyap</a><br>
  <sub>Continual learning infrastructure for production ML</sub>
</p>
