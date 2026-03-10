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
    <a href="https://github.com/itisrmk/clearn/actions"><img src="https://img.shields.io/badge/tests-114%20passed-brightgreen" alt="Tests"></a>
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
    metrics = cl_model.fit(loader, optimizer, task_id=f"task_{i}")
    print(f"Task {i}: loss={metrics.final_loss:.4f}, acc={metrics.final_accuracy:.2%}")

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

clearn ships five strategies:

### EWC (Elastic Weight Consolidation)

Regularization-based. Identifies which weights matter most via the Fisher Information Matrix, then protects them during future training. No need to store past data.

```python
model = clearn.wrap(net, strategy="ewc", lambda_=5000)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_` | `5000` | Regularization strength. Higher = less forgetting, less plasticity |
| `n_fisher_samples` | `200` | Samples used to estimate weight importance |

### SI (Synaptic Intelligence)

Online importance estimation. Tracks per-parameter contribution to loss reduction during training, then penalizes changes to important weights. No separate computation pass needed — importance is accumulated during training.

```python
model = clearn.wrap(net, strategy="si", c=1.0)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `c` | `1.0` | Regularization strength (analogous to EWC's lambda) |
| `epsilon` | `1e-3` | Numerical stability constant |

### DER++ (Dark Experience Replay)

Replay-based. Stores a small buffer of past examples and replays them during training, matching original logits via KL divergence with temperature scaling. Best general-purpose performance.

```python
model = clearn.wrap(net, strategy="der", buffer_size=500)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_size` | `200` | Number of past samples to store |
| `alpha` | `0.1` | Weight for cross-entropy replay loss |
| `beta` | `0.5` | Weight for KL divergence logit-matching loss |
| `temperature` | `2.0` | Temperature for KL divergence softmax |
| `buffer_device` | `"cpu"` | Device to store buffer on (`"cuda"` avoids transfers) |

### GEM (Gradient Episodic Memory)

Constraint-based. Stores episodic memories from past tasks and projects gradients to avoid increasing loss on any previous task. Uses the efficient A-GEM variant.

```python
model = clearn.wrap(net, strategy="gem", memory_size=256)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `memory_size` | `256` | Samples to store per task |

### LoRA-EWC (Parameter-Efficient Continual Learning)

Combines LoRA adapters (via `peft`) with EWC regularization. Only the low-rank adapter weights are trained and protected — the base model stays frozen. Ideal for LLMs.

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
           └── No   → Do you need online tracking?
                      ├── Yes  → SI (no Fisher pass needed)
                      └── No   → Want hard constraints?
                                 ├── Yes  → GEM (gradient projection)
                                 └── No   → EWC (classic, reliable)
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

## Training Metrics

Every `fit()` call returns detailed metrics:

```python
metrics = model.fit(loader, optimizer, task_id="q1", epochs=5)
print(metrics)
```

```
TrainingMetrics(task='q1')
├── epochs: 5
├── final_loss: 0.3421
├── final_accuracy: 91.20%
└── wall_time: 2.15s
```

Access per-epoch data: `metrics.epoch_losses`, `metrics.epoch_accuracies`.

---

## Strategy Diagnostics

Inspect the internals of your strategy at any time:

```python
diag = model.diagnostics()
# EWC example:
# {'strategy': 'ewc', 'lambda': 5000, 'consolidated': True,
#  'fisher_mean': 0.0023, 'fisher_max': 10000.0, 'current_penalty': 42.5, ...}

# DER++ example:
# {'strategy': 'der++', 'buffer_used': 200, 'buffer_utilization': 1.0,
#  'buffer_class_distribution': {0: 45, 1: 38, ...}, ...}
```

---

## Callbacks

Hook into training with the callback system:

```python
from clearn import ContinualCallback

class LogCallback(ContinualCallback):
    def on_task_start(self, model, task_id):
        print(f"Starting {task_id}")

    def on_batch_end(self, model, loss):
        pass  # Log to wandb, etc.

    def on_task_end(self, model, task_id, metrics):
        print(f"Finished {task_id}: {metrics.final_accuracy:.2%}")

model.fit(loader, optimizer, callbacks=[LogCallback()])
```

Built-in: `EarlyStoppingCallback(patience=50)`.

---

## Gradient Clipping & Mixed Precision

```python
# Gradient clipping
model.fit(loader, optimizer, grad_clip=1.0)

# Mixed precision (AMP) — requires CUDA
model.fit(loader, optimizer, use_amp=True)

# Both
model.fit(loader, optimizer, grad_clip=1.0, use_amp=True)
```

---

## Save & Load

```python
# Save full state (model + strategy + task history)
model.save("./checkpoints/my_model")

# Load it back — diff() works after load
model = clearn.load("./checkpoints/my_model", model=your_model)
print(model.diff())  # Retention report preserved
```

---

## HuggingFace Integration

First-class support for HuggingFace Transformers.

```python
# Load any HuggingFace model with continual learning
model = clearn.from_pretrained("bert-base-uncased", strategy="ewc", task="classification")
model = clearn.from_pretrained("gpt2", strategy="lora-ewc", task="causal-lm")

# Get the tokenizer too
model, tokenizer = clearn.from_pretrained(
    "bert-base-uncased", strategy="ewc", return_tokenizer=True
)

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

**Push to HuggingFace Hub:**

```python
model.push_to_hub("your-username/my-continual-model")
```

---

## API Reference

```python
import clearn

# Wrap any PyTorch model
model = clearn.wrap(model, strategy="ewc", **kwargs)

# Train on a task (returns TrainingMetrics)
metrics = model.fit(dataloader, optimizer, epochs=1, task_id=None,
                    loss_fn=None, grad_clip=None, callbacks=None, use_amp=False)

# Get retention report
report = model.diff()

# Get strategy diagnostics
diag = model.diagnostics()

# Save / Load (diff() works after load)
model.save("path/to/checkpoint")
model = clearn.load("path/to/checkpoint", model=your_model)

# HuggingFace (requires clearn-ai[hf])
model = clearn.from_pretrained("bert-base-uncased", strategy="ewc", task="classification")
model, tokenizer = clearn.from_pretrained("gpt2", strategy="lora-ewc",
                                           task="causal-lm", return_tokenizer=True)
model.push_to_hub("user/model-name")
```

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

## Project Structure

```
clearn/
├── clearn/
│   ├── core.py              # ContinualModel — the main wrapper
│   ├── strategies/
│   │   ├── base.py           # Abstract strategy interface
│   │   ├── ewc.py            # Elastic Weight Consolidation
│   │   ├── si.py             # Synaptic Intelligence
│   │   ├── der.py            # Dark Experience Replay++
│   │   ├── gem.py            # Gradient Episodic Memory (A-GEM)
│   │   └── lora_ewc.py       # LoRA + EWC hybrid
│   ├── metrics.py            # RetentionReport, TrainingMetrics, diff() logic
│   ├── callbacks.py          # ContinualCallback, EarlyStoppingCallback
│   └── integrations/
│       └── huggingface.py    # from_pretrained(), ContinualTrainer, push_to_hub
├── tests/                    # 114 tests, all passing
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
