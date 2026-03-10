# Reddit r/MachineLearning Post

**Flair:** [Project]

---

**Title:** [Project] clearn: Open-source continual learning library for PyTorch -- wrap any model, prevent catastrophic forgetting, get retention reports with diff()

---

**Body:**

## What is clearn?

clearn is an open-source Python library that wraps any PyTorch model with continual learning strategies to prevent catastrophic forgetting. The API is designed for practitioners who need to train models on sequential tasks without full retraining.

```
pip install clearn-ai
```

GitHub: https://github.com/itisrmk/clearn | MIT License | v0.3.0 | 114 tests passing

## Why another CL library?

Existing options (Avalanche, Continuum) are research frameworks -- powerful but heavy. clearn is built as infrastructure: a minimal wrapper with a clean API that works on any `nn.Module`. The design goal was "Stripe for continual learning" -- the smallest possible surface area that solves the problem.

The key differentiator is `model.diff()`, which returns a structured retention report showing exactly what your model remembers across all previous tasks:

```
RetentionReport
├── q1_fraud: 94.2% retained  (-5.8%)
├── q2_fraud: 100.0% (current task)
├── plasticity_score: 0.87
├── stability_score: 0.94
└── recommendation: "stable — no action needed"
```

No other CL library ships this kind of observability out of the box.

## Implemented Strategies

Five strategies spanning the major families of continual learning approaches:

**Regularization-based:**
- **EWC** (Elastic Weight Consolidation) -- Kirkpatrick et al., PNAS 2017. Computes Fisher Information Matrix after each task to identify important weights, then penalizes changes to them during subsequent training. No replay buffer needed.
- **SI** (Synaptic Intelligence) -- Zenke et al., ICML 2017. Online alternative to EWC that accumulates parameter importance during training rather than computing it post-hoc. Lower computational overhead.

**Replay-based:**
- **DER++** (Dark Experience Replay++) -- Buzzega et al., NeurIPS 2020. Maintains a memory buffer of (input, logit) pairs and replays them during training, matching soft outputs rather than hard labels. Best general-purpose strategy in our benchmarks.

**Gradient-based:**
- **A-GEM** (Averaged Gradient Episodic Memory) -- Chaudhry et al., ICLR 2019, building on GEM from Lopez-Paz & Ranzato, NeurIPS 2017. Projects gradients to avoid interfering with past task performance. Efficient approximation of the original GEM constraints.

**Parameter-efficient:**
- **LoRA-EWC** -- Combines Low-Rank Adaptation with EWC regularization for large model continual learning. Useful when full fine-tuning is too expensive.

## API

```python
import clearn

# Wrap any nn.Module
model = clearn.wrap(your_model, strategy="ewc")

# Train sequentially -- consolidation happens automatically
model.fit(task1_loader, optimizer, task_id="q1_fraud")
model.fit(task2_loader, optimizer, task_id="q2_fraud")

# Inspect retention
report = model.diff()
print(report)

# Save/load full state including task history
model.save("./checkpoints/my_model")
model = clearn.load("./checkpoints/my_model")
```

HuggingFace integration:

```python
model = clearn.from_pretrained("bert-base-uncased", strategy="lora-ewc")
model.fit(task_loader, optimizer, task_id="sentiment_v2")
model.push_to_hub("my-continual-bert")
```

## Benchmark Results

Split CIFAR-100 (20 sequential tasks, 5 classes each, ResNet-18):

| Method | Task 1 Accuracy After 20 Tasks | Forgetting |
|--------|-------------------------------|------------|
| Baseline SGD | ~8% | Catastrophic |
| clearn EWC (lambda=5000) | ~82% | -3% avg |
| clearn DER++ (buffer=500) | ~88% | -1.2% avg |

These numbers are consistent with the results reported in the original papers. The point isn't that we beat SOTA -- it's that you get these results with 4 lines of code.

## Technical Details

- Pure PyTorch dependency (+ numpy). No framework lock-in.
- Fisher computation in EWC is cached and runs post-training, never during the training loop.
- DER++ buffer uses reservoir sampling for O(1) insertion.
- `consolidate()` is called automatically inside `fit()` -- users never touch it.
- Strategy is passed as a string for simplicity, but power users can instantiate and pass strategy objects directly.
- Python 3.10+, type hints throughout, Google-style docstrings.

## What's Next

- More strategies (PackNet, Progressive Neural Networks)
- Automatic strategy selection based on model architecture and data characteristics
- Benchmark suite expansion (TinyImageNet, NLP sequence tasks)
- Hosted API for teams that want CL-as-a-service

## Background

I'm a PhD candidate at ASU working on continual learning for medical AI (specifically adaptive glucose prediction systems). This library grew out of wanting a cleaner interface for the CL methods I use daily in my research.

Happy to answer questions about implementation details, strategy selection, or the benchmark setup.

---

**Links:**
- GitHub: https://github.com/itisrmk/clearn
- PyPI: https://pypi.org/project/clearn-ai/
- HuggingFace demo model: https://huggingface.co/rahulmk/clearn-demo-ewc
