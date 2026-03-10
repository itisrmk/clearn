# Reddit r/pytorch Post

---

**Title:** I built clearn -- a library that wraps any PyTorch model with continual learning in 4 lines of code. No architecture changes needed.

---

**Body:**

Hey r/pytorch,

Ever had a model that works great on your training data, then completely falls apart the moment you fine-tune it on new data? That's catastrophic forgetting, and it's one of the most annoying problems in production ML.

I built **clearn** to fix it with minimal friction. It wraps any `nn.Module` -- your model, your architecture, no changes needed.

## The 4-Step Quickstart

```python
import clearn

# 1. Wrap your existing model (any nn.Module)
model = clearn.wrap(your_model, strategy="ewc")

# 2. Train on your first task
model.fit(task1_loader, optimizer, task_id="q1_fraud")

# 3. Train on new data -- old knowledge is protected
model.fit(task2_loader, optimizer, task_id="q2_fraud")

# 4. See what your model remembers
print(model.diff())
```

Output:

```
RetentionReport
├── q1_fraud: 94.2% retained  (-5.8%)
├── q2_fraud: 100.0% (current task)
├── plasticity_score: 0.87
├── stability_score: 0.94
└── recommendation: "stable — no action needed"
```

That `diff()` output is the feature I find most useful day-to-day. It's like `git diff` but for what your model knows. You can see at a glance which tasks degraded and by how much.

## Install

```
pip install clearn-ai
```

## What Strategies Are Available?

You pick a strategy with a string -- no need to import extra classes:

```python
model = clearn.wrap(my_model, strategy="ewc")      # Regularization-based
model = clearn.wrap(my_model, strategy="si")        # Online, lightweight
model = clearn.wrap(my_model, strategy="der", buffer_size=500)  # Replay-based, best results
model = clearn.wrap(my_model, strategy="agem")      # Gradient projection
model = clearn.wrap(my_model, strategy="lora-ewc")  # Parameter-efficient for large models
```

If you're not sure, start with `"ewc"` -- it's the simplest and works well for most cases. If you need better retention and can spare some memory for a replay buffer, use `"der"`.

## Real Benchmark Numbers

Split CIFAR-100 into 20 sequential tasks, ResNet-18:

- **Baseline SGD:** Task 1 accuracy drops from ~85% to **~8%** by Task 20
- **clearn EWC:** Task 1 stays at **~82%**
- **clearn DER++:** Task 1 stays at **~88%**

## HuggingFace Integration

If you work with HuggingFace models:

```python
model = clearn.from_pretrained("bert-base-uncased", strategy="lora-ewc")
model.fit(task_loader, optimizer, task_id="sentiment_v2")
model.push_to_hub("my-continual-bert")
```

## Practical Use Cases

- **Fraud detection:** Train on Q1 fraud patterns, then Q2, Q3... without forgetting old patterns
- **Recommendation systems:** Add new user cohorts without degrading existing ones
- **NLP fine-tuning:** Add new skills to a language model without losing previous capabilities
- **Any production model** that gets retrained on new data periodically

## Details

- Works with any `nn.Module` -- CNNs, transformers, MLPs, whatever
- `fit()` handles consolidation automatically -- you don't need to manage strategy internals
- Save/load preserves full state including task history: `model.save("path")` / `clearn.load("path")`
- Pure PyTorch + numpy, no other dependencies
- MIT licensed, 114 tests passing, v0.3.0

## Links

- **GitHub:** https://github.com/itisrmk/clearn
- **PyPI:** https://pypi.org/project/clearn-ai/
- **HuggingFace demo:** https://huggingface.co/rahulmk/clearn-demo-ewc

Would love feedback from the community. What strategies or features would you want to see? PRs are welcome.
