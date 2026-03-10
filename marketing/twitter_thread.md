# Twitter/X Thread: clearn Launch

**Post as a thread. Each section below = one tweet. Keep images/screenshots where noted.**

---

**Tweet 1 (Hook):**

Your neural network forgets everything it learned the moment you train it on new data.

It's called catastrophic forgetting, and it's why most production ML pipelines are a mess of full retrains.

I built an open-source library to fix it in 4 lines of code.

Thread:

---

**Tweet 2:**

Meet clearn -- "Wrap once. Train forever."

```python
import clearn

model = clearn.wrap(your_model, strategy="ewc")
```

That's it. Your PyTorch model now has continual learning superpowers. No architecture changes. No full retrains. Just wrap and keep training.

---

**Tweet 3:**

Train on sequential tasks, and your model remembers ALL of them:

```python
model.fit(task1_loader, optimizer, task_id="q1_fraud")
model.fit(task2_loader, optimizer, task_id="q2_fraud")
model.fit(task3_loader, optimizer, task_id="q3_fraud")
```

Each call protects knowledge from previous tasks automatically.

---

**Tweet 4:**

But here's the feature I'm most proud of -- `model.diff()`

It gives you a git-diff-style report of what your model remembers:

```
RetentionReport
├── q1_fraud: 94.2% retained  (-5.8%)
├── q2_fraud: 100.0% (current task)
├── plasticity_score: 0.87
├── stability_score: 0.94
└── recommendation: "stable — no action needed"
```

---

**Tweet 5:**

No other continual learning library ships this.

`diff()` tells you exactly how much each task degraded, whether your model is stable, and what to do about it.

It's like `git diff` but for model knowledge.

---

**Tweet 6:**

The benchmarks speak for themselves.

CIFAR-100, split into 20 sequential tasks:

- Baseline SGD: Task 1 accuracy drops to ~8%
- clearn EWC: Task 1 stays at ~82%
- clearn DER++: Task 1 stays at ~88%

That's the difference between a usable model and a broken one.

---

**Tweet 7:**

5 strategies, pick what fits your problem:

- EWC -- regularization, no replay needed
- SI -- online, lightweight
- DER++ -- replay-based, best general-purpose
- A-GEM -- constrained gradient updates
- LoRA-EWC -- parameter-efficient for large models

All accessible through one API: `clearn.wrap(model, strategy="...")`

---

**Tweet 8:**

HuggingFace integration is first-class:

```python
model = clearn.from_pretrained(
    "bert-base-uncased",
    strategy="lora-ewc"
)

# Train, then share
model.push_to_hub("my-continual-bert")
```

Load pretrained, train continually, push back to Hub.

---

**Tweet 9:**

The full quickstart is 5 lines:

```python
import clearn
model = clearn.wrap(your_model, strategy="ewc")
model.fit(task1_loader, optimizer, task_id="q1")
model.fit(task2_loader, optimizer, task_id="q2")
print(model.diff())
```

That's a production-ready continual learning pipeline.

---

**Tweet 10:**

Built this during my PhD at ASU, where I work on continual learning for medical AI.

The research is real. The API is clean. The library is MIT licensed.

114 tests passing. v0.3.0.

---

**Tweet 11:**

Try it now:

```
pip install clearn-ai
```

GitHub: github.com/itisrmk/clearn
PyPI: pypi.org/project/clearn-ai/
HuggingFace demo: huggingface.co/rahulmk/clearn-demo-ewc

Star the repo if this is useful. PRs welcome.

---

**Tweet 12 (CTA):**

If you've ever had to retrain a model from scratch because it forgot old data -- clearn is for you.

Wrap once. Train forever.

github.com/itisrmk/clearn
