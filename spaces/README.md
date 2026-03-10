---
title: clearn Demo
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: true
license: mit
short_description: "Prevent catastrophic forgetting in PyTorch"
---

# clearn Demo -- Continual Learning for PyTorch

**Wrap once. Train forever.**

This Space demonstrates [clearn](https://github.com/itisrmk/clearn), an open-source continual learning library for PyTorch. When you fine-tune a neural network on new data, it catastrophically forgets previously learned tasks. clearn wraps any PyTorch model with strategies that prevent this forgetting.

## What this demo does

**Tab 1 -- Train & Inspect:** Pick a continual learning strategy (EWC, SI, DER++, or GEM), configure its hyperparameters, and train a small MLP on synthetic sequential classification tasks. After training, view the `model.diff()` retention report and a per-task accuracy bar chart.

**Tab 2 -- Compare Strategies:** Run all strategies (plus a no-protection baseline) on the same data side by side. See how each strategy preserves knowledge from earlier tasks compared to naive fine-tuning.

## Quick start with clearn

```python
import clearn

model = clearn.wrap(your_model, strategy="ewc")
model.fit(task1_loader, optimizer)
model.fit(task2_loader, optimizer)
print(model.diff())
```

## Links

- [GitHub](https://github.com/itisrmk/clearn)
- [PyPI](https://pypi.org/project/clearn-ai/)
- [Documentation](https://clearn.ai)
