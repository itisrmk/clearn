# Hacker News: Show HN

---

**Title:** Show HN: clearn -- Continual learning for any PyTorch model in 4 lines of code

**URL:** https://github.com/itisrmk/clearn

---

**First Comment (post immediately after submission):**

Hi HN, I'm Rahul. I'm a PhD candidate at ASU working on continual learning for medical AI, and I built clearn because I was frustrated with the gap between CL research and practical usability.

**The problem:** When you fine-tune a neural network on new data, it catastrophically forgets what it previously learned. This is well-studied in research (Kirkpatrick et al. 2017, Buzzega et al. NeurIPS 2020) but the existing tools are research frameworks designed for running experiments, not for wrapping a production model.

**What clearn does:**

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

You wrap any `nn.Module`, train sequentially with `fit()`, and inspect retention with `diff()`. The strategy handles everything internally -- Fisher matrix computation for EWC, replay buffer management for DER++, gradient projection for A-GEM.

**What it does well:**

- Clean API -- 4 lines to get started, strategy is a string parameter
- `diff()` gives you observability into what your model remembers, which I haven't seen in other CL libraries
- 5 strategies covering the main CL families (regularization, replay, gradient-based, parameter-efficient)
- HuggingFace integration for loading pretrained models and pushing to Hub
- 114 tests, MIT license, pure PyTorch dependency

**Honest limitations:**

- This wraps existing CL algorithms, it doesn't invent new ones. The strategies are implementations of published methods (EWC, SI, DER++, A-GEM, LoRA-EWC).
- `diff()` requires evaluation data for each task to compute retention scores. If you don't have held-out eval data per task, the report will be limited.
- The library assumes discrete task boundaries -- you call `fit()` per task. Truly task-free / online continual learning is not supported yet.
- Benchmarked on standard vision tasks (CIFAR-100). I have not yet run large-scale NLP benchmarks, though HuggingFace integration is functional.
- v0.3.0 -- the API is stabilizing but I won't promise no breaking changes before v1.0.
- Single-GPU only for now. No distributed training support.

**Benchmark (Split CIFAR-100, 20 tasks, ResNet-18):**

- Baseline: Task 1 drops to ~8% by Task 20
- EWC: Task 1 stays at ~82%
- DER++: Task 1 stays at ~88%

These match expected results from the original papers. The value-add is accessibility, not algorithmic novelty.

**Why I built it:**

My PhD research involves continual learning for blood glucose prediction -- patients' physiology changes over time, and models need to adapt without forgetting baseline patterns. I kept rebuilding the same CL scaffolding for every experiment and eventually extracted it into a library. The name is a portmanteau of "continual learning" and "clean."

Install: `pip install clearn-ai`

Happy to answer technical questions about the implementation or the underlying algorithms.
