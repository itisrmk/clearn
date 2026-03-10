# Stop Your Neural Network From Forgetting: Introducing clearn

*Your model works perfectly on last quarter's data. You fine-tune it on this quarter's data. Now it's broken on last quarter's. Sound familiar?*

---

## The Problem Nobody Warns You About

Here's a scenario that plays out at every company running machine learning in production:

You train a fraud detection model on Q1 data. It performs well -- 95% accuracy, the team is happy, it goes to production. Q2 rolls around, new fraud patterns emerge, and you fine-tune the model on the new data.

Then someone checks the Q1 metrics. They've collapsed. Your model forgot everything it learned about Q1 fraud patterns the moment you trained it on Q2.

This isn't a bug in your code. It's a fundamental property of neural networks called **catastrophic forgetting**, and it's been a known problem in AI research since the 1980s. When a neural network learns new information, it overwrites the weights that stored old information. The more you train on new data, the more the old knowledge degrades.

The standard industry response? Retrain from scratch on all historical data every cycle. This works, but it's expensive, slow, and scales badly. As your data grows, your retraining window grows with it. Some teams spend more compute on retraining old tasks than learning new ones.

There's a better way. Researchers have spent decades developing algorithms that let neural networks learn continuously without forgetting. The problem is that these algorithms have been locked inside research frameworks that require deep expertise to use.

Today, I'm releasing **clearn** -- an open-source library that brings continual learning to any PyTorch model in 4 lines of code.

## What clearn Does

clearn wraps your existing PyTorch model with a continual learning strategy. You don't change your model architecture. You don't change your training loop. You wrap, train, and inspect.

```python
import clearn

# Your existing model -- any nn.Module
model = clearn.wrap(your_model, strategy="ewc")

# Train on sequential tasks
model.fit(task1_loader, optimizer, task_id="q1_fraud")
model.fit(task2_loader, optimizer, task_id="q2_fraud")

# See what your model remembers
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

That `diff()` output is the feature that makes clearn different from anything else in this space. It gives you a structured, human-readable report of exactly what your model remembers across all tasks it has ever been trained on. Plasticity score tells you how well it learned the new task. Stability score tells you how well it preserved old ones. And the recommendation field gives you plain-English guidance.

Think of it as `git diff` for model knowledge.

## How It Works: EWC Explained

The default strategy in clearn is **Elastic Weight Consolidation (EWC)**, published by Kirkpatrick et al. at DeepMind in 2017. The intuition behind it is elegant.

Imagine your neural network has millions of parameters (weights). After training on Task 1, some of those weights are critical for Task 1 performance and some are relatively unimportant. If you could identify which weights matter and protect them during future training, you could learn new tasks using the "slack" in the less important weights.

That's exactly what EWC does. After each task, it computes something called the **Fisher Information Matrix** -- a statistical measure of how sensitive each weight is to the training data. Weights with high Fisher values are important; changing them would hurt performance. Weights with low Fisher values are free to be modified.

During subsequent training, EWC adds a regularization penalty that discourages large changes to important weights. The strength of this penalty is controlled by a single parameter (`lambda`). Higher values mean stronger protection of old knowledge at the cost of reduced flexibility for new tasks.

The beauty is that this is purely a regularization technique. It doesn't require storing old training data. It doesn't require a replay buffer. It just needs the Fisher matrix, which is computed once after each task and cached.

## Beyond EWC: Five Strategies for Different Needs

clearn ships with five strategies, each suited to different situations:

**EWC (Elastic Weight Consolidation)** -- The default. Regularization-based, no replay needed. Best when you can't store old data due to privacy or storage constraints. Start here.

**SI (Synaptic Intelligence)** -- Similar to EWC but computes importance online during training rather than after. Lower overhead, slightly different trade-off profile. Good when you want something lighter than EWC.

**DER++ (Dark Experience Replay++)** -- Maintains a small buffer of past examples and their model outputs. During new training, it replays these examples and matches the model's soft outputs (logits), preserving nuanced knowledge. This is the best-performing strategy in our benchmarks, but it requires memory for the replay buffer.

**A-GEM (Averaged Gradient Episodic Memory)** -- Instead of regularization or replay, A-GEM projects the gradient during training to ensure it doesn't conflict with past task performance. It's an efficient approximation of the original GEM method.

**LoRA-EWC** -- Combines Low-Rank Adaptation with EWC for parameter-efficient continual learning. Designed for large models where full fine-tuning is impractical.

Choosing a strategy is one parameter:

```python
model = clearn.wrap(your_model, strategy="ewc")
model = clearn.wrap(your_model, strategy="der", buffer_size=500)
model = clearn.wrap(your_model, strategy="si")
model = clearn.wrap(your_model, strategy="agem")
model = clearn.wrap(your_model, strategy="lora-ewc")
```

## The Numbers

I benchmarked clearn on Split CIFAR-100 -- a standard continual learning benchmark where CIFAR-100 is divided into 20 sequential tasks of 5 classes each. The model is a ResNet-18, and each task is trained in sequence without access to previous task data (except for DER++, which uses a small replay buffer).

The results:

| Method | Task 1 Accuracy After 20 Tasks |
|--------|-------------------------------|
| Baseline (standard SGD) | ~8% |
| clearn EWC | ~82% |
| clearn DER++ | ~88% |

The baseline model's performance on Task 1 collapses almost completely by the time it finishes training on Task 20. With EWC, it retains 82% of its original accuracy. With DER++, 88%.

These results are consistent with what the original papers report. The contribution of clearn isn't algorithmic novelty -- it's making these results achievable with a four-line integration.

## HuggingFace Integration

If you work with HuggingFace models, clearn integrates directly:

```python
# Load a pretrained model with continual learning
model = clearn.from_pretrained("bert-base-uncased", strategy="lora-ewc")

# Train on your tasks
model.fit(task_loader, optimizer, task_id="sentiment_v2")

# Push back to Hub
model.push_to_hub("my-continual-bert")
```

There's also a `ContinualTrainer` class for teams that use the HuggingFace Trainer API. And a demo model is already on the Hub at [rahulmk/clearn-demo-ewc](https://huggingface.co/rahulmk/clearn-demo-ewc).

## Who Should Use This

clearn is useful whenever you have a model that needs to learn from new data over time:

- **Fraud detection systems** that see evolving patterns quarter over quarter
- **Recommendation engines** that onboard new user segments without degrading existing ones
- **NLP models** being fine-tuned with new domain knowledge or skills
- **Computer vision systems** that encounter new object categories in deployment
- **Any production ML pipeline** where periodic retraining is part of the workflow

If your current approach is "retrain from scratch on all historical data," clearn can eliminate that by letting you train incrementally while preserving past knowledge.

## What's Next

clearn is at v0.3.0. The API is stabilizing, the core strategies are solid, and the test suite covers 114 cases. Here's what's on the roadmap:

- **More strategies**: PackNet, Progressive Neural Networks, and other architecture-based approaches
- **Automatic strategy selection**: Recommending the best strategy based on your model architecture and data characteristics
- **Task-free continual learning**: Moving beyond discrete task boundaries to fully online learning
- **Expanded benchmarks**: TinyImageNet, NLP sequence tasks, and domain-specific evaluations

## Get Started

```
pip install clearn-ai
```

- **GitHub**: [github.com/itisrmk/clearn](https://github.com/itisrmk/clearn)
- **PyPI**: [pypi.org/project/clearn-ai](https://pypi.org/project/clearn-ai/)
- **HuggingFace**: [huggingface.co/rahulmk/clearn-demo-ewc](https://huggingface.co/rahulmk/clearn-demo-ewc)

The library is MIT licensed. Contributions, issues, and feedback are welcome.

---

*clearn is built by Rahul Kashyap, a PhD candidate at Arizona State University researching continual learning for medical AI. The library grew out of production research infrastructure and is designed for ML engineers who need continual learning to work, not just in notebooks, but in production.*
