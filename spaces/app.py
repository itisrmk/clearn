"""clearn HuggingFace Spaces Gradio Demo.

Interactive demo of continual learning strategies. Train a model on
sequential tasks and inspect retention with clearn's diff() report.
"""

from __future__ import annotations

from typing import Any

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import clearn

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Small 2-hidden-layer MLP for the demo."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, n_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_sequential_tasks(
    n_tasks: int = 5,
    input_dim: int = 128,
    samples_per_task: int = 200,
    classes_per_task: int = 2,
    centroid_scale: float = 3.0,
    std: float = 0.3,
    seed: int = 42,
) -> list[DataLoader]:
    """Create synthetic sequential classification tasks with clustered centroids.

    Each task introduces ``classes_per_task`` unique classes. Data is generated
    around fixed centroids so that patterns are learnable and distinct across
    tasks.

    Returns:
        A list of DataLoaders, one per task.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    total_classes = n_tasks * classes_per_task
    # Generate fixed centroids for all classes
    centroids = torch.randn(total_classes, input_dim) * centroid_scale

    dataloaders: list[DataLoader] = []
    for task_idx in range(n_tasks):
        class_start = task_idx * classes_per_task
        class_end = class_start + classes_per_task

        all_x: list[torch.Tensor] = []
        all_y: list[torch.Tensor] = []

        samples_per_class = samples_per_task // classes_per_task
        for cls_idx in range(class_start, class_end):
            centroid = centroids[cls_idx]
            x = centroid.unsqueeze(0) + torch.randn(samples_per_class, input_dim) * std
            y = torch.full((samples_per_class,), cls_idx, dtype=torch.long)
            all_x.append(x)
            all_y.append(y)

        X = torch.cat(all_x)
        Y = torch.cat(all_y)

        # Shuffle
        perm = torch.randperm(X.size(0))
        X = X[perm]
        Y = Y[perm]

        dataset = TensorDataset(X, Y)
        dataloaders.append(DataLoader(dataset, batch_size=32, shuffle=True))

    return dataloaders


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_single_run(
    strategy: str | None,
    n_tasks: int,
    epochs: int,
    dataloaders: list[DataLoader],
    strategy_kwargs: dict[str, Any] | None = None,
) -> tuple[clearn.ContinualModel | None, list[clearn.TrainingMetrics], nn.Module]:
    """Train a model on sequential tasks.

    If ``strategy`` is None, trains a bare model with no continual learning
    protection (baseline).

    Returns:
        (continual_model_or_none, list_of_metrics, raw_model)
    """
    total_classes = n_tasks * 2
    model = MLP(input_dim=128, hidden_dim=256, n_classes=total_classes)

    if strategy_kwargs is None:
        strategy_kwargs = {}

    if strategy is not None:
        cl_model = clearn.wrap(model, strategy=strategy, **strategy_kwargs)
    else:
        cl_model = None

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    all_metrics: list[clearn.TrainingMetrics] = []

    for task_idx in range(n_tasks):
        task_id = f"task_{task_idx + 1}"
        dl = dataloaders[task_idx]

        if cl_model is not None:
            metrics = cl_model.fit(dl, optimizer, epochs=epochs, task_id=task_id)
            all_metrics.append(metrics)
        else:
            # Baseline training without clearn
            loss_fn = nn.CrossEntropyLoss()
            model.train()
            for _ in range(epochs):
                for batch_x, batch_y in dl:
                    optimizer.zero_grad()
                    out = model(batch_x)
                    loss = loss_fn(out, batch_y)
                    loss.backward()
                    optimizer.step()

    return cl_model, all_metrics, model


def evaluate_task(model: nn.Module, dataloader: DataLoader) -> float:
    """Evaluate accuracy on a single task's dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            out = model(batch_x)
            preds = out.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    model.train()
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_retention_bar(task_scores: dict[str, float], strategy_name: str) -> plt.Figure:
    """Create a bar chart of per-task retention percentages."""
    fig, ax = plt.subplots(figsize=(8, 5))

    tasks = list(task_scores.keys())
    scores = list(task_scores.values())

    colors = []
    for s in scores:
        if s >= 90:
            colors.append("#22c55e")  # green
        elif s >= 70:
            colors.append("#eab308")  # yellow
        elif s >= 50:
            colors.append("#f97316")  # orange
        else:
            colors.append("#ef4444")  # red

    bars = ax.bar(tasks, scores, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{score:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylim(0, 115)
    ax.set_ylabel("Retention (%)", fontsize=12)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_title(f"Per-Task Retention  --  Strategy: {strategy_name}", fontsize=14, fontweight="bold")
    ax.axhline(y=90, color="#94a3b8", linestyle="--", alpha=0.5, label="90% threshold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_comparison(
    results: dict[str, list[float]],
    task_labels: list[str],
) -> plt.Figure:
    """Create a grouped bar chart comparing Task 1 accuracy across methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(results.keys())
    n_methods = len(methods)
    x = np.arange(len(task_labels))
    width = 0.8 / n_methods

    palette = ["#64748b", "#3b82f6", "#8b5cf6", "#f97316", "#22c55e"]
    for i, method in enumerate(methods):
        scores = results[method]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=method, color=palette[i % len(palette)], edgecolor="white", linewidth=0.5)
        for bar, score in zip(bars, scores):
            if score > 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{score:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Strategy Comparison  --  Per-Task Accuracy After All Training", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(y=90, color="#94a3b8", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 1: Train & Inspect
# ---------------------------------------------------------------------------


def train_and_inspect(
    strategy: str,
    ewc_lambda: float,
    si_c: float,
    der_buffer_size: int,
    der_alpha: float,
    der_beta: float,
    gem_memory_size: int,
    epochs: int,
    n_tasks: int,
) -> tuple[str, plt.Figure, str]:
    """Run training with the selected strategy and return results."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Map display name to clearn strategy name
    strategy_map = {
        "EWC": "ewc",
        "SI": "si",
        "DER++": "der",
        "GEM": "gem",
    }
    strat_name = strategy_map[strategy]

    # Build strategy kwargs
    kwargs: dict[str, Any] = {}
    if strat_name == "ewc":
        kwargs["lambda_"] = ewc_lambda
    elif strat_name == "si":
        kwargs["c"] = si_c
    elif strat_name == "der":
        kwargs["buffer_size"] = int(der_buffer_size)
        kwargs["alpha"] = der_alpha
        kwargs["beta"] = der_beta
    elif strat_name == "gem":
        kwargs["memory_size"] = int(gem_memory_size)

    dataloaders = generate_sequential_tasks(n_tasks=n_tasks)
    cl_model, all_metrics, raw_model = train_single_run(
        strategy=strat_name,
        n_tasks=n_tasks,
        epochs=epochs,
        dataloaders=dataloaders,
        strategy_kwargs=kwargs,
    )

    # Generate retention report
    report = cl_model.diff()
    report_text = str(report)

    # Generate bar chart
    fig = plot_retention_bar(report.task_scores, strategy)

    # Build metrics summary
    metrics_lines = ["Training Metrics Summary", "=" * 40]
    for m in all_metrics:
        metrics_lines.append(str(m))
        metrics_lines.append("")

    metrics_text = "\n".join(metrics_lines)

    return report_text, fig, metrics_text


# ---------------------------------------------------------------------------
# Tab 2: Compare Strategies
# ---------------------------------------------------------------------------


def compare_strategies(n_tasks: int, epochs: int) -> tuple[plt.Figure, str]:
    """Run all strategies on the same data and compare."""
    torch.manual_seed(42)
    np.random.seed(42)

    dataloaders = generate_sequential_tasks(n_tasks=n_tasks)

    strategies: dict[str, dict[str, Any] | None] = {
        "Baseline (no CL)": None,
        "EWC": {"lambda_": 5000},
        "SI": {"c": 1.0},
        "DER++": {"buffer_size": 200, "alpha": 0.1, "beta": 0.5},
        "GEM": {"memory_size": 256},
    }

    strategy_name_map = {
        "Baseline (no CL)": None,
        "EWC": "ewc",
        "SI": "si",
        "DER++": "der",
        "GEM": "gem",
    }

    all_results: dict[str, list[float]] = {}
    summary_lines = ["Strategy Comparison Report", "=" * 50, ""]

    task_labels = [f"task_{i + 1}" for i in range(n_tasks)]

    for display_name, kwargs in strategies.items():
        # Regenerate data with same seed for fairness
        torch.manual_seed(42)
        np.random.seed(42)
        dl_copy = generate_sequential_tasks(n_tasks=n_tasks)

        strat_key = strategy_name_map[display_name]
        cl_model, _, raw_model = train_single_run(
            strategy=strat_key,
            n_tasks=n_tasks,
            epochs=epochs,
            dataloaders=dl_copy,
            strategy_kwargs=kwargs if kwargs else {},
        )

        # Evaluate accuracy on all tasks
        # Regenerate data for evaluation
        torch.manual_seed(42)
        np.random.seed(42)
        eval_dls = generate_sequential_tasks(n_tasks=n_tasks)

        model_to_eval = raw_model
        task_accs = []
        for i in range(n_tasks):
            acc = evaluate_task(model_to_eval, eval_dls[i]) * 100.0
            task_accs.append(acc)

        all_results[display_name] = task_accs

        # Summary
        summary_lines.append(f"--- {display_name} ---")
        if cl_model is not None:
            report = cl_model.diff()
            summary_lines.append(str(report))
        else:
            summary_lines.append("  (No continual learning protection)")
            for i, acc in enumerate(task_accs):
                summary_lines.append(f"  task_{i + 1}: {acc:.1f}%")
        avg = sum(task_accs) / len(task_accs)
        summary_lines.append(f"  Average accuracy: {avg:.1f}%")
        summary_lines.append("")

    fig = plot_comparison(all_results, task_labels)
    summary_text = "\n".join(summary_lines)

    return fig, summary_text


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    with gr.Blocks(
        title="clearn Demo -- Continual Learning for PyTorch",
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="cyan"),
    ) as app:

        gr.Markdown(
            """
            # clearn -- Continual Learning for PyTorch
            **Wrap once. Train forever.**

            Prevent catastrophic forgetting when fine-tuning neural networks on sequential tasks.
            This demo trains a small MLP on synthetic sequential classification tasks and shows
            how different continual learning strategies preserve knowledge from earlier tasks.

            `pip install clearn-ai` | [GitHub](https://github.com/rahulkashyap411/clearn) | [Docs](https://clearn.ai)
            """
        )

        with gr.Tabs():

            # ----------------------------------------------------------
            # Tab 1: Train & Inspect
            # ----------------------------------------------------------
            with gr.TabItem("Train & Inspect"):
                gr.Markdown("### Configure a strategy, train on sequential tasks, and inspect the retention report.")

                with gr.Row():
                    with gr.Column(scale=1):
                        strategy_dd = gr.Dropdown(
                            choices=["EWC", "SI", "DER++", "GEM"],
                            value="EWC",
                            label="Strategy",
                        )

                        # EWC params
                        ewc_lambda = gr.Slider(
                            minimum=100, maximum=10000, value=5000, step=100,
                            label="EWC: lambda_ (regularization strength)",
                            visible=True,
                        )

                        # SI params
                        si_c = gr.Slider(
                            minimum=0.1, maximum=10.0, value=1.0, step=0.1,
                            label="SI: c (regularization strength)",
                            visible=False,
                        )

                        # DER++ params
                        der_buffer_size = gr.Slider(
                            minimum=50, maximum=500, value=200, step=10,
                            label="DER++: buffer_size",
                            visible=False,
                        )
                        der_alpha = gr.Slider(
                            minimum=0.01, maximum=1.0, value=0.1, step=0.01,
                            label="DER++: alpha (replay CE weight)",
                            visible=False,
                        )
                        der_beta = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.5, step=0.1,
                            label="DER++: beta (logit matching weight)",
                            visible=False,
                        )

                        # GEM params
                        gem_memory_size = gr.Slider(
                            minimum=50, maximum=500, value=256, step=10,
                            label="GEM: memory_size (per task)",
                            visible=False,
                        )

                        epochs_input = gr.Number(
                            value=5, minimum=1, maximum=20, step=1,
                            label="Epochs per task",
                            precision=0,
                        )
                        n_tasks_input = gr.Number(
                            value=5, minimum=2, maximum=10, step=1,
                            label="Number of tasks",
                            precision=0,
                        )

                        train_btn = gr.Button("Train", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        report_output = gr.Textbox(
                            label="Retention Report  --  model.diff()",
                            lines=12,
                            show_copy_button=True,
                        )
                        chart_output = gr.Plot(label="Per-Task Retention")
                        metrics_output = gr.Textbox(
                            label="Training Metrics",
                            lines=15,
                            show_copy_button=True,
                        )

                # Toggle strategy-specific sliders
                def update_visibility(strategy: str) -> tuple:
                    return (
                        gr.Slider(visible=(strategy == "EWC")),
                        gr.Slider(visible=(strategy == "SI")),
                        gr.Slider(visible=(strategy == "DER++")),
                        gr.Slider(visible=(strategy == "DER++")),
                        gr.Slider(visible=(strategy == "DER++")),
                        gr.Slider(visible=(strategy == "GEM")),
                    )

                strategy_dd.change(
                    fn=update_visibility,
                    inputs=[strategy_dd],
                    outputs=[ewc_lambda, si_c, der_buffer_size, der_alpha, der_beta, gem_memory_size],
                )

                train_btn.click(
                    fn=train_and_inspect,
                    inputs=[
                        strategy_dd,
                        ewc_lambda,
                        si_c,
                        der_buffer_size,
                        der_alpha,
                        der_beta,
                        gem_memory_size,
                        epochs_input,
                        n_tasks_input,
                    ],
                    outputs=[report_output, chart_output, metrics_output],
                )

            # ----------------------------------------------------------
            # Tab 2: Compare Strategies
            # ----------------------------------------------------------
            with gr.TabItem("Compare Strategies"):
                gr.Markdown(
                    "### Run Baseline, EWC, SI, DER++, and GEM on the **same** synthetic data and compare retention."
                )

                with gr.Row():
                    compare_tasks = gr.Number(
                        value=5, minimum=2, maximum=10, step=1,
                        label="Number of tasks",
                        precision=0,
                    )
                    compare_epochs = gr.Number(
                        value=5, minimum=1, maximum=20, step=1,
                        label="Epochs per task",
                        precision=0,
                    )
                    compare_btn = gr.Button("Compare All Strategies", variant="primary", size="lg")

                compare_chart = gr.Plot(label="Strategy Comparison")
                compare_summary = gr.Textbox(
                    label="Retention Reports",
                    lines=30,
                    show_copy_button=True,
                )

                compare_btn.click(
                    fn=compare_strategies,
                    inputs=[compare_tasks, compare_epochs],
                    outputs=[compare_chart, compare_summary],
                )

        gr.Markdown(
            """
            ---
            *Built with [clearn](https://github.com/rahulkashyap411/clearn) and [Gradio](https://gradio.app).
            Strategies: EWC (Kirkpatrick et al., 2017), SI (Zenke et al., 2017),
            DER++ (Buzzega et al., 2020), GEM/A-GEM (Lopez-Paz & Ranzato, 2017).*
            """
        )

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
