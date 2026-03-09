"""HuggingFace sentiment analysis with continual learning.

Shows how to use clearn with a HuggingFace transformer model
to learn multiple sentiment analysis tasks without forgetting.

Requires: pip install clearn-ai[hf]

This example uses synthetic data to avoid downloading datasets.
For real usage, replace with your own tokenized datasets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("HuggingFace not installed. Run: pip install clearn-ai[hf]")
    print("Falling back to synthetic demo...\n")

import clearn


def run_synthetic_demo():
    """Run a synthetic demo without HuggingFace dependencies."""
    print("=" * 55)
    print("  Continual Sentiment Analysis (Synthetic Demo)")
    print("=" * 55)
    print()

    # Simulate a "sentiment model" with a simple MLP
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 3),  # negative, neutral, positive
    )

    # Wrap with LoRA-EWC if peft available, otherwise EWC
    try:
        cl_model = clearn.wrap(
            model, strategy="lora-ewc",
            lora_r=4, lora_alpha=8, target_modules=["0", "3"],
        )
        print("Strategy: LoRA-EWC (parameter-efficient)")
    except ImportError:
        cl_model = clearn.wrap(model, strategy="ewc", lambda_=5000)
        print("Strategy: EWC (peft not installed, using full EWC)")

    # Simulate sentiment data from different domains
    domains = {
        "product_reviews": (42, "Product reviews sentiment"),
        "movie_reviews": (123, "Movie reviews sentiment"),
        "financial_news": (456, "Financial news sentiment"),
    }

    print()
    for domain, (seed, desc) in domains.items():
        torch.manual_seed(seed)
        # Create clustered data so there's a learnable pattern
        centroids = torch.randn(3, 128) * 2
        xs, ys = [], []
        for c in range(3):
            x = centroids[c] + torch.randn(100, 128) * 0.3
            xs.append(x)
            ys.append(torch.full((100,), c, dtype=torch.long))

        X = torch.cat(xs)
        y = torch.cat(ys)
        loader = DataLoader(TensorDataset(X, y), batch_size=32)
        opt = torch.optim.Adam(cl_model.model.parameters(), lr=0.001)
        cl_model.fit(loader, opt, epochs=5, task_id=domain)
        print(f"  Trained on: {desc}")

    print()
    print("=" * 55)
    print("  Retention Report")
    print("=" * 55)
    print()
    print(cl_model.diff())


def run_hf_demo():
    """Run demo with actual HuggingFace model."""
    print("=" * 55)
    print("  Continual Sentiment Analysis (HuggingFace)")
    print("=" * 55)
    print()

    # Load a small model for demo purposes
    cl_model = clearn.from_pretrained(
        "distilbert-base-uncased",
        strategy="ewc",
        task="classification",
        num_labels=3,
        lambda_=5000,
    )
    print(f"Model: distilbert-base-uncased")
    print(f"Strategy: EWC")
    print()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Synthetic tokenized data (replace with real datasets)
    domains = ["product_reviews", "movie_reviews", "financial_news"]
    for domain in domains:
        # Create fake tokenized data
        input_ids = torch.randint(0, tokenizer.vocab_size, (100, 64))
        attention_mask = torch.ones(100, 64, dtype=torch.long)
        labels = torch.randint(0, 3, (100,))

        dataset = TensorDataset(input_ids, attention_mask, labels)

        # Custom collate to produce dict batches
        def collate_fn(batch):
            ids, mask, lab = zip(*batch)
            return {
                "input_ids": torch.stack(ids),
                "attention_mask": torch.stack(mask),
                "labels": torch.stack(lab),
            }

        loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
        opt = torch.optim.Adam(cl_model.model.parameters(), lr=2e-5)
        cl_model.fit(loader, opt, epochs=1, task_id=domain)
        print(f"  Trained on: {domain}")

    print()
    print("=" * 55)
    print("  Retention Report")
    print("=" * 55)
    print()
    print(cl_model.diff())


if __name__ == "__main__":
    # Use synthetic demo by default (faster, no model download)
    run_synthetic_demo()
