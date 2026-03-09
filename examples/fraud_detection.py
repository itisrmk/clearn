"""Fraud detection with domain drift — clearn prevents forgetting past patterns.

Scenario: A fraud detection model trained on quarterly data. Each quarter
brings new fraud patterns, but the model must remember old ones too.
Without continual learning, training on Q2 data would erase Q1 knowledge.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import clearn


def make_fraud_data(n_samples: int, fraud_pattern_seed: int) -> DataLoader:
    """Generate synthetic fraud data with a specific pattern."""
    torch.manual_seed(fraud_pattern_seed)
    centroid = torch.randn(1, 30) * 2
    X = centroid + torch.randn(n_samples, 30) * 0.5
    # Binary: fraud (1) or legit (0)
    y = torch.randint(0, 2, (n_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=32)


# Fraud detection model
model = nn.Sequential(
    nn.Linear(30, 64),
    nn.ReLU(),
    nn.Linear(64, 2),  # 2 classes: legit, fraud
)

# Wrap with EWC to prevent forgetting past fraud patterns
cl_model = clearn.wrap(model, strategy="ewc", lambda_=5000)

# Train on quarterly data — each quarter has different fraud patterns
quarters = {
    "q1_fraud": make_fraud_data(300, seed)
    for seed, quarter in [(10, "q1"), (20, "q2"), (30, "q3"), (40, "q4")]
    for _ in [None]  # trick to use both seed and quarter
}
quarters = {
    "q1_fraud": make_fraud_data(300, 10),
    "q2_fraud": make_fraud_data(300, 20),
    "q3_fraud": make_fraud_data(300, 30),
    "q4_fraud": make_fraud_data(300, 40),
}

print("Training fraud detection model across 4 quarters...\n")
for task_id, loader in quarters.items():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cl_model.fit(loader, optimizer, epochs=5, task_id=task_id)
    print(f"  Completed: {task_id}")

# Check retention across all quarters
print("\n" + "=" * 50)
print("Retention Report — Did we forget old fraud patterns?")
print("=" * 50 + "\n")
print(cl_model.diff())
