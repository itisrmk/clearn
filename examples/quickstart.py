"""clearn quickstart — 20 lines to prevent catastrophic forgetting."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import clearn

# Your model
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 4))

# Wrap it with continual learning (EWC strategy)
cl_model = clearn.wrap(model, strategy="ewc", lambda_=5000)

# Simulate two sequential tasks with different data distributions
for task_id in ["task_a", "task_b"]:
    X = torch.randn(200, 20)
    y = torch.randint(0, 4, (200,))
    loader = DataLoader(TensorDataset(X, y), batch_size=32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    cl_model.fit(loader, optimizer, epochs=3, task_id=task_id)

# See what was retained
print(cl_model.diff())
