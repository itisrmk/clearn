"""Sequential skill learning with DER++ replay buffer.

Scenario: A model learns 3 skills in sequence. DER++ stores past examples
in a replay buffer and matches their original logits to preserve knowledge.
This is the best general-purpose strategy for preventing forgetting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import clearn


def make_skill_data(skill_id: int, n_samples: int = 200) -> DataLoader:
    """Generate learnable data for a skill (cluster-based)."""
    torch.manual_seed(skill_id * 100)
    n_classes = 4
    xs, ys = [], []
    centroids = torch.randn(n_classes, 50) * 3
    for c in range(n_classes):
        x = centroids[c] + torch.randn(n_samples // n_classes, 50) * 0.3
        xs.append(x)
        ys.append(torch.full((n_samples // n_classes,), c, dtype=torch.long))
    return DataLoader(TensorDataset(torch.cat(xs), torch.cat(ys)), batch_size=32)


# Model
model = nn.Sequential(
    nn.Linear(50, 128),
    nn.ReLU(),
    nn.Linear(128, 4),
)

# Wrap with DER++ — uses a replay buffer for best retention
cl_model = clearn.wrap(model, strategy="der", buffer_size=200, alpha=0.5, beta=1.0)

skills = ["classification", "sentiment", "entity_recognition"]

print("Learning skills sequentially with DER++ replay...\n")
for i, skill in enumerate(skills):
    loader = make_skill_data(skill_id=i)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cl_model.fit(loader, optimizer, epochs=10, task_id=skill)
    print(f"  Learned: {skill}")

print(f"\nBuffer size: {len(cl_model.strategy._buffer_inputs)} samples stored")
print("\n" + "=" * 50)
print("Retention Report")
print("=" * 50 + "\n")
print(cl_model.diff())

# Note: For actual LLM fine-tuning, see clearn.from_pretrained() (coming in v0.2)
