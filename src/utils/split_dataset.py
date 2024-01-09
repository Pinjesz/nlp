import json
import sys

sys.path.append("/Users/piotr/Documents/studia/mgr/SEM3/NLP/nlp/src")

import torch
import pandas as pd
from data.datasets import TrainDataset
import pandas as pd
import torch

VAL_SIZE = 0.15
TEST_SIZE = 0.15

base_ds = TrainDataset()

# Calculate sizes for each split
train_size = len(base_ds) - int(len(base_ds) * (VAL_SIZE + TEST_SIZE))
val_size = int(len(base_ds) * VAL_SIZE)
test_size = len(base_ds) - train_size - val_size

# Generate indices for the entire dataset
indices = torch.randperm(len(base_ds), generator=torch.Generator())

# Split the indices
train_indices = indices[:train_size]
val_indices = indices[train_size : train_size + val_size]
test_indices = indices[train_size + val_size :]

# Verify lengths of splits
print(
    len(train_indices), len(val_indices), len(test_indices)
)  # Debugging: Check lengths

# Save indices to a CSV file
all_indices = {
    "train_indices": train_indices.tolist(),
    "val_indices": val_indices.tolist(),
    "test_indices": test_indices.tolist(),
}

# Save all indices to a single JSON file
with open("all_indices.json", "w") as file:
    json.dump(all_indices, file)
