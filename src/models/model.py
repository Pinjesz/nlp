import lightning as L
import numpy as np
from torch.nn import functional as F, Linear
from torch.optim import AdamW
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
    OptimizerLRScheduler,
)
import random
from torch.utils.data import DataLoader, Dataset
import torch
from data.tokenize_data import get_tokenized_data

class RandomModel(L.LightningModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.layer = Linear(512, 1)

    def forward(self, x):
        return F.relu(self.layer(x))

    def training_step(self, batch, batch_nb) -> STEP_OUTPUT:
        x, y = batch
        return F.mse_loss(self(x), y)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return AdamW(self.parameters())

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(TrainDataset(), num_workers=10, persistent_workers=True)


class TrainDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        data = get_tokenized_data("bert-base-cased")
        self.x = data[0]
        self.y = data[1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
