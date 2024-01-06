import pickle
from torch.utils import data

# import lightning as L
# import numpy as np
# from torch.nn import functional as F, Linear
# from torch.optim import AdamW
# from lightning.pytorch.utilities.types import (
#     STEP_OUTPUT,
#     TRAIN_DATALOADERS,
#     OptimizerLRScheduler,
# )
# import random
# from torch.utils.data import DataLoader, Dataset
# import torch
# from data.tokenize_data import get_tokenized_data


# class RandomModel(L.LightningModule):
#     def __init__(
#         self,
#     ) -> None:
#         super().__init__()
#         self.layer = Linear(512, 1)

#     def forward(self, x):
#         return F.relu(self.layer(x))

#     def training_step(self, batch, batch_nb) -> STEP_OUTPUT:
#         x, y = batch
#         return F.mse_loss(self(x), y)

#     def configure_optimizers(self) -> OptimizerLRScheduler:
#         return AdamW(self.parameters())

#     def train_dataloader(self) -> TRAIN_DATALOADERS:
#         return DataLoader(TrainDataset(), num_workers=10, persistent_workers=True)


class TrainDataset(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        # data = get_tokenized_data("bert-base-cased")
        tokenized_data_path = "data/tokenized/train.pkl"

        with open(tokenized_data_path, "rb") as file:
            data = pickle.load(file)

        self.x = data["tokens"]
        self.y = data["labels"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
