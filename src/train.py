# from model import RandomModel
# import lightning as L
# from lightning.pytorch.loggers import WandbLogger


from datetime import datetime

import torch
from data.datasets import TrainDataset
from torch.utils.data import DataLoader

from modeling.trainers import BertMultitaskPL
import pytorch_lightning as pl
from torch.utils.data import Subset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="cfgs", config_name="config.yaml")
def main(cfg: DictConfig):
    today_date = datetime.today().strftime(r"%d-%m-%Y")
    project_name = f"nlp-{today_date}"
    wandb_logger = WandbLogger(project=project_name, entity="phondra")
    wandb_logger.experiment.config.update({**cfg})

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        # dirpath="checkpoints/",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=cfg.early_stopping.min_delta,
        patience=cfg.early_stopping.patience,
        verbose=True,
        mode="min",
    )

    base_ds = TrainDataset(cfg.tokenize_data_path)

    val_size = int(cfg.val_size * len(base_ds))
    test_size = int(cfg.test_size * len(base_ds))
    train_size = len(base_ds) - (test_size + val_size)

    indices = list(range(len(base_ds)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_set = Subset(base_ds, train_indices)
    val_set = Subset(base_ds, val_indices)
    test_set = Subset(base_ds, test_indices)

    model = BertMultitaskPL(cfg)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=cfg.num_epochs,
        logger=wandb_logger,
        devices=1,
        accelerator=accelerator,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=False,
        precision="bf16",
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.test(
        dataloaders=test_loader,
        model=model,
        ckpt_path=checkpoint_callback.best_model_path,
        verbose=True,
    )


if __name__ == "__main__":
    main()
