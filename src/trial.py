from dataclasses import dataclass
from datetime import datetime
import json

import torch
from data.datasets import TrialDataset
from torch.utils.data import DataLoader
from data.untokenize import untokenize

from modeling.trainers import BertMultitaskPL
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="cfgs", config_name="trial.yaml")
def main(cfg: DictConfig):
    today_date = datetime.today().strftime(r"%d-%m-%Y")
    project_name = f"nlp-{today_date}"
    # wandb_logger = WandbLogger(project=project_name, entity="jasiekjeschke")
    # wandb_logger.experiment.config.update({**cfg})

    trial_set = TrialDataset(cfg.tokenize_data_path)

    trial_loader = DataLoader(
        trial_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        devices=1,
        # logger=wandb_logger,
        accelerator=accelerator,
        # precision="bf16-mixed",
    )

    # model = BertMultitaskPL(cfg).load_from_checkpoint(cfg.checkpoint)
    model = BertMultitaskPL(cfg)

    predictions = trainer.predict(model, trial_loader)
    emotions = torch.cat(
        [pred[0] for pred in predictions], dim=0
    )  # shape: [dataset_length]
    causes = torch.cat(
        [pred[1] for pred in predictions], dim=0
    )  # shape: [dataset_length, 512]

    # FIXME: emotions i causes na untokanizer
    untokenized = untokenize(predictions)

    with open(cfg.out_file, "w") as file:
        json.dump(untokenized, file)


if __name__ == "__main__":
    main()
