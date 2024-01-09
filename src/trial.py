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

    model = BertMultitaskPL(cfg)
    # model = BertMultitaskPL.load_from_checkpoint(cfg.checkpoint)

    predictions = trainer.predict(model, trial_loader)
    emotions = torch.cat(
        [pred[0] for pred in predictions], dim=0
    )  # shape: [dataset_length]
    causes = torch.cat(
        [pred[1] for pred in predictions], dim=0
    )  # shape: [dataset_length, 512]
    con_IDs = torch.cat(
        [pred[2] for pred in predictions], dim=0
    )  # shape: [dataset_length]
    utt_IDs = torch.cat(
        [pred[3] for pred in predictions], dim=0
    )  # shape: [dataset_length]
    word_ids = torch.cat(
        [pred[4] for pred in predictions], dim=0
    )  # shape: [dataset_length, 512]

    container = []
    for i in range(len(trial_set)):
        container.append(
            {
                "conversation_ID": con_IDs[i].item(),
                "utterance_ID": utt_IDs[i].item(),
                "word_ids": word_ids[i].tolist()[0],
                "emotion": emotions[i].item(),
                "tagged": causes[i].tolist(),
            }
        )

    untokenized = untokenize(container, cfg.data_path)

    with open(cfg.out_file, "w") as file:
        json.dump(untokenized, file, indent=4)


if __name__ == "__main__":
    main()
