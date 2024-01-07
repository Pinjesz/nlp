import torch
from torch.utils.data import DataLoader
from modeling.models import BertMultitask
import pytorch_lightning as pl
from transformers import BertModel, BertConfig
from torch.nn import CrossEntropyLoss


class BertMultitaskPL(pl.LightningModule):
    def __init__(self, model_name="bert-base-cased"):
        super().__init__()
        config = BertConfig.from_pretrained(model_name)
        self.bert_model = BertMultitask(config)
        self.emotion_loss = CrossEntropyLoss()
        self.causes_loss = CrossEntropyLoss()

    def forward(self, **tokens):
        return self.bert_model(**tokens)

    def training_step(self, batch, batch_idx):
        tokens, labels = batch
        logits_emotions, logits_causes = self.self(**tokens)
        emotion_loss = self.emotion_loss(logits_emotions, labels["emotion"])
        causes_loss = self.causes_loss(
            logits_causes.view(-1, 3), labels["tagged"].view(-1)
        )
        loss = emotion_loss + causes_loss
        self.log_dict({"emotion_loss": emotion_loss, "causes_loss": causes_loss})
        self.log("train_loss", loss)  # Logging the loss
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)
