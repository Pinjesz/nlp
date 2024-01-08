import torch
from torch.utils.data import DataLoader
from modeling.models import BertMultitask
import pytorch_lightning as pl
from transformers import BertModel, BertConfig
from torch.nn import CrossEntropyLoss
from sklearn.metrics import balanced_accuracy_score


class BertMultitaskPL(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.learning_rate = cfg.lr
        self.bert_model = BertMultitask(cfg)

        if cfg.training_strategy == "heads":
            for param in self.bert_model.bert.parameters():
                param.requires_grad = False

        self.emotion_loss = CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        self.causes_loss = CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    def forward(self, *args, **kwargs):
        return self.bert_model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        tokens, labels = batch

        logits_emotions, logits_causes = self(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            token_type_ids=tokens["token_type_ids"],
        )
        emotion_loss = self.emotion_loss(logits_emotions, labels["emotion"])
        causes_loss = self.causes_loss(
            logits_causes.view(-1, 3), labels["tagged"].view(-1)
        )
        loss = emotion_loss + causes_loss

        self.log_dict(
            {
                "train_emotion_loss": emotion_loss,
                "train_causes_loss": causes_loss,
                "train_loss": loss,
            },
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, labels = batch

        logits_emotions, logits_causes = self(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            token_type_ids=tokens["token_type_ids"],
        )
        emotion_loss = self.emotion_loss(logits_emotions, labels["emotion"])
        causes_loss = self.causes_loss(
            logits_causes.view(-1, 3), labels["tagged"].view(-1)
        )
        loss = emotion_loss + causes_loss

        # Logging validation loss and individual losses
        self.log_dict(
            {
                "val_emotion_loss": emotion_loss,
                "val_causes_loss": causes_loss,
                "val_loss": loss,
            },
            on_epoch=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        tokens, labels = batch
        logits_emotions, logits_causes = self(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            token_type_ids=tokens["token_type_ids"],
        )

        emotions_pred = torch.argmax(logits_emotions, dim=-1)
        causes_pred = torch.argmax(logits_causes, dim=-1)
        emotions_true = (
            labels["emotion"].cpu().detach().numpy()
        )  # Assuming labels dictionary has emotions and causes
        causes_true = labels["tagged"].squeeze(1).cpu().detach().numpy()

        # Assuming emotions_pred and causes_pred are tensors
        emotions_pred = emotions_pred.cpu().detach().numpy()
        causes_pred = causes_pred.cpu().detach().numpy()

        ignore_index = -100

        mask = causes_true != ignore_index
        masked_true_labels = causes_true[mask]
        masked_predicted_labels = causes_pred[mask]

        # Calculate balanced accuracy
        balanced_acc_emotions = balanced_accuracy_score(emotions_true, emotions_pred)
        balanced_acc_causes = balanced_accuracy_score(
            masked_true_labels, masked_predicted_labels
        )

        # Logging balanced accuracies
        self.log_dict(
            {
                "test_balanced_accuracy_emotions": balanced_acc_emotions,
                "test_balanced_accuracy_causes": balanced_acc_causes,
            }
        )

        return {
            "test_balanced_accuracy_emotions": balanced_acc_emotions,
            "test_balanced_accuracy_causes": balanced_acc_causes,
        }

        # emotions_pred_list = emotions_pred.tolist()
        # causes_pred_list = causes_pred.tolist()
        # tokens_list = tokens["input_ids"].tolist()
        # labels_list = labels["emotion"].tolist()  # Assuming 'emotion' labels are used

        # Combine predictions and batches into a dictionary
        # data = {
        #     "predictions": {"emotions": emotions_pred_list, "causes": causes_pred_list},
        #     "batch": {"tokens": tokens_list, "labels": labels_list},
        # }

        # # Save data to a JSON file
        # with open("predictions_and_batches.json", "a") as file:
        #     file.write(json.dumps(data))
        #     file.write("\n")

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
        )
