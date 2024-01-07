from typing import Optional

from transformers import (
    BertModel,
    BertPreTrainedModel,
)
import torch
from torch import nn


class BertMultitask(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.sequence_num_labels = 7  # tyle ile emocji
        self.causes_num_labels = 3
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier_emotions = nn.Linear(
            config.hidden_size, self.sequence_num_labels
        )
        self.classifier_causes = nn.Linear(config.hidden_size, self.causes_num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids.squeeze(1),
            attention_mask=attention_mask.squeeze(1),
            token_type_ids=token_type_ids.squeeze(1),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits_emotions = self.classifier_emotions(pooled_output)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits_causes = self.classifier_causes(sequence_output)

        return logits_emotions, logits_causes
