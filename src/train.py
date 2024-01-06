# from model import RandomModel
# import lightning as L
# from lightning.pytorch.loggers import WandbLogger


import pickle

from models.model import TrainDataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
)
import torch


def main():
    # logger = WandbLogger(project='NLP')
    # model = RandomModel()

    # trainer = L.Trainer(max_epochs=100, logger=False)
    # trainer.fit(model)
    train_ds = TrainDataset()

    print("x")

    # # Load pre-trained BERT model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # model = BertForSequenceClassification.from_pretrained(
    #     model_name, num_labels=6
    # )  # num_emotions would be the number of emotions in your task

    # # Example conversation

    model = BertModel.from_pretrained(model_name)
    conversation = [
        "I'm really excited about the upcoming event!",
        "I'm feeling a bit anxious about the presentation.",
        "That sounds great! I can't wait to attend.",
        "I'm not sure if I'm ready for it.",
    ]

    # Tokenize and process the conversation
    encoded_inputs = tokenizer(
        # " [SEP] ".join(conversation[:-1]),
        # conversation[-1],
        conversation,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    decoded = tokenizer.decode(encoded_inputs["input_ids"].flatten())

    with torch.no_grad():
        outputs = model(**encoded_inputs)

    # # Pass the tokenized inputs through the model
    # with torch.no_grad():
    #     outputs = model(**encoded_inputs)

    # # Extract predicted emotions for each utterance
    # predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()

    # # Map predicted labels to emotions
    # emotion_labels = ["Emotion1", "Emotion2", ...]  # Replace with actual emotion labels
    # extracted_emotions = [emotion_labels[label] for label in predicted_labels]

    # print("Extracted Emotions:", extracted_emotions)

    # Example text
    text = "I felt sad because I lost my job."

    # Tokenize the text
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    inputs = tokenizer.encode_plus(text, return_tensors="pt", add_special_tokens=True)

    # Load pre-trained BERT model for token classification
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model.eval()

    # Get predictions for token-level binary sequence labeling
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

    # Map token-level predictions to text
    predicted_labels = [predictions[0][i].item() for i in range(len(tokens))]
    predicted_spans = []
    current_span = []
    for i, (token, label) in enumerate(zip(tokens, predicted_labels)):
        if label == 1:  # Assuming 1 represents the label for emotion cause
            current_span.append(token)
        elif current_span:
            predicted_spans.append(" ".join(current_span))
            current_span = []

    if current_span:  # In case the span ends at the last token
        predicted_spans.append(" ".join(current_span))

    print("Predicted emotion cause spans:", predicted_spans)


if __name__ == "__main__":
    main()
