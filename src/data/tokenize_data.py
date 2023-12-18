from transformers import AutoTokenizer
import json
import pickle
from utils.evaluate import get_span_position

def find_index(sequence, subsequence) -> tuple[int, int]:
    l = len(subsequence)
    for i in range(len(sequence) - l + 1):
        if sequence[i : i + l] == subsequence:
            return i, i + l
    return -1, -1


def get_tokenized_data(tokenizer_checkpoint: str):
    data_path = "data/raw/Subtask_1_train.json"
    with open(data_path, "r") as file:
        data = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

    conversations = []
    for conversation in data:
        con = conversation["conversation"][0]["text"]
        for i in range(1, len(conversation["conversation"])):
            con += " [CLS] " + conversation["conversation"][i]["text"]
        conversations.append(con)

    tokens = tokenizer(conversations, padding="max_length")["input_ids"]

    labels = []
    for i, conversation in enumerate(data):
        label = []
        for pair in conversation["emotion-cause_pairs"]:
            underscore_index_0 = pair[0].find('_')
            target_index = int(pair[0][0:underscore_index_0])
            emotion = pair[0][underscore_index_0 + 1:]

            underscore_index_1 = pair[1].find('_')
            source_index = int(pair[1][0:underscore_index_1])
            span = pair[1][underscore_index_1 + 1:]
            tokenized_span = tokenizer(span)["input_ids"][1:-1]
            tokenized_utterance = tokenizer(data[i]["conversation"][source_index - 1]["text"])["input_ids"]
            start, end = get_span_position(tokenized_utterance, tokenized_span)

            if start == -1:
                print("Check your code or dataset!")

            label.append(
                {
                    "target_index": target_index,
                    "emotion": emotion,
                    "source_index": source_index,
                    "span_start": start,
                    "span_end": end,
                }
            )
        labels.append(label)

    filtered_tokens = []
    filtered_labels = []
    for i, t in enumerate(tokens):
        if len(t) <= tokenizer.model_max_length:
            filtered_tokens.append(t)
            filtered_labels.append(labels[i])

    return filtered_tokens, filtered_labels


if __name__ == "__main__":
    tokenizer = "bert-base-cased"
    tokenized_data_path = "data/tokenized/train.pkl"

    tokens, labels = get_tokenized_data(tokenizer)
    with open(tokenized_data_path, 'wb') as file:
        pickle.dump({"tokens": tokens, "labels": labels}, file)
