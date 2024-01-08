import numpy as np
import torch
from transformers import BertTokenizerFast
import json
import pickle
from tqdm import tqdm


emotions = ["neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise"]
emotion_to_index = dict(zip(emotions, range(len(emotions))))
index_to_emotion = dict(zip(range(len(emotions)), emotions))

categories = ["B-cause", "I-cause", "O"]
category_to_index = dict(zip(categories, range(len(categories))))
index_to_category = dict(zip(range(len(categories)), categories))


def find_index(sequence, subsequence) -> tuple[int, int]:
    l = len(subsequence)
    for i in range(len(sequence) - l + 1):
        if sequence[i : i + l] == subsequence:
            return i, i + l
    return -1, -1


def get_tokenized_data(tokenizer_checkpoint: str, data="data/raw/Subtask_1_train.json", data_type="train"):
    if type(data) is str:
        with open(data, "r") as file:
            data = json.load(file)

    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(
        tokenizer_checkpoint
    )
    tokenized = []
    labels = []

    for conversation in tqdm(data):
        utterances = conversation["conversation"]
        utt_num = len(utterances)
        for i in range(utt_num):
            previous = " [SEP] ".join([ut["text"] for ut in utterances[:i]])
            current = utterances[i]["text"]
            encoded = tokenizer(
                previous,
                current,
                padding="max_length",
                add_special_tokens=True,
            )
            if len(encoded["input_ids"]) > tokenizer.model_max_length:
                continue

            word_ids_temp = encoded.word_ids()
            encoded_list = encoded["input_ids"]

            word_index = -1
            word_ids = []
            for j, en in enumerate(encoded_list):
                if en == tokenizer.sep_token_id or word_ids_temp[j] is None:
                    word_index = -1
                    word_ids.append(-10)
                else:
                    if word_ids_temp[j] != word_ids_temp[j - 1]:
                        word_index += 1
                    word_ids.append(word_index)

            tokenized.append(
                {
                    "input_ids": torch.tensor([encoded["input_ids"]]),
                    "token_type_ids": torch.tensor([encoded["token_type_ids"]]),
                    "attention_mask": torch.tensor([encoded["attention_mask"]]),
                    "conversation_ID": conversation["conversation_ID"],
                    "utterance_ID": i + 1,
                    "word_ids": word_ids,
                }
            )

            if data_type == "train":
                tagged = ((np.array(word_ids) != -10).astype(int) - 1) * (
                    100 + category_to_index["O"]
                )
                tagged += category_to_index["O"]

                emotion_idx = emotion_to_index[utterances[i]["emotion"]]

                cause_spans = []
                for pair in conversation["emotion-cause_pairs"]:
                    target_index = int(pair[0][0 : pair[0].find("_")])
                    if i + 1 == target_index:
                        underscore_index = pair[1].find("_")
                        source_index = int(pair[1][0:underscore_index])
                        if target_index < source_index:
                            continue
                        span = pair[1][underscore_index + 1 :]
                        tokenized_span = tokenizer(span, add_special_tokens=False)[
                            "input_ids"
                        ]
                        start, end = find_index(encoded_list, tokenized_span)
                        if start == -1:
                            print("Check your code or dataset!")
                            continue
                        cause_spans.append([start, end])

                for start, end in cause_spans:
                    tagged[start + 1 : end] = category_to_index["I-cause"]
                    id = word_ids[start]
                    while word_ids[start] == id:
                        tagged[start] = category_to_index["B-cause"]
                        start += 1

                labels.append(
                    {
                        "emotion": emotion_idx,
                        "tagged": torch.tensor(
                            tagged.reshape([1, tokenizer.model_max_length])
                        ),
                    }
                )
    if data_type == "train":
        return tokenized, labels
    return tokenized


if __name__ == "__main__":
    tokenizer = "bert-base-uncased"
    train_data_path = "data/tokenized/train.pkl"
    trial_data_path = "data/tokenized/trial.pkl"

    tokens, labels = get_tokenized_data(tokenizer)
    print("Length of train data:", len(tokens), len(labels))
    with open(train_data_path, "wb") as file:
        pickle.dump({"tokens": tokens, "labels": labels}, file)

    trial = get_tokenized_data(tokenizer, data="data/raw/Subtask_1_trial.json", data_type="trial")
    print("Length of trial data:", len(trial))
    with open(trial_data_path, "wb") as file:
        pickle.dump({"tokens": trial}, file)
