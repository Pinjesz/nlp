import json
import torch
from transformers import BertTokenizerFast
from .tokenize_data import *


def untokenize(predicted: list, data_path: str):
    """
    `predicted = [{"conversation_ID" : <int>, "utterance_ID" : <int>, "word_ids":<list[int, None]>, "emotion": <int>, "tagged": <pytorch.tensor>}, ...]`
    """
    pairs: dict[int, list[list[str]]] = {}

    for p in predicted:
        if index_to_emotion[p["emotion"]] == "neutral":
            continue
        if pairs.get(p["conversation_ID"]) is None:
            pairs[p["conversation_ID"]] = []

        idx_to_utterance = [-1]
        utt = 0
        for i in range(1, len(p["word_ids"])):
            if p["word_ids"][i - 1] < 0 and p["word_ids"][i] >= 0:
                utt += 1
            idx_to_utterance.append(utt)

        utt_tokens_ids = {i : [] for i in range(idx_to_utterance[-1]+1)}
        utt_cause_ids = {i : [] for i in range(idx_to_utterance[-1]+1)}
        for i, idx in enumerate(idx_to_utterance):
            if idx >= 0 and p["word_ids"][i] >= 0:
                utt_tokens_ids[idx].append(i)
        for i, ids in utt_tokens_ids.items():
            for token_idx in ids:
                if p["tagged"][token_idx] == category_to_index["B-cause"] or p["tagged"][token_idx] == category_to_index["I-cause"]:
                    utt_cause_ids[i].append(token_idx)

        for i, ids in utt_cause_ids.items():
            if len(ids) == 0:
                continue
            token_min = min(ids)
            token_max = max(ids)
            pairs[p["conversation_ID"]].append(
                [
                    f'{p["utterance_ID"]}_{index_to_emotion[p["emotion"]]}',
                    f"{i}_{token_min}_{token_max+1}",
                ]
            )

    conversation_ids = sorted(pairs.keys())

    with open(data_path, "r") as file:
        data = json.load(file)

        data_by_id = {d["conversation_ID"]: d for d in data}

        result = []
        for conv_id in conversation_ids:
            con_pairs = pairs[conv_id]
            data_con = data_by_id[conv_id]
            result.append(
                {
                    "conversation_ID": conv_id,
                    "conversation": data_con["conversation"],
                    "emotion-cause_pairs": con_pairs,
                }
            )

    return result


if __name__ == "__main__":
    data_path = "data/raw/Subtask_1_train.json"
    tokenizer_checkpoint = "bert-base-cased"
    untokenized_data_path = "data/tokenized/out.json"

    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(
        tokenizer_checkpoint
    )
    text = "Weathered mountains are highly dangerous . [SEP] I like trains very much , they are awesome ."
    tokens = tokenizer(text)
    word_ids = tokens.word_ids()
    for j, en in enumerate(tokens["input_ids"]):
        if en == tokenizer.sep_token_id or word_ids[j] is None:
            word_ids[j] = -10
    pred = [
        {
            "conversation_ID": 1,
            "utterance_ID": 2,
            "word_ids": word_ids,
            "emotion": 1,
            "tagged": [-100, 0, 0, 1, 2, 2, 0, 1, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2, 2, -100],
        },
        {
            "conversation_ID": 1,
            "utterance_ID": 1,
            "word_ids": word_ids,
            "emotion": 0,
            "tagged": [-100, 0, 0, 1, 2, 2, 0, 1, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2, 2, -100],
        },
        {
            "conversation_ID": 1,
            "utterance_ID": 1,
            "word_ids": word_ids,
            "emotion": 3,
            "tagged": [-100, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 0, 1, 2, 2, 2, -100],
        }
    ]

    data = untokenize(pred, data_path)
    with open(untokenized_data_path, "w") as file:
        json.dump(data, file)
