import json
import torch
from transformers import BertTokenizerFast
from tokenize_data import *


def untokenize(predicted:list):
    """
    `predicted = [{"conversation_ID" : <int>, "utterance_ID" : <int>, "word_ids":<list[int, None]>, "emotion": <int>, "tagged": <pytorch.tensor>}, ...]`
    """
    pairs:dict[int, list[list[str]]] = {}

    for p in predicted:
        if pairs.get(p["conversation_ID"]) is None:
            pairs[p["conversation_ID"]] = []
        spans = []
        start = None
        for i, tag in enumerate(p["tagged"][0]):
            if tag.item() == category_to_index["B-cause"] and start is None:
                start = i
            if start is not None and tag.item() != category_to_index["B-cause"] and tag.item() != category_to_index["I-cause"]:
                spans.append([start, i])
                start = None

        idx_to_utterance = [-1]
        utt = -1
        for i in range(1, len(p["word_ids"])):
            if p["word_ids"][i - 1] is None and p["word_ids"][i] is not None:
                utt += 1
            idx_to_utterance.append(utt)

        for start, end in spans:
            if idx_to_utterance[start] != idx_to_utterance[end]:
                continue
            pairs[p["conversation_ID"]].append([f'{p["utterance_ID"]}_{index_to_emotion[p["emotion"]]}', f"{idx_to_utterance[start]}_{start}_{end}"])

    conversation_ids = sorted(pairs.keys())

    data_path = "data/raw/Subtask_1_train.json"
    with open(data_path, "r") as file:
        data = json.load(file)

        result = []
        for conv_id in conversation_ids:
            con_pairs = pairs[conv_id]
            data_con = data[conv_id - 1]
            result.append({
                "conversation_ID": conv_id,
                "conversation":data_con["conversation"],
                "emotion-cause_pairs":con_pairs
            })

    return result


if __name__ == "__main__":
    tokenizer_checkpoint = "bert-base-cased"
    untokenized_data_path = "data/tokenized/out.json"

    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(
        tokenizer_checkpoint
    )
    text = "Weathered mountains are highly dangerous . [SEP] I like trains very much , they are awesome ."
    tokens = tokenizer(text)
    word_ids = tokens.word_ids()
    for j, en in enumerate(tokens["input_ids"]):
        if en == tokenizer.sep_token_id:
            word_ids[j] = None
    pred = [{"conversation_ID": 1,
             "utterance_ID": 2,
             "word_ids" : word_ids,
             "emotion": 1,
             "tagged": torch.tensor([[-100, 0, 0, 1, 1, 1, 2, 2, -100, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, -100]])
             }]

    data = untokenize(pred)
    with open(untokenized_data_path, "w") as file:
        json.dump(data, file)