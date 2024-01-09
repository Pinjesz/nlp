# -*- coding:utf-8 -*-
import sys, os, json, copy, string
import numpy as np

emotion_idx = dict(zip(['neutral','anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'], range(7)))


def get_json_data(json_file):
    with open(json_file, 'r') as fi:
        json_data = json.load(fi)
    return json_data


def convert_list_to_dict(data_list, main_key=''):
    new_dict = {}
    for x in data_list:
        if 'ID' in main_key:
            key = int(x[main_key])
        else:
            key = x[main_key]
        if key not in new_dict:
            new_dict[key] = x
        else:
            sys.exit('Instance repeat!')
    return new_dict


# Participants need to provide the position indexes of the cause span in the prediction file!!!
# You can use this function to obtain the position indexes of golden annotations.
def get_span_position(span, utterance):
    begin_id, end_id = 0, 0
    cause_token = span.split()
    utterance_token = utterance.split()
    for wi in range(len(utterance_token)):
        if (wi+len(cause_token))<=len(utterance_token) and utterance_token[wi:wi+len(cause_token)] == cause_token:
            begin_id = wi
            end_id = wi+len(cause_token)
            break
    return [begin_id, end_id] # start from 0, [begin_id, end_id)


'''
Strict Match: emotion_utt and cause_utt are the same, and the cause spans completely match.
Fuzzy Match: emotion_utt and cause_utt are the same, and the cause spans overlap
'''
def judge_cause_span_pair_emocate(pred_span_pair, true_spans_pos_dict, span_mode='fuzzy'): # strict/fuzzy
    d_id, emo_id, cau_id, start_cur, end_cur, emo = pred_span_pair
    cur_key = 'dia{}_emoutt{}_causeutt{}'.format(d_id, emo_id, cau_id)
    if cur_key in true_spans_pos_dict:
        if span_mode == 'strict':
            if [start_cur, end_cur, emo] in true_spans_pos_dict[cur_key]:
                true_spans_pos_dict[cur_key].remove([start_cur, end_cur, emo])
                return True
        else:
            for t_start, t_end, emo_y in true_spans_pos_dict[cur_key]:
                if emo == emo_y and not(end_cur<=t_start or start_cur>=t_end):
                    true_spans_pos_dict[cur_key].remove([t_start, t_end, emo_y])
                    return True
    return False


def cal_prf_span_pair_emocate(span_pair_dict, pred_pairs, span_mode='strict'):
    conf_mat = np.zeros([7,7])
    for p in pred_pairs: # [conv_id, emo_utt_id, cau_utt_id, span_start_id, span_end_id, emotion_category]
        if judge_cause_span_pair_emocate(p, span_pair_dict, span_mode=span_mode):
            conf_mat[p[5]][p[5]] += 1
        else:
            conf_mat[0][p[5]] += 1
    for k, v in span_pair_dict.items():
        for p in v:
            conf_mat[p[2]][0] += 1
    p = np.diagonal(conf_mat / np.reshape(np.sum(conf_mat, axis = 0)+(1e-8), [1,7]))
    r = np.diagonal(conf_mat / np.reshape(np.sum(conf_mat, axis = 1)+(1e-8), [7,1]))
    f = 2*p*r/(p+r+(1e-8))
    weight0 = np.sum(conf_mat, axis = 1)
    weight = weight0[1:] / np.sum(weight0[1:])
    w_avg_p = np.sum(p[1:] * weight)
    w_avg_r = np.sum(r[1:] * weight)
    w_avg_f1 = np.sum(f[1:] * weight)

    micro_acc = np.sum(np.diagonal(conf_mat)[1:])
    micro_p = micro_acc / (sum(np.sum(conf_mat, axis = 0)[1:])+(1e-8))
    micro_r = micro_acc / (sum(np.sum(conf_mat, axis = 1)[1:])+(1e-8))
    micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r+1e-8)

    return [w_avg_p, w_avg_r, w_avg_f1, micro_p, micro_r, micro_f1]


def get_match_scores(pred_span, true_spans):
    match_id, match_gold_length, match_length, match_score = 0, 0, 0, 0
    p_start, p_end, p_emo = pred_span
    for ii, (t_start, t_end, t_emo) in enumerate(true_spans):
        if p_emo == t_emo and not (p_end<=t_start or p_start>=t_end):
            cur_match_length = min(p_end, t_end) - max(p_start, t_start)
            cur_gold_length = t_end - t_start
            cur_match_score = cur_match_length / float(cur_gold_length)
            if cur_match_score > match_score:
                match_id = ii
                match_gold_length = cur_gold_length
                match_length = cur_match_length
                match_score = cur_match_score
            if (cur_match_score == match_score) and (cur_match_score > 0):
                if cur_match_length > match_length:
                    match_id = ii
                    match_gold_length = cur_gold_length
                    match_length = cur_match_length
                    match_score = cur_match_score
    return match_id, match_gold_length, match_length, match_score


'''
Proportional Match (span): Each predicted span is compared with all golden spans, and determine which golden span it matches based on the overlap ratio (match score). Then the Precision, Recall, and F1 are calculated based on the overlapping tokens.
'''
def cal_prf_span_pair_emocate_proportional(true_span_pair_dict, pred_span_pair_dict): # 'dia{}_emoutt{}_causeutt{}': [[span_start_id, span_end_id, emotion_category], ...]
    prf_mat = np.zeros([7,5]) # row: emotion category; col: correct_num, true_num, pred_num, matched_true_span_num, true_span_num
    true_span_pair_dict_copy = copy.deepcopy(true_span_pair_dict)
    for k, v in pred_span_pair_dict.items():
        for pred_span in v:
            prf_mat[pred_span[2]][2] += pred_span[1] - pred_span[0]
            if k in true_span_pair_dict:
                true_spans = true_span_pair_dict[k]
                match_id, match_gold_length, match_length, match_score = get_match_scores(pred_span, true_spans)
                if match_length > 0:
                    prf_mat[pred_span[2]][0] += match_length
                    prf_mat[pred_span[2]][1] += match_gold_length # Multiple predicted spans may match the same golden span.
                    prf_mat[pred_span[2]][3] += 1
                    if true_spans[match_id] in true_span_pair_dict_copy[k]:
                        true_span_pair_dict_copy[k].remove(true_spans[match_id])

    for k, v in true_span_pair_dict_copy.items():
        for true_span in v:
            prf_mat[true_span[2]][1] += true_span[1] - true_span[0]
            prf_mat[true_span[2]][3] += 1
    for k, v in true_span_pair_dict.items():
        for true_span in v:
            prf_mat[true_span[2]][4] += 1

    p_scores = prf_mat[1:,0] / (prf_mat[1:,2]+(1e-8))
    r_scores = prf_mat[1:,0] / (prf_mat[1:,1]+(1e-8))
    f1_scores = 2*p_scores*r_scores/(p_scores+r_scores+(1e-8))
    weight = prf_mat[1:,4] / sum(prf_mat[1:,4]) # Calculate the weight based on the actual number of golden spans.
    w_avg_p = sum(p_scores*weight)
    w_avg_r = sum(r_scores*weight)
    w_avg_f1 = sum(f1_scores*weight)

    total_correct = sum(prf_mat[1:,0])
    micro_p = total_correct / (sum(prf_mat[1:,2])+(1e-8))
    micro_r = total_correct / (sum(prf_mat[1:,1])+(1e-8))
    micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r+1e-8)

    return [w_avg_p, w_avg_r, w_avg_f1, micro_p, micro_r, micro_f1]


# Remove the punctuation token at the beginning and end of the cause span
def clean_span(span):
    while 1:
        span = span.strip()
        if span[0] not in string.punctuation and span[-1] not in string.punctuation:
            break
        else:
            if span[0] in string.punctuation:
                span = span[1:]
            if span[-1] in string.punctuation:
                span = span[:-1]
    return span


def has_letter(text):
    for c in text:
        if c.isalpha():
            return True
    return False


def evaluate_1_2(pred_data, gold_data):
    gold_data_dict = convert_list_to_dict(gold_data, main_key="conversation_ID")
    pred_data_dict = convert_list_to_dict(pred_data, main_key="conversation_ID")

    pred_pairs, true_pairs = [], []
    conv_context_dict = {}
    for id, ins in gold_data_dict.items(): # The public evaluation data may contain some interference data that is not used for evaluation.
        if id not in pred_data_dict:
            sys.exit('Conversation {} are missing!'.format(id))
        else:
            pred = pred_data_dict[id]
            all_utterances = [x["text"] for x in ins["conversation"]]
            conv_context_dict[id] = all_utterances

            def get_new_pair_list(span_pair_list, pred=False):
                new_span_pair_list  = []
                for x in span_pair_list:
                    if not isinstance(x, list):
                        sys.exit('emotion-cause_pairs format error!')
                    else:
                        if len(x) != 2:
                            sys.exit('emotion-cause_pairs format error!')
                        else:
                            emo_id, emotion = x[0].split('_')
                            if emotion not in emotion_idx:
                                sys.exit('Unknown emotion category!')
                            else:
                                if 'U' in emo_id:
                                    emo_id = emo_id.replace('U','')
                                if pred:
                                    if has_letter(x[1]):
                                        sys.exit('emotion-cause_pairs format error! You should provide the position index range of the cause span, not the text itself.')
                                    else:
                                        cause_info = x[1].split('_')
                                        if len(cause_info) != 3:
                                            sys.exit('emotion-cause_pairs format error!')
                                        else:
                                            cause_id, span_start_id, span_end_id = cause_info
                                            if 'U' in cause_id:
                                                cause_id = cause_id.replace('U','')
                                            span_idx_list = [int(span_start_id), int(span_end_id)]
                                else:
                                    cause_id, cur_span = x[1].split('_')
                                    if 'U' in cause_id:
                                        cause_id = cause_id.replace('U','')
                                    cur_span = clean_span(cur_span)
                                    span_idx_list = get_span_position(cur_span, all_utterances[int(cause_id)-1])

                                new_pair = [id, int(emo_id), int(cause_id)] + span_idx_list + [emotion_idx[emotion]]
                                if new_pair not in new_span_pair_list:
                                    new_span_pair_list.append(new_pair)
                return new_span_pair_list # [[conv_id, emo_utt_id, cau_utt_id, span_start_id, span_end_id, emotion_category], ...]

            true_pairs.extend(get_new_pair_list(ins["emotion-cause_pairs"])) #

            if "emotion-cause_pairs" not in pred:
                sys.exit("Cannot find the key 'emotion-cause_pairs'!")
            else:
                pred_pairs.extend(get_new_pair_list(pred["emotion-cause_pairs"], pred=True))

    def get_span_pair_dict(pairs):
        span_pair_dict = {}
        for p in pairs:
            cur_key = 'dia{}_emoutt{}_causeutt{}'.format(p[0], p[1], p[2])
            if cur_key in span_pair_dict:
                span_pair_dict[cur_key].append(p[3:])
            else:
                span_pair_dict[cur_key] = [p[3:]]
        return span_pair_dict

    true_span_pair_dict = get_span_pair_dict(true_pairs)
    pred_span_pair_dict = get_span_pair_dict(pred_pairs)

    true_span_pair_dict_copy = copy.deepcopy(true_span_pair_dict)
    score_list = cal_prf_span_pair_emocate(true_span_pair_dict, pred_pairs, span_mode='strict')
    score_list_2 = cal_prf_span_pair_emocate_proportional(true_span_pair_dict_copy, pred_span_pair_dict)
    return score_list, score_list_2


def eval(pred_data, gold_data):
    score_list, score_list_1 = evaluate_1_2(pred_data, gold_data)

    result = {
        "weighted_strict_precision": score_list[0],
        "weighted_strict_recall": score_list[1],
        "weighted_strict_f1": score_list[2],
        "weighted_Proportional_precision": score_list_1[0],
        "weighted_Proportional_recall": score_list_1[1],
        "weighted_Proportional_f1": score_list_1[2],
        "strict_precision": score_list[3],
        "strict_recall": score_list[4],
        "strict_f1": score_list[5],
        "Proportional_precision": score_list_1[3],
        "Proportional_recall": score_list_1[4],
        "Proportional_f1": score_list_1[5],
    }

    return result


if __name__ == "__main__":
    gold_file = "Subtask_1_gold.json"
    pred_file = "Subtask_1_pred.json"
    output_file = "scores.json"
    if len(sys.argv) > 1:
        [_, gold_file, pred_file, output_file] = sys.argv

    participate_subtask_num = 0

    if os.path.exists(pred_file):
        participate_subtask_num += 1
        pred_data = get_json_data(pred_file)
        gold_data = get_json_data(gold_file)
        result = eval(pred_data, gold_data)
        with open(output_file, "w") as file:
            json.dump(result, file, indent=4)

    if participate_subtask_num == 0:
        sys.exit("Could not find valid json file in your zip package!")
