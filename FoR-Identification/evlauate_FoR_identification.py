import argparse

import pandas as pd
import json
import os
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def find_answer(pred):
    def find_potential_answer(pred):
        for potential_label in ["internal intrinsic", "internal relative", "external relative", "external intrinsic"]:
            if potential_label in pred:
                return potential_label
        return pred

    pred = pred.lower()
    find_answer = pred.find("answer:")

    if find_answer == -1:
        find_category = pred.find("category:")
        if find_category != -1:
            pred_label = pred[find_category + len("category:"):].strip()
        else:  # Worst case finding potential label in explanation
            pred_label = find_potential_answer(pred)
    else:
        pred_label = pred[find_answer + len("answer:"):].strip()
        find_first_dot = pred[find_answer + len("answer:"):].find(".")
        if find_first_dot != -1:
            pred_label = pred_label[:find_first_dot]
        if pred_label not in ["internal intrinsic", "internal relative", "external relative", "external intrinsic"]:
            pred_label = find_potential_answer(pred_label)
    return pred_label


def get_result_ambiguous(file_path):
    data = pd.read_csv(os.path.join("../LLM_results", file_path))

    set_label = ['external intrinsic external relative', 'external relative internal relative',
                 'external intrinsic external relative internal intrinsic internal relative', 'external relative']
    simple_label = ["external relative", "external intrinsic", "internal intrinsic", "internal relative"]
    count_case = {label: {answer_label: 0 for answer_label in simple_label} for label in set_label}
    # print(count_case)
    acc = 0
    total = 0
    for row, result in data.iterrows():
        label = " ".join(sorted(eval(result["label"]))).lower()
        pred = find_answer(result["GPT_predict"]).lower()
        total += 1
        if pred not in count_case[label]:
            print(pred)
            continue
        count_case[label][pred] += 1
        acc += 1 if pred in label else 0

    result = {label: [] for label in set_label}
    for label in set_label:
        total = sum(count_case[label].values())
        for answer_label in simple_label:
            result[label].append("{:.2f}".format(count_case[label][answer_label] * 100 / total))

    return result, acc / total


def get_result_clear(file_path):
    data = pd.read_csv(os.path.join("../LLM_results", file_path))

    set_label = ['external intrinsic external relative', 'external relative internal relative',
                 'external intrinsic external relative internal intrinsic internal relative', 'external relative']
    simple_label = ["external relative", "external intrinsic", "internal intrinsic", "internal relative"]
    count_case = {label: {answer_label: 0 for answer_label in simple_label} for label in set_label}
    # print(count_case)
    acc = 0
    total = 0
    for row, result in data.iterrows():
        label = " ".join(sorted(eval(result["label"]))).lower()
        pred = find_answer(result["GPT_predict"]).lower()
        total += 1
        if pred not in count_case[label]:
            print(pred)
            continue
        count_case[label][pred] += 1
        acc += 1 if pred in label else 0

    result = {label: [] for label in set_label}
    for label in set_label:
        total = sum(count_case[label].values())
        for answer_label in simple_label:
            result[label].append(count_case[label][answer_label] * 100 / total)

    return result, acc


def main(args):
    CANDIDATE_ANSWER = ['external relative', 'external intrinsic', 'internal intrinsic', 'internal relative']
    if args.clear:
        DEFAULT_LABEL_SET = ['external relative', 'external intrinsic', 'internal intrinsic', 'internal relative']
    else:
        # Label for Cow, Car, Box, Pen, respectively
        DEFAULT_LABEL_SET = ['external intrinsic external relative',
                             'external intrinsic external relative internal intrinsic internal relative',
                             'external relative internal relative',
                             'external relative']

    result, acc = get_result_clear(args.result_file)

    print("Percentage of response per FoR class")
    print("Accuracy: {:.2f}%".format(acc))
    for label in DEFAULT_LABEL_SET:
        print("Class {}: ".format(label))
        print("| ")
        for answer_idx in range(len(CANDIDATE_ANSWER)):
            print(f"\t{result[label][answer_idx]}", end="\t|")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--clear', type=bool, default=False,
                        help='Whether dataset is from C-split (clear) or A-split (ambiguous)')

    parser.add_argument('--result_file', type=str, default='result_llama.csv',
                        help='Result file (result file must be in csv format)')

    parser.add_argument('--specific_label', type=str, default='',
                        help='Whether to see results from specific label. Can be list "A,B" -> [A, B]')

    args = parser.parse_args()
    main(args)
