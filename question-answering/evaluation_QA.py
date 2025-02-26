import argparse

import pandas as pd
import json
import os
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def find_answer(pred):
    def find_potential_answer(pred_answer):
        for potential_label in ["left", "right", "front", "back"]:
            if potential_label in pred_answer:
                return potential_label
        return pred_answer

    pred = pred.lower()
    find_real_answer = pred.find("answer:")

    if find_real_answer == -1:
        find_category = pred.find("category:")
        if find_category != -1:
            pred_label = pred[find_category + len("category:"):].strip()
        else:  # Worst case finding potential label in explanation
            pred_label = find_potential_answer(pred)
    else:
        pred_label = pred[find_real_answer + len("answer:"):].strip()
        find_first_dot = pred[find_real_answer + len("answer:"):].find(".")
        if find_first_dot != -1:
            pred_label = pred_label[:find_first_dot]
        if pred_label not in ["left", "right", "front", "back"]:
            pred_label = find_potential_answer(pred_label)
    return pred_label


def get_result_ambiguous(file_path, default_relation_index=1, specific_category=()):
    data = pd.read_csv(os.path.join("../LLM_results_QA", file_path))

    # Only use for FoR classes
    with open(os.path.join("../Dataset", "A-split_QA_camera_total.json"), 'r') as file:
        normal_data_label = json.load(file)["data"]

    FoR_classes = ['external intrinsic external relative',
                   'external intrinsic external relative internal intrinsic internal relative',
                   'external relative internal relative', 'external relative']
    simple_label = ["left", "right", "front", "back"]
    acc_case = {FoR_label: [0, 0] for FoR_label in FoR_classes}
    same_answer = {FoR_label: 0.0 for FoR_label in FoR_classes}
    total_case = {FoR_label: 0 for FoR_label in FoR_classes}
    convert_relation = {"behind": "back"}
    acc = 0
    total = 0

    map_context_label = dict()

    for context_data in normal_data_label:
        context = context_data["context"]
        FoR_label = " ".join(sorted(context_data["label"])).lower()
        map_context_label[context] = FoR_label

    for row, result in data.iterrows():
        FoR_label = map_context_label[result["context"]]
        label = " ".join((eval(result["label"]))).lower().split()
        pred = find_answer(result["GPT_predict"]).lower().replace(".", "")

        if specific_category and FoR_label not in specific_category:
            continue

        total += 1
        total_case[FoR_label] += 1

        if pred in convert_relation:
            pred = convert_relation[pred]
        if pred not in simple_label:
            continue

        if len(label) > 1:
            # acc_case[default_relation_index] - relative
            # acc_case[1 - default_relation_index] - intrinsic
            if label[default_relation_index] == label[1 - default_relation_index]:
                same_answer[FoR_label] += 1

            if pred == label[default_relation_index]:
                acc_case[FoR_label][default_relation_index] += 1
            elif pred == label[1 - default_relation_index]:
                acc_case[FoR_label][1 - default_relation_index] += 1
            # print(pred, label, acc_case)
        else:
            acc_case[FoR_label][0] += 1 if pred in label else 0
        acc += 1 if pred in label else 0

    # print(file_path, acc / total)
    result = {label: [0, 0] for label in FoR_classes}
    for FoR_label in FoR_classes:
        if total_case[FoR_label]:
            for i in range(2):
                result[FoR_label][i] = 100 * acc_case[FoR_label][i] / total_case[FoR_label]
            same_answer[FoR_label] = same_answer[FoR_label] / total_case[FoR_label]

    return result, acc * 100 / total, same_answer


def get_result_clear(file_path, specific_category=None):
    data = pd.read_csv(os.path.join("../LLM_results_QA", file_path))

    # Only use for FoR classes
    with open(os.path.join("../Dataset", "C-split_QA_camera_total.json"), 'r') as file:
        questions_info = json.load(file)["data"]

    set_label = ['external relative', 'external intrinsic', 'internal intrinsic', 'internal relative']
    simple_label = ["left", "right", "front", "back"]
    convert_relation = {"behind": "back"}
    acc_case = {FoR_label: 0 for FoR_label in set_label}
    total_case = {FoR_label: 0 for FoR_label in set_label}
    total = 0
    acc = 0
    pred_list = []
    label_list = []
    map_context_label = dict()

    for context_data in questions_info:
        context = context_data["context"]
        map_context_label[context] = str(" ".join(sorted(context_data["label"])).lower())

    for row, result in data.iterrows():
        for_label = map_context_label[result["context"]]
        label = " ".join(sorted(eval(result["label"]))).lower().split()[0]
        pred = find_answer(result["GPT_predict"]).lower().replace(".", "")
        total += 1
        total_case[for_label] += 1
        if pred in convert_relation:
            pred = convert_relation[pred]
        if pred not in simple_label:
            continue

        if specific_category and for_label not in specific_category:
            continue
        pred_list.append(pred)
        label_list.append(label)
        acc_case[for_label] += 1 if pred == label else 0
        acc += 1 if pred == label else 0

    result = {label: 0.0 for label in set_label}
    for for_label in set_label:
        if total_case[for_label]:
            result[for_label] = acc_case[for_label] * 100 / total_case[for_label]

    return result, acc * 100 / total, pred_list, label_list


def main(args):
    if args.clear:
        DEFAULT_LABEL_SET = ['external relative', 'external intrinsic', 'internal intrinsic', 'internal relative']

        set_label = DEFAULT_LABEL_SET
        if args.specific_label != "":
            set_label = args.specific_label.split(",")

        result, acc, pred_list, label_list = get_result_clear(args.result_file, specific_category=set_label)

        for FoR_case in set_label:
            print("{}: {}".format(FoR_case, result[FoR_case]))
        print("Avg acc: {}".format(acc))

        if args.show_confusion_matrix is True:
            plt.grid(False)
            matrix = confusion_matrix(pred_list, label_list, normalize="true",
                                      labels=["left", "right", "front", "back"])
            disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["left", "right", "front", "back"])
            fig, ax = plt.subplots(figsize=(5, 5))
            disp.plot(ax=ax, cmap=plt.cm.Blues)
            plt.show()
    else:
        # Label for Cow, Car, Box, Pen, respectively
        CASES_NAMES = ["cow", "car", "box", "pen"]
        DEFAULT_LABEL_SET = ['external intrinsic external relative',
                             'external intrinsic external relative internal intrinsic internal relative',
                             'external relative internal relative',
                             'external relative']
        MAP_CASE_NAME_LABEL_SET = {CASES_NAMES[i]: DEFAULT_LABEL_SET[i] for i in range(4)}
        POSSILBLE_ANSWER = [2, 2, 1, 1]

        specific_classes = []
        if args.specific_label != "":
            specific_classes = args.specific_label.split(",")
            for class_name in specific_classes:
                if class_name.lower() not in MAP_CASE_NAME_LABEL_SET:
                    print("Invalid Class Name:", class_name)
                    return
            specific_classes = [MAP_CASE_NAME_LABEL_SET[class_name.lower()] for class_name in specific_classes]

        result = get_result_ambiguous(args.result_file, default_relation_index=0, specific_category=specific_classes)

        for label_index, FoR_case in enumerate(DEFAULT_LABEL_SET):
            avg = 0
            print("-" * 40)
            print("{} case".format(CASES_NAMES[label_index]))

            for i in range(POSSILBLE_ANSWER[label_index]):
                if POSSILBLE_ANSWER[label_index] >= 2:

                    if i == 0:
                        # Subtract duplicate cases
                        percentage = 100 * ((result[0][FoR_case][i] - result[2][FoR_case]) / (
                                result[0][FoR_case][0] + result[0][FoR_case][1] - - result[2][FoR_case]))
                        print("External Relative (%): {:.2f}%".format(percentage))
                    else:
                        percentage = 100 * (result[0][FoR_case][i] / (
                                result[0][FoR_case][0] + result[0][FoR_case][1] - result[2][FoR_case]))
                        print("External Intrinsic (%): {:.2f}%".format(percentage))

                # All cases avg
                # (Same as accuracy from all cases,
                # this is not calculate separately only the result from function is separate)
                avg += result[0][FoR_case][i]
            print("Accuracy: {:.2f}".format(avg))

        print("-" * 40)
        print("Overall Accuracy ${:.2f}$ ".format(result[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--clear', type=bool, default=True,
                        help='Whether dataset is from C-split (clear) or A-split (ambiguous)')

    parser.add_argument('--result_file', type=str, default='Example_QA_clear.csv',
                        help='Result file (result file must be in csv format)')

    parser.add_argument('--specific_label', type=str, default='',
                        help='Whether to see results from specific label. Can be list "A,B" -> [A, B]')

    parser.add_argument('--show_confusion_matrix', type=bool, default=False,
                        help='Whether to print confusion matrix of response')

    args = parser.parse_args()
    main(args)
