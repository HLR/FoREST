import sys
import argparse
import json
import torch
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import math

sys.path.append("./VPEval/src/dino")
sys.path.append("./VPEval/src/dino/groundingdino")
from VPEval.src.dino.vpeval.model.modeling import Model as DinoModel

def calculate_score(args, dino_model, image, text, objects, relation, labels,gt_orientation_dir=()):
    def test_spatial_relation(obj1, obj2, check_direction=False):
        diff_threshold = 1  # Follow VPEval

        rel_relation = relation
        # Changing the relation depend on the direction of obj2
        if check_direction:
            if obj2[3] == "front":
                # Front and back doesn't affect here
                if relation == "left":
                    rel_relation = "right"
                if relation == "right":
                    rel_relation = "left"

            elif obj2[3] == "back":
                # Left and right doesn't affect here
                if relation == "front":
                    rel_relation = "back"
                if relation == "back":
                    rel_relation = "front"

            elif obj2[3] == "right":
                if relation == "front":
                    rel_relation = "right"
                if relation == "right":
                    rel_relation = "front"
                if relation == "back":
                    rel_relation = "left"
                if relation == "left":
                    rel_relation = "back"

            elif obj2[3] == "left":
                if relation == "front":
                    rel_relation = "left"
                if relation == "right":
                    rel_relation = "back"
                if relation == "back":
                    rel_relation = "right"
                if relation == "left":
                    rel_relation = "front"

        # Depth compare (lower is on the back)
        if rel_relation == "front":
            return obj1[2] > obj2[2]
        if rel_relation == "back":
            return obj1[2] < obj2[2]

        # Left case
        if rel_relation == "left":
            return (obj2[0] - obj1[0]) > diff_threshold
        # Right case
        return (obj1[0] - obj2[0]) > diff_threshold

    # model(images=[image], texts=[text]).item()

    # Detect the object in the image
    datum = {"image_path": image, "gt_labels": objects}
    predicted_labels, predicted_boxes, _ = dino_model([datum])
    predicted_labels = predicted_labels[0]
    predicted_boxes = predicted_boxes[0]

    # Cannot detect enough object to compare
    missing_obj = []

    # Inspried from VPEval
    obj1_list = []
    obj2_list = []
    for i, obj_name in enumerate(predicted_labels):
        box = predicted_boxes[i]
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        depth = box[4]
        if obj_name == objects[0] or gt_orientation_dir[1] == "":
            direction = ""
        else:
            direction = gt_orientation_dir[1]  # 0-index when obj_na

        if obj_name == objects[0]:
            obj1_list.append((x, y, depth, direction))

        if obj_name == objects[1]:
            obj2_list.append((x, y, depth, direction))

    # Cannot detect some object
    if len(obj1_list) == 0:
        missing_obj.append(objects[0])
    if len(obj2_list) == 0:
        missing_obj.append(objects[1])

    if len(missing_obj):  # If there is object that is missing, return -1
        return -1, -1, missing_obj

    score_intrinsic = 0
    for obj1 in obj1_list:
        for obj2 in obj2_list:
            if test_spatial_relation(obj1, obj2, check_direction=True):
                score_intrinsic = 1
                break

    score_relative = 0
    for obj1 in obj1_list:
        for obj2 in obj2_list:
            if test_spatial_relation(obj1, obj2, check_direction=False):
                score_relative = 1
                break

    return score_intrinsic, score_relative, missing_obj


def main(args, dino_model):
    args.ground_truth = False
    # Load CLIP model
    VQA_score = 0
    total = 0

    result_file = open(args.text_result_dir, 'a')

    with open(args.text_file) as data_file:
        datum = json.load(data_file)["data"]

    # Mapping context to data
    obj_total = {}
    obj_miss = {}

    acc_rel = {}
    rel_total = {}

    # Only for gt
    check_gt_dir = {}

    score_pandas = []

    map_context_data = {}
    for data in datum:
        context = data["context"]
        map_context_data[context] = data

    obj_with_dir = ["bench", "cabinet", "chair", "chicken", "dog", "cat", "deer", "horse", "tiger", "bicycle", "bus",
                    "car"]

    count_missing = 0

    for data in tqdm(datum[:], "Evaluating"):
        text = data["context"]
        img_id = data["id"]
        obj1 = data["obj1"]
        obj2 = data["obj2"]
        label = tuple(map_context_data[text]["label"])
        relation = map_context_data[text]["relation"]
        objects = [obj1, obj2]

        img_id_split = img_id.split("_")
        gt_orientation_dir_used = ["", img_id_split[1] if len(img_id_split) > 1 else ""]

        for i in range(4):
            if obj1 not in obj_total:
                obj_total[obj1] = 0
                obj_miss[obj1] = 0

            if obj2 not in obj_total:
                obj_total[obj2] = 0
                obj_miss[obj2] = 0

            if label not in acc_rel:
                acc_rel[label] = 0
                rel_total[label] = 0

            # Need to save temp image to use t2v
            obj_total[obj1] += 1
            obj_total[obj2] += 1
            rel_total[label] += 1

            img_name = f"{args.image_dir}/{img_id}_gen_{i}.png"

            if not os.path.exists(img_name):
                count_missing += 1
                score_intrinsic, score_relative, miss = 0, 0, []
                score_pandas.append([f"{img_id}_gen_{i}.png", text, label, relation, None, None, None])
            else:
                score_intrinsic, score_relative, miss = calculate_score(args, dino_model,
                                                                        img_name, text, objects,
                                                                        relation, label,
                                                                        gt_orientation_dir=gt_orientation_dir_used)
                score_pandas.append(
                    [f"{img_id}_gen_{i}.png", text, label, relation, score_intrinsic, score_relative, miss])
                # score_pandas.append([f"{img_id}_gen_{i}.png", text, label, relation, return_score, miss])

            if "external intrinsic" in label and score_intrinsic:
                VQA_score += 1
                acc_rel[label] += 1
            elif "external relative" in label and score_relative:
                VQA_score += 1
                acc_rel[label] += 1

            if score_intrinsic == -1:
                for obj in miss:
                    obj_miss[obj] += 1
            total += 1

    print(args.image_dir, file=result_file)
    print(f"Acc avg: {VQA_score * 100 / total} %", file=result_file)
    for obj in obj_total:
        print(f"Missing {obj}: {obj_miss[obj] * 100 / obj_total[obj]}", file=result_file)
    for category in rel_total:
        print(f"Acc {category}: {acc_rel[category] * 100 / rel_total[category]}", file=result_file)

    results = pd.DataFrame(score_pandas,
                           columns=["img_location", "context", "label", "relation", "score_intrinsic", "score_relative",
                                    "missing_objects"])

    results.to_csv(args.output_result_dir)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    ## GroundingDino Args
    parser.add_argument("--image_dir", type=str,
                        default="/models/GLIGEN/image_gen_4_shots_dir")
    parser.add_argument("--text_file", type=str,
                        default="/Dataset/test_dataset_v3.json")
    parser.add_argument("--grounding_dino_config_path", type=str,
                        default="VPEval/src/dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grounding_dino_weights_path", type=str, default="VPEval/weights/groundingdino_swint_ogc.pth")
    # parser.add_argument("--ground_truth", type=bool, default=False)
    parser.add_argument("--output_result_dir", type=str, default="dino_vqa_results/result")
    parser.add_argument("--text_result_dir", type=str, default="result.txt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    dino_model = DinoModel(args)
    dino_model.eval()
    main(args, dino_model)

