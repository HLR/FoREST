import copy
import os
import random
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def create_data(obj1_list: list, obj2_list: list, relation: str,
                context_format_list: list, label_list: list,
                specify_format_list: list = (),
                specify_variable_need: int = 0):
    data = []
    for obj1 in obj1_list:
        for obj2 in obj2_list:
            if obj1 == obj2:
                continue
            context_format = random.choice(context_format_list)
            extra_context = "" if not specify_format_list else " " + random.choice(specify_format_list)
            if specify_variable_need:
                obj2_without_a = " ".join(obj2.split()[1:])
                extra_context = extra_context.format(obj2_without_a)
            context = context_format.format(obj1, obj2) + extra_context
            context = context[0].upper() + context[1:] + "."
            data.append({"id": "",
                         "context": context,
                         "label": label_list,
                         "obj1": " ".join(obj1.split()[1:]),
                         "obj2": " ".join(obj2.split()[1:]),
                         "obj2_dir": "nan",
                         "relation": relation})
    return data


def filter_external_label(dataset):
    filter_dataset = []
    for data in dataset:
        filter_label = [label for label in data["label"] if "internal" not in label]
        if len(filter_label) == 0:
            continue
        filter_dataset.append(data)
    return copy.deepcopy(filter_dataset)


def create_json_dataset(dataset, description, name_dataset):
    dataset = pd.DataFrame(dataset)
    train, test = train_test_split(dataset, train_size=0.8, random_state=0, stratify=dataset["label"])
    dataset = dataset.to_dict('records')
    train = train.to_dict('records')
    test = test.to_dict('records')

    json.dump({"dataset": f"{description}", "data": dataset},
              open(os.path.join("Dataset", f"{name_dataset}_total.json"), "w"), indent=3)

    json.dump({"dataset": f"{description}", "data": train},
              open(os.path.join("Dataset", f"{name_dataset}_train.json"), "w"), indent=3)

    json.dump({"dataset": f"{description}", "data": test},
              open(os.path.join("Dataset", f"{name_dataset}_test.json"), "w"), indent=3)


def create_image_dataset(dataset, object_with_dir, specify_direction=False, filter_external=True):
    if filter_external:
        filter_dataset = filter_external_label(dataset)
    else:
        filter_dataset = copy.deepcopy(dataset)
    image_dataset = []
    updated_directions = ["facing toward", "facing left relative to", "facing away from", "facing right relative to"]
    directions = ["front", "left", "back", "right"]
    for image_data in filter_dataset:
        if specify_direction:
            if image_data["obj2"] in object_with_dir:
                for direction, direction_text in zip(directions, updated_directions):
                    image_data_new = copy.deepcopy(image_data)
                    if "my" in image_data_new["context"]:
                        direction_augment = " me"
                    elif "observer" in image_data_new["context"]:
                        direction_augment = " the observer"
                    else:
                        direction_augment = " the camera"
                    obj2_with_dir = "The " + image_data["obj2"] + " is " + direction_text + direction_augment + "."

                    image_data_new["context"] = image_data_new["context"] + " " + obj2_with_dir
                    image_data_new["obj2_dir"] = direction
                    image_data_new["id"] = image_data_new["id"] + f"_{direction}"
                    image_dataset.append(image_data_new)
            else:
                image_data_new = copy.deepcopy(image_data)
                image_dataset.append(image_data_new)
        else:
            image_data_new = copy.deepcopy(image_data)
            image_dataset.append(image_data_new)
    return image_dataset


def convert_obj_camera_direction(obj_direction):
    if obj_direction == "left":
        transforms_relation = {"left": "front",
                               "right": "back",
                               "front": "left",
                               "back": "right"}
    elif obj_direction == "right":
        transforms_relation = {"left": "back",
                               "right": "front",
                               "front": "right",
                               "back": "left"}

    elif obj_direction == "back":
        transforms_relation = {"left": "left",
                               "right": "right",
                               "front": "back",
                               "back": "front"}
    else:
        transforms_relation = {"left": "right",
                               "right": "left",
                               "front": "front",
                               "back": "back"}
    return transforms_relation


def create_QA_questions(dataset, obj_with_dir=None):
    qa_template_camera = [
        "Based on the camera perspective, where is the {} from the {}'s position?",
        "From the camera perspective, what is the relation of the {} to the {}?",
        "In the camera view, how is {} positioned in relation to {}?",
        "Looking through the camera perspective, how does {} appear to be oriented relative to {}'s position?",
        "Based on the camera angle, where is {} located with respect to {}'s location?"
    ]

    qa_template_intrinsic = [
        "Based on the {}'s perspective, where is the {} from the {}'s position?",
        "From the {}'s perspective, what is the relation of the {} to the {}?",
        "In the {} view, how is {} positioned in relation to {}?",
        "Looking through the {}’s perspective, how does {} appear to be oriented relative to {}'s position?",
        "Based on the {} angle, where is {} located with respect to {}'s location?"
    ]

    # context, id, label, obj1, obj2, obj2_dir, relation
    random.seed(0)
    QA_dataset = []
    for data in dataset:

        candidate_answer = []
        # if "external relative" in data["label"] or "internal relative" in data["label"]:
        #     candidate_answer.append(data["relation"])

        # if "external intrinsic" in data["label"] or "internal intrinsic" in data["label"]:
        #     # Transform relatum relation to camera
        #     relation = data["relation"]
        #     transforms_relation = convert_obj_camera_direction(data["obj2_dir"])
        #     candidate_answer.append(transforms_relation[relation])

        cond1 = "external intrinsic" in data["label"] or "internal intrinsic" in data["label"]
        cond2 = "external relative" in data["label"] or "internal relative" in data["label"]
        if obj_with_dir is not None:
            if data["obj2"] not in obj_with_dir:
                continue
            cond1, cond2 = cond2, cond1 # Change condition for checking swap
            question = random.choice(qa_template_intrinsic).format(data["obj2"], data["obj1"], data["obj2"])
        else:
            question = random.choice(qa_template_camera).format(data["obj1"], data["obj2"])

            if "my" in data["context"]:
                question = question.replace("the camera's", "my")
            elif "observer" in data["context"]:
                print(question)
                question = question.replace("the camera", "the observer's")
                print(question)

        if cond1:
            # Transform relatum/camera relation to camera/relatum
            relation = data["relation"]
            transforms_relation = convert_obj_camera_direction(data["obj2_dir"])
            candidate_answer.append(transforms_relation[relation])
        if cond2:
            candidate_answer.append(data["relation"])

        
            
            # print(question)
        QA_data = copy.deepcopy(data)
        QA_data["question"] = question
        QA_data["candidate_answer"] = candidate_answer

        QA_dataset.append(QA_data)

    return QA_dataset


def create_QA_intrinsic_question(QA_dataset, object_with_dir):
    qa_template = [
        "Based on the {}'s perspective, what is the {}'s position?",
        "If considering {}’s perspective, where is {} located relative to {}?",
        "According to {}'s perspective, where is {} located?",
        "Looking at it through B’s eyes, what is {}’s position?",
        "Based on {}’s viewpoint, where can we find {}?"
    ]


def main():
    random.seed(0)
    # 3 Big cases of frame of reference context

    # UPDATE DATASET
    small_obj = ["an umbrella", "a bag", "a suitcase", "a fire hydrant"]
    obj_with_dir = ["a bench", "a chair"]
    big_obj_without_dir = ["a water tank"]
    container = ["a box", "a container"]
    small_animal = ["a chicken", "a dog", "a cat"]
    big_animal = ["a deer", "a horse", "a cow", "a sheep"]
    small_vehicle = ["a bicycle"]
    vehicle = ["a bus", "a car"]
    plant = ["a tree"]

    intrinsic_objs = set(
        " ".join(obj.split()[1:]) for obj in obj_with_dir + small_animal + big_animal + small_vehicle + vehicle)

    dataset = []
    clear_dataset = []

    # Ambiguous case of internal and external

    # 1) case external intrinsic / relative (relatum must have direction but not inside)
    based_obj = (small_obj + obj_with_dir + big_obj_without_dir + container + small_animal +
                 big_animal + small_vehicle + vehicle + plant)
    reference_obj = obj_with_dir + small_animal + big_animal + small_vehicle
    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is to the left of {:}", "{:} is on the left of {:}"],
                        label_list=["external intrinsic", "external relative"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is to the right of {:}", "{:} is on the right of {:}"],
                        label_list=["external intrinsic", "external relative"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is in front of {:}"],
                        label_list=["external intrinsic", "external relative"])

    # Rear of
    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is behind {:}", "{:} is back of {:}"],
                        label_list=["external intrinsic", "external relative"])
    dataset.extend(data1 + data2 + data3 + data4)

    # 1.1 Making this to only external intrinsic
    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is to the left of {:}", "{:} is on the left of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["external intrinsic"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is to the right of {:}", "{:} is on the right of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["external intrinsic"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is in front of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["external intrinsic"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is behind {:}", "{:} is back of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["external intrinsic"])
    clear_dataset.extend(data1 + data2 + data3 + data4)

    # 1.2 Making this to only external relative
    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is to the left of {:}", "{:} is on the left of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["external relative"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is to the right of {:}", "{:} is on the right of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["external relative"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is in front of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["external relative"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is behind {:}", "{:} is back of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["external relative"])
    clear_dataset.extend(data1 + data2 + data3 + data4)

    # Object is containable and have direction but based object should not be inside
    based_obj = vehicle + big_obj_without_dir + ["a container"] + ["a tree"]
    reference_obj = vehicle
    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is to the left of {:}", "{:} is on the left of {:}"],
                        label_list=["external intrinsic", "external relative"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is to the right of {:}", "{:} is on the right of {:}"],
                        label_list=["external intrinsic", "external relative"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is in front of {:}"],
                        label_list=["external intrinsic", "external relative"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is behind {:}", "{:} is back of {:}"],
                        label_list=["external intrinsic", "external relative"])
    dataset.extend(data1 + data2 + data3 + data4)

    # 1.1 Making this to only external intrinsic
    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is to the left of {:}", "{:} is on the left of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["external intrinsic"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is to the right of {:}", "{:} is on the right of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["external intrinsic"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is in front of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["external intrinsic"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is behind {:}", "{:} is back of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["external intrinsic"])
    clear_dataset.extend(data1 + data2 + data3 + data4)

    # 1.2 Making this to only external relative
    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is to the left of {:}", "{:} is on the left of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["external relative"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is to the right of {:}", "{:} is on the right of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["external relative"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is in front of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["external relative"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is behind {:}", "{:} is back of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["external relative"])
    clear_dataset.extend(data1 + data2 + data3 + data4)

    # 2) case external / internal relative (relatum must have contained ability but not direction)
    based_obj = small_obj + obj_with_dir + big_obj_without_dir + small_animal + plant + ["a box"]
    reference_obj = ["a box"]
    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is on the left of {:}"],
                        label_list=["external relative", "internal relative"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is on the right of {:}"],
                        label_list=["internal relative", "external relative"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is in front of {:}"],
                        label_list=["internal relative", "external relative"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is back of {:}"],
                        label_list=["internal relative", "external relative"])
    dataset.extend(data1 + data2 + data3 + data4)

    # This will not be considered in image cases
    # 2.1 Making this to only internal
    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is on the left of {:}"],
                        specify_format_list=["inside the {:}", "within the {:}"],
                        specify_variable_need=1,
                        label_list=["internal relative"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is on the left of {:}"],
                        specify_format_list=["inside the {:}", "within the {:}"],
                        specify_variable_need=1,
                        label_list=["internal relative"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is in front of {:}"],
                        specify_format_list=["inside the {:}", "within the {:}"],
                        specify_variable_need=1,
                        label_list=["internal relative"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is back of {:}"],
                        specify_format_list=["inside the {:}", "within the {:}"],
                        specify_variable_need=1,
                        label_list=["internal relative"])
    clear_dataset.extend(data1 + data2 + data3 + data4)

    # 2.2 Making this to only external relative
    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is on the left of {:}"],
                        specify_format_list=["and outside of the {:}"],
                        specify_variable_need=1,
                        label_list=["external relative"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is on the left of {:}"],
                        specify_format_list=["and outside of the {:}"],
                        specify_variable_need=1,
                        label_list=["external relative"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is in front of {:}"],
                        specify_format_list=["and outside of the {:}"],
                        specify_variable_need=1,
                        label_list=["external relative"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is back of {:}"],
                        specify_format_list=["and outside of the {:}"],
                        specify_variable_need=1,
                        label_list=["external relative"])
    clear_dataset.extend(data1 + data2 + data3 + data4)

    based_obj = (small_obj + obj_with_dir + big_obj_without_dir + container
                 + small_animal + big_animal + small_vehicle + vehicle + plant)
    reference_obj = ["a container"]
    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is on the left of {:}"],
                        label_list=["external relative", "internal relative"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is on the right of {:}"],
                        label_list=["internal relative", "external relative"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is in front of {:}"],
                        label_list=["internal relative", "external relative"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is back of {:}"],
                        label_list=["internal relative", "external relative"])
    dataset.extend(data1 + data2 + data3 + data4)

    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is on the left of {:}"],
                        specify_format_list=["inside the {:}", "within the {:}"],
                        specify_variable_need=1,
                        label_list=["internal relative"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is on the left of {:}"],
                        specify_format_list=["inside the {:}", "within the {:}"],
                        specify_variable_need=1,
                        label_list=["internal relative"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is in front of {:}"],
                        specify_format_list=["inside the {:}", "within the {:}"],
                        specify_variable_need=1,
                        label_list=["internal relative"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is back of {:}"],
                        specify_format_list=["inside the {:}", "within the {:}"],
                        specify_variable_need=1,
                        label_list=["internal relative"])
    clear_dataset.extend(data1 + data2 + data3 + data4)

    # 2.2 Making this to only external relative
    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is on the left of {:}"],
                        specify_format_list=["and outside of the {:}"],
                        specify_variable_need=1,
                        label_list=["external relative"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is on the left of {:}"],
                        specify_format_list=["and outside of the {:}"],
                        specify_variable_need=1,
                        label_list=["external relative"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is in front of {:}"],
                        specify_format_list=["and outside of the {:}"],
                        specify_variable_need=1,
                        label_list=["external relative"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is back of {:}"],
                        specify_format_list=["and outside of the {:}"],
                        specify_variable_need=1,
                        label_list=["external relative"])
    clear_dataset.extend(data1 + data2 + data3 + data4)

    # 3) all cases (relatum have both intrinsic and contain ability)
    based_obj = small_obj + ["a box"] + obj_with_dir + big_obj_without_dir + small_animal + big_animal + ["a bicycle"]
    reference_obj = vehicle
    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is to the left of {:}", "{:} is on the left of {:}"],
                        label_list=["external intrinsic", "external relative",
                                    "internal intrinsic", "internal relative"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is to the right of {:}", "{:} is on the right of {:}"],
                        label_list=["external intrinsic", "external relative",
                                    "internal intrinsic", "internal relative"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is in front of {:}"],
                        label_list=["external intrinsic", "external relative",
                                    "internal intrinsic", "internal relative"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is behind {:}", "{:} is back of {:}"],
                        label_list=["external intrinsic", "external relative",
                                    "internal intrinsic", "internal relative"])

    dataset.extend(data1 + data2 + data3 + data4)

    # 3.1 Making this to only external intrinsic
    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is outside and to the left of {:}",
                                             "{:} is outside and on the left of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["external intrinsic"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is outside and to the right of {:}",
                                             "{:} is outside and on the right of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["external intrinsic"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is outside and in front of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["external intrinsic"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is outside and behind {:}", "{:} is outside and back of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["external intrinsic"])
    clear_dataset.extend(data1 + data2 + data3 + data4)

    # 3.2 Making this to only internal intrinsic
    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is inside and at the left of {:}",
                                             "{:} is inside and on the left of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["internal intrinsic"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is inside and at the right of {:}",
                                             "{:} is inside and on the right of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["internal intrinsic"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is inside and at front of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["internal intrinsic"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is inside and at the back of {:}"],
                        specify_format_list=["from the {:}'s perspective", "relative to the {:}"],
                        specify_variable_need=1,
                        label_list=["internal intrinsic"])
    clear_dataset.extend(data1 + data2 + data3 + data4)

    # 3.4 Making this to only external relative

    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is outside and to the left of {:}",
                                             "{:} is outside and on the left of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["external relative"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is outside and to the right of {:}",
                                             "{:} is outside and on the right of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["external relative"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is outside and in front of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["external relative"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is outside and behind {:}", "{:} is outside and back of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["external relative"])
    clear_dataset.extend(data1 + data2 + data3 + data4)

    # 3.4 Making this to only internal relative
    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is inside and at the left of {:}",
                                             "{:} is inside and on the left of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["internal relative"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is inside and at the right of {:}",
                                             "{:} is inside and on the right of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["internal relative"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is inside and at front of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["internal relative"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is inside and back of {:}"],
                        specify_format_list=["from the camera's perspective", "from my perspective",
                                             "from my point of view", "from the camera angle", "relative to camera", "relative to observer"],
                        label_list=["internal relative"])
    clear_dataset.extend(data1 + data2 + data3 + data4)

    # Non-ambiguous

    # 1) Only external relative
    based_obj = (small_obj + obj_with_dir + big_obj_without_dir
                 + container + small_animal + big_animal + small_vehicle + vehicle + plant)
    reference_obj = small_obj + big_obj_without_dir + plant

    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is to the left of {:}", "{:} is on the left of {:}"],
                        label_list=["external relative"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is to the right of {:}", "{:} is on the right of {:}"],
                        label_list=["external relative"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is in front of {:}"],
                        label_list=["external relative"])
    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is behind {:}", "{:} is back of {:}"],
                        label_list=["external relative"])
    dataset.extend(data1 + data2 + data3 + data4)
    clear_dataset.extend(data1 + data2 + data3 + data4)

    # Object should be outside the box
    based_obj = ["a container"] + big_animal + small_vehicle + vehicle
    reference_obj = ["a box"]

    data1 = create_data(based_obj, reference_obj,
                        relation="left",
                        context_format_list=["{:} is to the left of {:}", "{:} is on the left of {:}"],
                        label_list=["external relative"])

    data2 = create_data(based_obj, reference_obj,
                        relation="right",
                        context_format_list=["{:} is to the right of {:}", "{:} is on the right of {:}"],
                        label_list=["external relative"])

    data3 = create_data(based_obj, reference_obj,
                        relation="front",
                        context_format_list=["{:} is in front of {:}"],
                        label_list=["external relative"])

    data4 = create_data(based_obj, reference_obj,
                        relation="back",
                        context_format_list=["{:} is behind {:}", "{:} is back of {:}"],
                        label_list=["external relative"])
    dataset.extend(data1 + data2 + data3 + data4)
    clear_dataset.extend(data1 + data2 + data3 + data4)

    # # Object should be outside of bus and car
    # based_obj = vehicle + plant
    # reference_obj = vehicle
    #
    # data1 = create_data(based_obj, reference_obj,
    #                     relation="left",
    #                     context_format_list=["{:} is to the left of {:}", "{:} is on the left of {:}"],
    #                     label_list=["external relative"])
    #
    # data2 = create_data(based_obj, reference_obj,
    #                     relation="right",
    #                     context_format_list=["{:} is to the right of {:}", "{:} is on the right of {:}"],
    #                     label_list=["external relative"])
    #
    # data3 = create_data(based_obj, reference_obj,
    #                     relation="front",
    #                     context_format_list=["{:} is in front of {:}"],
    #                     label_list=["external relative"])
    #
    # data4 = create_data(based_obj, reference_obj,
    #                     relation="back",
    #                     context_format_list=["{:} is behind {:}", "{:} is back of {:}"],
    #                     label_list=["external relative"])
    # dataset.extend(data1 + data2 + data3 + data4)
    # clear_dataset.extend(copy.deepcopy(data1) + copy.deepcopy(data2)  + copy.deepcopy(data3)  + copy.deepcopy(data4) )

    random.shuffle(dataset)
    for i, data in enumerate(dataset):
        dataset[i]["id"] = "FRTEXT{:06d}".format(i)
    create_json_dataset(dataset, "Texture frame of reference", "text")
    print("Normal context:", len(dataset))

    random.shuffle(clear_dataset)
    for i, data in enumerate(clear_dataset):
        clear_dataset[i]["id"] = "FRCLTEXT{:06d}".format(i)
    create_json_dataset(clear_dataset, "Cleared Texture frame of reference", "clear_text")
    print("Cleared context:", len(clear_dataset))

    # Filter internal case

    image_dataset = create_image_dataset(dataset, intrinsic_objs)
    image_clear_dataset = create_image_dataset(clear_dataset, intrinsic_objs)

    image_dataset_dir = create_image_dataset(dataset, intrinsic_objs, specify_direction=True)
    image_clear_dataset_dir = create_image_dataset(clear_dataset, intrinsic_objs, specify_direction=True)

    random.shuffle(image_dataset)
    for i, data in enumerate(image_dataset):
        image_dataset[i]["id"] = "FRIMG{:06d}".format(i)
    create_json_dataset(image_dataset, "Normal prompt to generate images", "prompt_image")
    print("Normal prompt:", len(image_dataset))

    random.shuffle(image_clear_dataset)
    for i, data in enumerate(image_clear_dataset):
        image_clear_dataset[i]["id"] = "FRCLIMG{:06d}".format(i)
    create_json_dataset(image_clear_dataset, "Cleared prompt to generate images", "clear_prompt_image")
    print("Cleared prompt:", len(image_clear_dataset))

    # random.shuffle(image_dataset_dir)
    # for i, data in enumerate(image_dataset_dir):
    #     image_dataset_dir[i]["id"] = "FRIMGDI{:06d}".format(i)
    create_json_dataset(image_dataset_dir, "Normal prompt with direction to generate images", "dir_prompt_image")
    print("Normal prompt with direction:", len(image_dataset_dir))

    # random.shuffle(image_clear_dataset_dir)
    # for i, data in enumerate(image_clear_dataset_dir):
    #     image_clear_dataset_dir[i]["id"] = "FRCLIMGDI{:06d}".format(i)
    create_json_dataset(image_clear_dataset_dir, "Cleared prompt with direction to generate images",
                        "clear_dir_prompt_image")
    print("Cleared prompt with direction:", len(image_clear_dataset_dir))

    # Q1. From the camera's perspective, where is the {} positioned relative to the {}?
    # Q2. From the camera’s perspective, what is the spatial relation of the sheep to the dog?
    qa_dataset_dir = create_image_dataset(dataset, intrinsic_objs, specify_direction=True, filter_external=False)
    qa_clear_dataset_dir = create_image_dataset(clear_dataset, intrinsic_objs, specify_direction=True,
                                                filter_external=False)
    qa_ambiguous = create_QA_questions(qa_dataset_dir)
    qa_clear = create_QA_questions(qa_clear_dataset_dir)
    create_json_dataset(qa_ambiguous, "QA dataset of A-split", "text_ambiguous_question")
    create_json_dataset(qa_clear, "QA dataset of C-split", "text_clear_question")

    qa_ambiguous_intrinsic = create_QA_questions(qa_dataset_dir, obj_with_dir=intrinsic_objs)
    qa_clear_intrinsic = create_QA_questions(qa_clear_dataset_dir, obj_with_dir=intrinsic_objs)
    create_json_dataset(qa_ambiguous_intrinsic, "QA dataset of A-split relatum convert", "text_ambiguous_question_relatum")
    create_json_dataset(qa_clear_intrinsic, "QA dataset of C-split  relatum convert", "text_clear_question_relatum")


if __name__ == "__main__":
    main()
