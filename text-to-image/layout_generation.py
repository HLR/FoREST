import pandas as pd
import time
from tqdm import tqdm
import transformers
import torch
import json
import os
from sklearn.metrics import confusion_matrix

# Optional quantization to 4bit
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# consider topology, distance, direction

prompt_layout = (
    "Your task is to generate the bounding boxes of objects mentioned in the caption, along with direction that objects facing. "
    "The image is size 512x512."
    "The bounding box should be in the format of (x, y, width, height)."
    "The direction that object is facing should be one of these options, [front, back, left, right]"
    "Please considering the frame of reference of caption and direction of reference object."
    "If needed, you can make the reasonable guess.")


def call_llm(messages, model, max_token=2048):
    prompt = model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True)

    terminators = [
        model.tokenizer.eos_token_id,
        model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model(
        prompt,
        max_new_tokens=max_token,
        eos_token_id=terminators,
        temperature=0.0000001,
        pad_token_id=model.tokenizer.eos_token_id
    )

    return outputs[0]["generated_text"][len(prompt):]


def setup_llm_call(model, dataset, prompt=prompt_layout, model_id="meta-llama/Meta-Llama-3-8B-Instruct",
                   save_file=None,
                   few_shot=(), additional_info_data=None,
                   save_columns=("id", "caption", "layout"),
                   debug=False):
    result_llm = []

    print(model_id, "debug:", debug)
    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]
        context, id = data["context"], data["id"]
        addition_info = ""
        if additional_info_data is not None:  # Information used to help classified text
            addition_info = additional_info_data[idx]

        chat_msg = ([{"role": "system", "content": prompt + addition_info}]
                    + list(few_shot)
                    + [{"role": "user", "content": "context:" + context}])
        if debug:
            print(chat_msg)
            continue
        pred = call_llm(chat_msg, model=model)
        result_llm.append([f"{id}", context, pred])

    if save_file:
        df = pd.DataFrame(result_llm, columns=save_columns)
        df.to_csv("LLMs_results/" + save_file + ".csv")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--model_size", type=str, default="8B")
    parser.add_argument("--few_shot", type=int, default=4)
    parser.add_argument("--clear", type=bool, default=False)
    parser.add_argument("--SG_information", type=bool, default=False)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    # for i in range(1):
    #     args.clear = ((i % 2) == 0)

    if args.clear:
        print("Generate Layout for clear prompt")
        path = "clear_dir_prompt_image_total"
        clear_tag = "_clear"
    else:
        print("Generate Layout for ambiguous prompt")
        path = "dir_prompt_image_total"
        clear_tag = ""

    addition_tag = ""
    additional_data = None

    if args.SG_information:
        print("Using spatial information + frame of reference regarding context")
        addition_tag = "_augmented_info"
        with open( f"llama3_{args.model_size}{clear_tag}_dataset_SG-information_{args.few_shot}-shot.csv") as data_file:
            additional_data = pd.read_csv(data_file)["GPT_predict"].tolist()

    data_path = os.path.join("Dataset", f"{path}.json")
    with open(data_path) as json_file:
        data = json.load(json_file)
    dataset = data["data"]

    with open(f"few_shot_layout/layout_{args.few_shot}_shots_dir.json", 'r') as file:
        few_shot_example = json.load(file)

    model_id = f"meta-llama/Meta-Llama-3-{args.model_size}-Instruct"

    model = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"quantization_config": quantization_config},
        device_map="auto",
    )

    setup_llm_call(model, dataset, save_file=f"{path}_layout_dir_{args.few_shot}_shots_{args.model_size}{addition_tag}",
                   prompt=prompt_layout,
                   additional_info_data=additional_data,
                   few_shot=few_shot_example)
