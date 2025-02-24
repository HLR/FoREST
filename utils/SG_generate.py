import pandas as pd
import time
from tqdm import tqdm
import transformers
import torch
import json
import os
import argparse
from sklearn.metrics import confusion_matrix
import torch
from LLMs_functions import select_llm_caller
from prompt import SG_example, FoR_identification_prompt_SG
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def setup_llm_call_FoR_identification(dataset, model, prompt,
                                      save_file=None,
                                      few_shot=(), additional_prompt="",
                                      save_columns=("context", "label", "GPT_predict"),
                                      debug=False,
                                      additional_info=None):
    result_gpt = []

    i = 0
    for data in tqdm(dataset):
        context, label = data["context"], data["label"]
        if additional_info is not None:
            explanation = additional_info[i]
            chat_msg = ([{"role": "system", "content": prompt + additional_prompt}]
                        + list(few_shot)
                        + [{"role": "user",
                            "content": "context:" + context + "\n" + "extra information:" + explanation}])
        else:
            chat_msg = ([{"role": "system", "content": prompt + additional_prompt}]
                        + list(few_shot)
                        + [{"role": "user", "content": "context:" + context}])
        if debug:
            print(chat_msg)
            continue

        pred = call_llm(chat_msg, model=model)
        result_gpt.append([context, label, pred])
        i += 1

    if save_file:
        df = pd.DataFrame(result_gpt, columns=save_columns)
        df.to_csv("LLMs_results/" + save_file + ".csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--few_shot", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="Llama3")  # Model option: 7B, 70B
    parser.add_argument("--model_size", type=str, default="7B")  # Model option: 7B, 70B
    parser.add_argument("--clear", type=bool, default=False)
    parser.add_argument("--data_path", type=str, default="Dataset/C-split_QA_camera_perspective.json")
    parser.add_argument("--convert_type", type=str, default="")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    clear = "_clear" if args.clear else ""

    with open(args.data_path) as json_file:
        data = json.load(json_file)
    dataset = data["data"]

    call_llm, model_id = select_llm_caller(args.model_name.lower(), args.model_size)

    model = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"quantization_config": quantization_config},
        # "torch_dtype": torch.bfloat16  # comment this line and uncomment below to use 4bit
        device_map="auto",
    )

    print(f"Running {args.few_shot}-shot to generate spatial information + SG")

    setup_llm_call_FoR_identification(dataset, model=model,
                   save_file=f"{args.model_name}_{args.model_size}{clear}_dataset_SG-information_{args.few_shot}-shot.csv",
                   prompt=FoR_identification_prompt_SG,
                   few_shot=SG_example if args.few_shot == 4 else [])
