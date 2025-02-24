import pandas as pd
import time
from tqdm import tqdm
import torch
import json
import os
import argparse
import torch
from utils.prompt import QA_prompt, QA_prompt_COT, QA_SG_COT, QA_few_shot, QA_COT_ex, QA_SG_COT_ex
from utils.LLMs_functions import select_llm_caller

import transformers
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

def setup_llm_call(dataset, model,
                   prompt=QA_prompt,
                   save_file=None,
                   few_shot=(), additional_prompt="",
                   save_columns=("context", "label", "LLM_predict"),
                   debug=False,
                   FoR_info_file=None):
    result_llm = []

    FoR_info = {}
    real_id = {question_info["context"]: question_info["id"] for question_info in dataset}

    if FoR_info_file:
        df = pd.read_csv("LLMs_results/{}".format(FoR_info_file))
        for _, data in df.iterrows():
            response = data["GPT_predict"]
            context = data["context"]
            if context not in real_id:
                continue
            context_id = real_id[context]
            response = response.replace("Explanation:", "Frame of Reference Explanation:").replace("Answer:",
                                                                                                   "Frame of Reference:")
            FoR_info[context_id] = response

        # Checking all question has FoR information generate
        for question in dataset:
            context_id = real_id[context]
            if context_id not in FoR_info:
                print("Error not found FoR information")
                return

    for data in tqdm(dataset):
        context = data["context"]
        question = data["question"]
        candidate_answers = data["candidate_answer"]
        if FoR_info_file:
            context_id = real_id[context]
            FoR_context = FoR_info[context_id]
            chat_msg = ([{"role": "system", "content": prompt + additional_prompt}]
                        + list(few_shot)
                        + [{"role": "user", "content": f"Context: {context} {FoR_context} Question: {question}"}])
        else:
            chat_msg = ([{"role": "system", "content": prompt + additional_prompt}]
                        + list(few_shot)
                        + [{"role": "user", "content": f"Context: {context} Question: {question}"}])
        if debug:
            print(chat_msg)
            continue

        pred = call_llm(chat_msg, model=model)
        result_llm.append([context, candidate_answers, pred])

    if save_file:
        df = pd.DataFrame(result_llm, columns=save_columns)
        df.to_csv("LLMs_results_QA/" + save_file + ".csv")


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
        model=args.model_id,
        model_kwargs={"quantization_config": quantization_config},
        # "torch_dtype": torch.bfloat16  # comment this line and uncomment below to use 4bit
        device_map="auto",
    )

    if args.method == "CoT":
        print(f"Running {args.few_shot}-shot with CoT")
        setup_llm_call(dataset,
                        model=model,
                        save_file=f"QA{args.convert_type}_{args.model_name}_{args.model_size}{clear}_dataset_COT_{args.few_shot}-shot",
                        prompt=QA_prompt_COT,
                        few_shot=QA_few_shot if args.few_shot == 4 else [])
    elif args.method == "SG":
        print(f"Running {args.few_shot}-shot with CoT+SG")
        setup_llm_call(dataset,
                       model=model,
                       save_file=f"QA{args.convert_type}_{args.model_name}_{args.model_size}{clear}_dataset_SG_{args.few_shot}-shot",
                       prompt=QA_SG_COT,
                       few_shot=QA_SG_COT_ex if args.few_shot == 4 else [],
                       FoR_info_file=f"{args.model_name}_{args.model_size}{clear}_dataset_SG-information_{args.few_shot}-shot.csv")
    else:
        print(f"Running {args.few_shot}-shot with default setting")
        setup_llm_call(dataset,
                       model=model,
                       save_file=f"QA{args.convert_type}_{args.model_name}_{args.model_size}{clear}_dataset_{args.few_shot}-shot",
                       prompt=QA_prompt,
                       few_shot=QA_few_shot if args.few_shot == 4 else [])
