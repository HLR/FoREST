import pandas as pd
import openai
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils.prompt import *
from utils.LLMs_functions import call_gpt_api
import os

# consider topology, distance, direction

client = openai.OpenAI(
    api_key=os.getenv('openAI_API_key'),
    organization=os.getenv('openAI_API_organization_key')
)


def setup_gpt_api_FoR(dataset, model="gpt-3.5-turbo", save_file=None,
                      few_shot=(),
                      prompt=QA_prompt,
                      additional_prompt="",
                      save_columns=("context", "label", "LLM_predict"),
                      debug=False):
    for data in tqdm(dataset):
        context = data["context"]
        question = data["question"]
        candidate_answers = data["candidate_answer"]
        message = ([{"role": "system", "content": prompt + additional_prompt}]
                   + list(few_shot)
                   + [{"role": "user", "content": f"Context: {context} Question: {question}"}])
        if debug:
            print(message)
            continue

        pred = call_gpt_api(client, message, model=model)
        result_gpt.append([context, candidate_answers, pred])

    if save_file:
        df = pd.DataFrame(result_gpt, columns=save_columns)
        df.to_csv("LLMs_results_QA/" + save_file + ".csv")



# setup_gpt_api(dataset, save_file="init_result3_explanation")
import os
import json
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--few_shot", type=int, default=4)
    parser.add_argument("--model", type=str, default="3")  # Model option: 8B, 70B
    parser.add_argument("--clear", type=bool, default=False)
    parser.add_argument("--convert_type", type=str, default="")
    parser.add_argument("--data_path", type=str, default="Dataset/C-split_QA_camera_perspective.json")
    parser.add_argument("--method", type=str, default="0-shot")
    args = parser.parse_args()

    clear = "_clear" if args.clear else ""

    with open(args.data_path) as json_file:
        data = json.load(json_file)
    dataset = data["data"]

    if args.model == "gpt-4o-mini":
        model = "gpt-4o-mini"
    if args.model == "gpt-4o":
        model = "gpt-4o-2024-11-20"
    else:
        model = "gpt-3.5-turbo-0125"
    print(model)

    if args.method == "CoT":
        print(f"Running {args.few_shot}-shot with CoT")
        setup_gpt_api(dataset,
                      model=model,
                      save_file=f"QA{args.convert_type}_{model}{clear}_dataset_COT_{args.few_shot}-shot",
                      prompt=QA_prompt_COT,
                      few_shot=QA_COT_ex if args.few_shot == 4 else [])

    elif args.method == "SG":
        print(f"Running {args.few_shot}-shot with SG+CoT")
        setup_gpt_api(dataset,
                      model=model,
                      save_file=f"QA{args.convert_type}{model}{clear}_dataset_SG_{args.few_shot}-shot",
                      prompt=QA_SG_COT,
                      few_shot=QA_SG_COT_ex if args.few_shot == 4 else [],
                      FoR_info_file=f"{model}{clear}_datasetQA_v5-4_COT_aspect_{args.few_shot}-shot.csv")

    else:
        print(f"Running {args.few_shot}-shot")
        setup_gpt_api(dataset, model=model,
                      prompt=QA_prompt,
                      save_file=f"QA{args.convert_type}_{model}{clear}_dataset_{args.few_shot}-shot",
                      few_shot=QA_few_shot if args.few_shot == 4 else [])
