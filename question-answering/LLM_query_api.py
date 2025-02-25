import pandas as pd
import openai
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils.prompt import *
import os

# consider topology, distance, direction

client = openai.OpenAI(
    api_key=os.getenv('openAI_API_key'),
    organization=os.getenv('openAI_API_organization_key')
)


def setup_gpt_api(dataset, model="gpt-3.5-turbo", save_file=None,
                  few_shot=(),
                  prompt=QA_prompt,
                  additional_prompt="",
                  save_columns=("context", "label", "LLM_predict"),
                  debug=False, FoR_info_file=None):

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

    result_gpt = []
    i = 0
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

        pred = call_gpt_api(chat_msg, model=model)
        result_gpt.append([context, candidate_answers, pred])

    if save_file:
        df = pd.DataFrame(result_gpt, columns=save_columns)
        df.to_csv("LLMs_results_QA/" + save_file + ".csv")


def call_gpt_api(message, model="gpt-3.5-turbo", temperature=0, max_token=1024, max_tried=10):
    chat_prompt = {
        "model": model,
        "messages": message,
        "temperature": temperature,
        "max_tokens": max_token
    }

    for _ in range(max_tried):
        try:
            respond = client.chat.completions.create(**chat_prompt)
            pred = respond.choices[0].message.content
            return pred
        except openai.BadRequestError as e:
            print(f"Invalid API request: {e}")
            return ""
        except openai.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
        except openai.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
        except openai.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
        except openai.AuthenticationError as e:
            print(f"OpenAI API authentication error: {e}")
            pass
        except:
            print("Other service error")
            pass
        time.sleep(12)

    return ""


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
