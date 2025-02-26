import pandas as pd
import time
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import argparse
from sklearn.metrics import confusion_matrix
import torch
from utils.LLMs_functions import select_llm_caller
from utils.prompt import FoR_identification_prompt, FoR_identification_prompt_COT, FoR_identification_prompt_SG, \
    SG_example, COT, few_shot
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
                                      debug=False, ):
    result_gpt = []

    i = 0
    for data in tqdm(dataset):
        context, label = data["context"], data["label"]
        message = ([{"role": "system", "content": prompt + additional_prompt}]
                   + list(few_shot)
                   + [{"role": "user", "content": "context:" + context}])
        if debug:
            print(message)
            continue

        pred = call_llm(message, model=model)
        result_gpt.append([context, label, pred])
        i += 1

    if save_file:
        df = pd.DataFrame(result_gpt, columns=save_columns)
        df.to_csv("LLMs_results_FoR/" + save_file + ".csv")


def setup_llm_call_Gemma(dataset, model, prompt,
                         save_file=None,
                         few_shot=(), additional_prompt="",
                         save_columns=("context", "label", "GPT_predict"),
                         debug=False):
    result_LLM = []

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

    print(model_id, "debug:", debug)
    i = 0
    for data in tqdm(dataset[:]):
        context, label = data["context"], data["label"]
        chat_msg = ([{"role": "user", "content": prompt + additional_prompt}]
                    + [{"role": "assistant", "content": "Sure, I got the assignment."}]
                    + list(few_shot)
                    + [{"role": "user", "content": "context:" + context}])
        if debug:
            print(chat_msg)
            continue

        pred = call_llm(chat_msg, model=model, tokenizer=tokenizer)
        pattern = r"<start_of_turn>model\n(.*)\n<end_of_turn><eos>"

        st_answer = pred.rfind("<start_of_turn>model")
        ed_answer = pred.rfind("<end_of_turn>")

        if st_answer != -1 and ed_answer != -1:
            pred = pred[st_answer + len("<start_of_turn>model"): ed_answer]

        result_LLM.append([context, label, pred])
        i += 1

    if save_file:
        df = pd.DataFrame(result_LLM, columns=save_columns)
        df.to_csv("LLMs_results_FoR/" + save_file + ".csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--few_shot", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="Llama3")  # Model option: 7B, 70B
    parser.add_argument("--model_size", type=str, default="7B")  # Model option: 7B, 70B
    parser.add_argument("--clear", type=bool, default=False)
    parser.add_argument("--data_path", type=str, default="Dataset/C-split_QA_camera_perspective.json")
    parser.add_argument("--method", type=str, default="0-shot")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    clear = "_clear" if args.clear else ""

    with open(args.data_path) as json_file:
        data = json.load(json_file)
    dataset = data["data"]

    call_llm, model_id = select_llm_caller(args.model_name.lower(), args.model_size)

    if args.model_name == "gemma":
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it",
                                                     quantization_config=quantization_config, device_map="auto")
        setup_call = setup_llm_call_FoR_identification
    else:
        model = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"quantization_config": quantization_config},
            # "torch_dtype": torch.bfloat16  # comment this line and uncomment below to use 4bit
            device_map="auto", )
        setup_call = setup_llm_call_Gemma

    print(f"Running {args.few_shot}-shot to generate spatial information + SG")

    if args.method == "CoT":
        print(f"Running {args.few_shot}-shot with CoT")
        setup_call(dataset,
                   model=model,
                   save_file=f"QA_{args.model_name}_{args.model_size}{clear}_dataset_COT_{args.few_shot}-shot",
                   prompt=FoR_identification_prompt_COT,
                   few_shot=COT if args.few_shot == 4 else [])
    elif args.method == "SG":
        print(f"Running {args.few_shot}-shot with CoT+SG")
        setup_call(dataset,
                   model=model,
                   save_file=f"QA_{args.model_name}_{args.model_size}{clear}_dataset_SG_{args.few_shot}-shot",
                   prompt=FoR_identification_prompt_SG,
                   few_shot=SG_example if args.few_shot == 4 else [])
    else:
        print(f"Running {args.few_shot}-shot with default setting")
        setup_call(dataset,
                   model=model,
                   save_file=f"QA_{args.model_name}_{args.model_size}{clear}_dataset_{args.few_shot}-shot",
                   prompt=FoR_identification_prompt,
                   few_shot=few_shot if args.few_shot == 4 else [])
