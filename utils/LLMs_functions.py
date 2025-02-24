import pandas as pd


def call_llm_qwen2(messages, model, max_token=1024):
    prompt = model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True)

    outputs = model(
        prompt,
        max_new_tokens=max_token,
        temperature=0.00000001,
    )
    return outputs[0]["generated_text"][len(prompt):]


def call_llm_llama3(messages, model, max_token=1024):
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
        temperature=0.00000001,
        pad_token_id=model.tokenizer.eos_token_id
    )

    return outputs[0]["generated_text"][len(prompt):]


def select_llm_caller(llm_name, model_size):
    if llm_name == "qwen2":
        model_id = f"Qwen/Qwen2-{model_size}-Instruct"
        call_llm = call_llm_qwen2
    elif llm_name == "llama3":
        model_id = f"meta-llama/Meta-Llama-3-{model_size}-Instruct"
        call_llm = call_llm_llama3
    else:
        raise ValueError(f"Invalid model name: {llm_name}")

    return call_llm, model_id
