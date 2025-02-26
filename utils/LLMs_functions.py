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


def call_gpt_api(client, message, model="gpt-3.5-turbo", temperature=0, max_token=1024, max_tried=10):
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