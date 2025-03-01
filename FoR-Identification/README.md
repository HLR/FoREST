# Frame of Reference Identification

---

## Task Description

We evaluate the LLMs' performance in recognizing the FoR classes from given spatial expressions. 
The LLMs receive spatial expression, denoted as $T$, and output one FoR class from the valid set of FoR classes, 
{external relative, external intrinsic, internal intrinsic, internal relative}. 


---

## Evaluation Models

We evaluate on three models: Llama3.0-70B, Qwen2-72B, and GPT-4o.

**Zero-shot baseline.**

We call the LLM with instructions, a spatial context (T) expecting the Frame of Reference class as the response. 
The prompt instructs the model to answer the question with one of the candidate Frame of Reference classes without any explanations.


**Few-shot baseline.**

We manually craft four spatial expressions for each FoR class. 
To avoid creating bias, each spatial expression is ensured to fit in only one FoR class. These expressions serve as examples of our \textit{few-shot}setting.
We provide these examples in addition to the instruction as a part of the prompt, followed by T and query FoR from the LLM.

**Chain-of-Thought baseline.**

We modify the prompt to require reasoning before answering.
Then, we manually crafted reasoning explanations with the necessary information for each example used in few-shot.
Finally, we call the LLMs, adding modified instructions to updated examples, followed by T and query F. 

**SG Prompting** 

we propose Spatial-Guided Prompting to direct the model in identifying the type of relations before querying F. 
We revise the prompting instruction to guide the model in considering these three aspects. 
Then, we manually explain these three aspects.
We insert these new explanations in examples and call the model with the updated instructions followed by T to query F.

### Commands To Run
The command for conducing experiment is listed below,

```bash
python LLM_query_OS.py
```

All options for parameters are described below
- --cuda: number of cuda used to load model
- --few_shot: The option is either "0" or "4" indicating number of example used for prompt engineering.
- --model_name: All options for this experiment are {llama3, qwen2, gemma}.
- --model_size: For llama3, options are {8B, 70B}. For qwen2, options are {7B, 72B}. No option of model size for Gemma
- --clear: Whether the dataset evaluation come from either clear (C-split) or ambiguous (A-split). Enter any character to indicate C-split; otherwise left empty.
- --data_path: directory for evaluated data.
- --method: which method to run include ["CoT", "SG"]. ignore this option if running without these two methods.


**Note:** 
- Some model like Llama3 requires signing agreement forms, please refer to their original document in huggingface. 


**FoR GPT4o**, please use following command instead. The parameters are the same for this command except cuda,

```bash
python LLM_query_api.py --model gpt-4o
```

**Note for GPT4o:**
- The model options (``--model``) are gpt-4o-mini, gpt-4o, and gpt3.5-turbo. All invalid options will be used gpt3.5-turbo.
- Running with GPT4o require the API key which should be replaced, ``api_key=os.getenv('openAI_API_key')``
- The organization key, ``organization=os.getenv('openAI_API_organization_key')``,can be ignored or removed.

---
## Evaluation Results
The output is expected to be one of {external relative, external intrinsic, internal intrinsic, internal relative}.
The evaluation program extracts the answer from the template expected from LLM. 

The A-split (ambiguous) context can have multiple correct answers, while the C-split only has one correct answer.

To running the evaluation, please prepare the result in .csv file with at least three columns for ["context", "label", "LLM_predict"]

```bash
python evaluation_QA.py
```

All options for parameters are described below
- --clear: Use '--clear T' for C-split; otherwise (A-split) ignore this option.
- --result_file: address of result obtain from LLMs, specify the file address (--result_file Dataset/result.csv)

---
