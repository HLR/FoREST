# Question Answering (QA)
![QA Evaluation](QA_evaluation.gif)

---

## Task Description

This QA task evaluates LLMs’ ability to adapt contextual perspectives across different FoRs.
Both A and C splits are used in this task. 
The input is the context, consisting of a spatial expression (T) and relatum orientation, if available, and a question (Q) that queries the spatial relation from either an observer or the relatum’s perspective. 
The output is a spatial relation (S), restricted to {left, right, front, back}


---

## Evaluation Models

We evaluate on three models: Llama3.0-70B, Qwen2-72B, and GPT-4o.

**Zero-shot baseline.**
We call the LLM with instructions, a spatial context (T) and a question (Q) expecting a spatial relation as the response. 
The prompt instructs the model to answer the question with one of the candidate spatial relations without any explanations.


**Few-shot baseline.**
We create four spatial expressions, each assigned to a single FoR class to prevent bias. 
We generate a corresponding question and answer for each. 
These serve as examples in our few-shot prompting. 
The input to the model is instruction, example, spatial context, and the question.

**Chain-of-Thought baseline.**
To create Chain-of-Thought (CoT) examples, we modify the prompt to require reasoning before answering.
We manually crafted reasoning explanations with the necessary information for each example we used in the few-shot setting.
The input to the model is instruction, CoT example, spatial context, and the question.


The command for conducing experiment is listed below,

```bash
python LLM_query_open_source.py
```

All options for parameters are described below
- --cuda: number of cuda used to load model
- --few_shot: The option is either "0" or "4" indicating number of example used for prompt engineering.
- --model_name: All options for this experiment are {llama3, qwen2}.
- --model_size: For llama3, options are {8B, 70B}. For qwen2, options are {7B, 72B}.
- --clear: Whether the dataset evaluation come from either clear (C-split) or ambiguous (A-split). Enter any character to indicate C-split; otherwise left empty.
- --data_path: directory for evaluated data.
- --convert_type: Whether "intrinsic" or "camera". Default is "" which can be used as "camera".

Example of command with all parameters

```bash
python LLM_query_open_source.py --cuda 0 --few_shot 4 --model_name llama3 --model_size 70B --clear T --data_path Dataset/test.json --convert_type intrinsic
```

**Note:** 
- Some model like Llama3 requires signing agreement forms please refers to their original document in huggingface. 


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

This evaluates the result from the LLMs model on question-answering that requires a frame of reference understanding.
Questions ask the spatial relation from the given context. 
So, the output is expected to be one of {left, right, front, back}.
The evaluation program extracts the answer from the template expected from LLM. 

The A-split (ambiguous) context can have multiple correct answers, while the C-split only has one correct answer.
Examples of evaluation is provided below, where red boxes represent the wrong answer, while green boxes represent the correct answer.
**The images only for illustration and are not included as the input.**
![QA Evaluation](QA_evaluation.gif)


To running the evaluation, please prepare the result in .csv file with at least three columns for ["context", "label", "LLM_predict"]

```bash
python evaluation_QA.py
```

All options for parameters are described below
- --clear: Use '--clear T' for C-split; otherwise (A-split) ignore this option.
- --result_file: address of result obtain from LLMs, specify the file address (--result_file Dataset/result.csv)
- --specific_label: whether to specify FoR class to evaluate. All option described below in notes.
- --show_confusion_matrix: whether to show confusion matrix of the result. Use '--show_confusion_matrix T' for showing confusion matrix; otherwise ignore this option.

**Note** All FoR class for C-split are ["external relative", "external intrinsic", "internal intrinsic", "internal relative"]. All FoR class for A-split are ["cow", "car", "box", "pen"]

---