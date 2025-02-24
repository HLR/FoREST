# Question Answering (QA)

---

## Task Description

**Under development**


---

## Evaluation Models

**Under development**

---
## Evaluation Results

This evaluates the result from LLMs model on question-answering that requires frame of reference understanding.
Questions ask the spatial relation from the given context. So, the output is expected to be one of {left, right, front, back}.
The evaluation program already extract the answer from templated expected from LLM. 

To running the evaluation, please prepare the result in .csv file with at least three columns for ["context", "label", "LLM_predict"]

```bash
python evaluation_QA.py
```

All options for parameters are described below
- --clear: indicate whether the result from clear (C-split) or ambiguous (A-split)
- --result_file: address of result obtain from LLMs
- --specific_label: whether to specify FoR class to evaluate.
- --show_confusion_matrix: whether to show confusion matrix of the result

**Note** All FoR class for C-split are ["]. All FoR class for A-split are ["cow", "car", "box", "pen"]

---