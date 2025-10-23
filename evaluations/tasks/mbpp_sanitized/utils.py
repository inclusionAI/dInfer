import evaluate as hf_evaluate
import re
from datasets import Dataset

# Hugging Face code_eval
try:
    pass_at_k = hf_evaluate.load("code_eval")
except Exception as e:
    raise e

def process_docs(dataset: Dataset) -> Dataset:
    def _process_doc(doc):
        doc["text"] = doc["prompt"]
        doc["test_list_str"] = "\n".join(doc["test_list"])
        return doc
    return dataset.map(_process_doc)


def pass_at_1(references, predictions):
    processed_predictions = []
    for pred in predictions:
        pred_after_step1 = humaneval_postprocess(pred)
        pred_after_step2 = _process_answer(pred_after_step1)
        processed_predictions.append(pred_after_step2)

    results = pass_at_k.compute(
        references=list(references),
        predictions=[processed_predictions],
        k=[1],
    )
    return results[0]["pass@1"]


def humaneval_postprocess(text: str) -> str:
    blocks = re.findall(r'```(?:python)?\n(.*?)\n```', text, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    return text.strip()

def _process_answer(text: str) -> str:
    patterns = [
        r"'(.*)'\s*$$DONE$$",
        r"$$BEGIN$$\s*'(.*)'\s*$$DONE$$",
        r"BEGIN\s*'(.*)'\s*$$DONE$$",
        r"$$BEGIN$$\s*'(.*)'\s*DONE",
        r"BEGIN\s*'(.*)'\s*DONE",
        r"$$BEGIN$$\s*'(.*)\s*$$DONE$$",
        r"BEGIN\s*'(.*)\s*$$DONE$$",
        r"$$BEGIN$$\s*'(.*)\s*DONE",
        r"BEGIN\s*'(.*)\s*DONE",
        r'$$BEGIN$$\s*(.*)\s*$$DONE$$',
        r'BEGIN\s*(.*)\s*$$DONE$$',
        r'$$BEGIN$$\s*(.*)\s*DONE',
        r'BEGIN\s*(.*)\s*DONE',
        r'```python\s*(.*)\s*```',
        r'```\s*(.*)\s*```',
        r'```python\s*(.*)\s*$',
        r'```\s*(.*)\s*$',
        r'(.*)\s*```.*',
        r"\[BEGIN\]\s*'(.*)",
        r'\[BEGIN\](.*)',
    ]

    for p in patterns:
        match = re.search(p, text, re.DOTALL)
        if match:
            text = match.group(1)
            break
            
    text = text.split('```')[0]
    text = re.split(r"'?\s*\$\$?DONE\$\$?", text)[0]
    text = text.replace('\\_', '_')
    text = text.strip()
    return text

