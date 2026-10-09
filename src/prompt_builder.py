"""Build ALCE-style prompts (same logic as ALCE utils.make_demo / run.py).

Prompt file: prompts/asqa_default.json, copied from princeton-nlp/ALCE. Fields:
  instruction, demo_sep, demo_prompt ("{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}"),
  doc_prompt ("Document [{ID}](Title: {T}): {P}\n"), demos
Passages are numbered [1]..[k] in the order the selector returned them.
"""
import numpy as np

from .data_loader import load_json

DEFAULT_PROMPT_FILE = "prompts/asqa_default.json"


def load_prompt_file(path=DEFAULT_PROMPT_FILE):
    return load_json(path)


def format_docs(docs, doc_prompt):
    return "".join(
        doc_prompt.replace("{T}", d["title"]).replace("{P}", d["text"]).replace("{ID}", str(i + 1))
        for i, d in enumerate(docs)
    )


def make_demo(item, prompt_data, ndoc, test=False):
    prompt = prompt_data["demo_prompt"]
    prompt = prompt.replace("{INST}", prompt_data["instruction"]).replace("{Q}", item["question"])
    if ndoc == 0:
        prompt = prompt.replace("{D}\n", "")
    else:
        prompt = prompt.replace("{D}", format_docs(item["docs"][:ndoc], prompt_data["doc_prompt"]))
    if test:
        return prompt.replace("{A}", "").rstrip()
    return prompt.replace("{A}", "").rstrip() + item["answer"]


def build_head_prompt(prompt_data, shot=2, ndoc_in_demo=5, seed=42):
    """Few-shot demos, sampled the same way as ALCE run.py (np.random.seed(seed))."""
    np.random.seed(seed)
    ids = np.random.choice(len(prompt_data["demos"]), shot, replace=False)
    head = ""
    for i in ids:
        head += make_demo(prompt_data["demos"][i], prompt_data, ndoc_in_demo)
        head += prompt_data["demo_sep"]
    return head


def build_prompt(item, prompt_data, shot=2, seed=42):
    """Full prompt for one item whose `docs` are the already-selected passages."""
    ndoc = len(item["docs"])
    head = build_head_prompt(prompt_data, shot=shot, ndoc_in_demo=ndoc, seed=seed)
    return head + make_demo(item, prompt_data, ndoc, test=True)
