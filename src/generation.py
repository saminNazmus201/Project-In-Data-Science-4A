"""Generate cited answers for a selection file. Backend: HuggingFace or an
OpenAI-compatible API. Output is ALCE eval.py format with `output` added.

Usage:
  # just print prompts, no model
  python -m src.generation --input results/topk_selected.json --dry_run
  # HuggingFace (e.g. on Colab GPU)
  python -m src.generation --input results/topk_selected.json --backend hf --model Qwen/Qwen2.5-3B-Instruct
  # OpenAI-compatible API (key from env OPENAI_API_KEY; --api_base for other providers)
  python -m src.generation --input results/topk_selected.json --backend api --model gpt-4o-mini

Note: ALCE eval.py keeps only the first line of `output`.
"""
import argparse
import os
import sys
from pathlib import Path

from .data_loader import load_json, save_json
from .prompt_builder import DEFAULT_PROMPT_FILE, build_prompt, load_prompt_file


class HFGenerator:
    def __init__(self, model_name, use_chat_template=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto")
        self.use_chat_template = use_chat_template and self.tokenizer.chat_template is not None

    def generate(self, prompt, max_new_tokens, temperature, top_p):
        if self.use_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=not self.use_chat_template)
        inputs = inputs.to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class APIGenerator:
    def __init__(self, model_name, api_base=None):
        from openai import OpenAI
        self.model_name = model_name
        self.client = OpenAI(base_url=api_base or os.environ.get("OPENAI_BASE_URL"))

    def generate(self, prompt, max_new_tokens, temperature, top_p):
        resp = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        return resp.choices[0].message.content


def get_generator(backend, model_name, api_base=None, use_chat_template=True):
    if backend == "hf":
        return HFGenerator(model_name, use_chat_template)
    if backend == "api":
        return APIGenerator(model_name, api_base)
    raise ValueError(f"Unknown backend: {backend}")


def default_output_path(input_path, model_name):
    p = Path(input_path)
    stem = p.stem.replace("_selected", "")
    return (p.parent / f"{stem}_{model_name.split('/')[-1]}_gen.json").as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="results/{method}_selected.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--backend", default="hf", choices=["hf", "api"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--api_base", default=None, help="base URL for OpenAI-compatible APIs")
    parser.add_argument("--no_chat_template", action="store_true", help="HF: feed raw prompt (base models)")
    parser.add_argument("--prompt_file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--shot", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=None, help="only the first n questions")
    # ALCE run.py defaults
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--dry_run", action="store_true", help="print prompts only, do not call a model")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # Windows GBK console: don't crash on accents

    selected = load_json(args.input)
    data = selected["data"][:args.n]
    prompt_data = load_prompt_file(args.prompt_file)
    for item in data:
        item["prompt"] = build_prompt(item, prompt_data, shot=args.shot, seed=args.seed)

    if args.dry_run:
        for i, item in enumerate(data):
            print(f"\n{'=' * 30} [{i}] prompt ({len(item['prompt'])} chars) {'=' * 30}")
            print(item["prompt"])
        return

    gen = get_generator(args.backend, args.model, args.api_base, not args.no_chat_template)
    for i, item in enumerate(data):
        item["output"] = gen.generate(item["prompt"], args.max_new_tokens, args.temperature, args.top_p).strip()
        print(f"[{i}] {item['question']}\n    -> {item['output']}\n")

    out = args.output or default_output_path(args.input, args.model)
    save_json({"args": {**selected["args"], "generation": vars(args)}, "data": data}, out)
    print(f"Saved {len(data)} generations -> {out}")


if __name__ == "__main__":
    main()
