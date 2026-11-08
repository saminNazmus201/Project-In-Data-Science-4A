# Author: William Degrer
import argparse
import json
import os
import time

import numpy as np

DATA_DIR = "data/space/processed"
METHOD_FILES = {
    "Top-K": DATA_DIR + "/space_top_k_results.json",
    "MMR": DATA_DIR + "/space_mmr_results.json",
    "Proportional": DATA_DIR + "/space_proportional_results.json",
}
CANDIDATES_FILE = DATA_DIR + "/space_retrieved_reviews.json"
PREPARED_FILE = DATA_DIR + "/space_prepared.json"

INSTRUCTION = ("Write a short summary (3-5 sentences) of what the reviews below say, answering the question. "
               "Use only information from the reviews. Cite every claim with the number of the review(s) it "
               "comes from, e.g. [1] or [2][3]. Write the summary as one paragraph.")
CITE_REMINDER = ("Remember: end every sentence with the number(s) of the review(s) that support it, "
                 "e.g. \"The rooms were spotless [1][4].\"")
ASPECT_PHRASE = {"cleanliness": "the cleanliness", "service": "the service"}


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_items(seed=0, k=5):
    cases = load(CANDIDATES_FILE)
    refs = {}
    for h in load(PREPARED_FILE):
        refs[str(h["entity_id"])] = h["summaries"]
    methods = {}
    for m in METHOD_FILES:
        methods[m] = {}
        for c in load(METHOD_FILES[m]):
            methods[m][(str(c["hotel_id"]), c["aspect"])] = c

    rng = np.random.default_rng(seed)
    items = []
    for c in cases:
        key = (str(c["hotel_id"]), c["aspect"])
        cands = {}
        for r in c["reviews"]:
            cands[str(r["review_id"])] = r

        selections = {}
        for m in methods:
            mc = methods[m][key]
            sel = mc["selected_reviews"] if "selected_reviews" in mc else mc["results"]
            selections[m] = [str(r["review_id"]) for r in sel]
        rand_idx = sorted(rng.choice(len(c["reviews"]), k, replace=False))
        selections["Random"] = [str(c["reviews"][i]["review_id"]) for i in rand_idx]

        for m in selections:
            docs = []
            for rid in selections[m]:
                r = cands[rid]
                docs.append({"review_id": r["review_id"], "text": r["text"], "rating": r["rating"],
                             "sentiment": r["sentiment"], "relevance_score": r["relevance_score"]})
            items.append({
                "id": key[0] + "_" + key[1] + "_" + m,
                "hotel_id": key[0],
                "hotel_name": c["hotel_name"],
                "aspect": key[1],
                "method": m,
                "question": "What do guests say about " + ASPECT_PHRASE[key[1]] + " at " + c["hotel_name"] + "?",
                "docs": docs,
                "pool_sentiment": [r["sentiment"] for r in c["reviews"]],
                "references": refs[key[0]].get(key[1], []),
            })
    return items


def build_prompt(item):
    docs = ""
    for i, d in enumerate(item["docs"]):
        docs += "Review [" + str(i + 1) + "]: " + d["text"] + "\n"
    return INSTRUCTION + "\n\nQuestion: " + item["question"] + "\n\n" + docs + "\n" + CITE_REMINDER + "\nSummary:"


def save(path, model_name, results):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "data": results}, f, ensure_ascii=False, indent=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--n_cases", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = args.out
    if out is None:
        out = "results/eval/summaries_" + args.model.split("/")[-1] + ".json"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    items = build_items()
    if args.n_cases:
        keys = sorted(set((it["hotel_id"], it["aspect"]) for it in items))[:args.n_cases]
        items = [it for it in items if (it["hotel_id"], it["aspect"]) in keys]

    # so a stopped run can continue
    done = {}
    if os.path.exists(out):
        for it in load(out)["data"]:
            if it.get("output"):
                done[it["id"]] = it

    tok = AutoTokenizer.from_pretrained(args.model)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()

    results = []
    t0 = time.time()
    for n, it in enumerate(items):
        if it["id"] in done:
            results.append(done[it["id"]])
            continue
        prompt = tok.apply_chat_template([{"role": "user", "content": build_prompt(it)}],
                                         tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        it["output"] = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        it["prompt_tokens"] = int(enc["input_ids"].shape[1])
        results.append(it)
        if (n + 1) % 10 == 0:
            save(out, args.model, results + [done[i["id"]] for i in items[n + 1:] if i["id"] in done])
            print("[%d/%d] %.0fs" % (n + 1, len(items), time.time() - t0), flush=True)
    save(out, args.model, results)
    print("Saved", len(results), "summaries to", out)


if __name__ == "__main__":
    main()
