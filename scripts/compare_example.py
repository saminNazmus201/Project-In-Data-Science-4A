"""Show, for one question, the passages each method selected, which answers
(qa_pairs) each passage contains, which of the candidate pool contain an answer,
and the pairwise cosine similarity between the selected passages
(to pick examples for slides).

A passage "covers" interpretation Ai if any of its short_answers appears in the
passage title + text after ALCE-style normalization (lowercase, no punctuation,
no articles, collapsed whitespace).

For SPACE / Yelp results (no qa_pairs) it shows the star rating of every
sentence instead, and the rating distribution of each selection vs. the pool.

Usage:
  python scripts/compare_example.py --idx 0
  python scripts/compare_example.py --idx 0 --methods topk mmr --width 50
  python scripts/compare_example.py --results_dir results/space_k20 --sample_id 100597_rooms --methods random topk mmr
"""
import argparse
import re
import string
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_json, load_pool
from src.embeddings import DEFAULT_MODEL, Embedder


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def covered_answers(doc, qa_pairs):
    """Indices of the qa_pairs whose short answer appears in the passage."""
    text = normalize_answer(f"{doc['title']} {doc['text']}")
    hits = []
    for i, qa in enumerate(qa_pairs):
        answers = [normalize_answer(a) for a in qa["short_answers"]]
        if any(a and f" {a} " in f" {text} " for a in answers):
            hits.append(i)
    return hits


def fmt_cov(hits):
    return ",".join(f"A{i + 1}" for i in hits) if hits else "-"


def side_by_side(columns, width):
    """columns: list of (header, list[str]) -> printed next to each other."""
    blocks = []
    for header, lines in columns:
        wrapped = [header, "-" * width]
        for line in lines:
            wrapped += textwrap.wrap(line, width) or [""]
            wrapped.append("")
        blocks.append(wrapped)
    height = max(len(b) for b in blocks)
    for row in range(height):
        print(" | ".join((b[row] if row < len(b) else "").ljust(width) for b in blocks))


def print_sim_matrix(name, docs, embedder):
    E = embedder.encode_docs(docs)
    S = E @ E.T
    k = len(docs)
    off_diag = S[~np.eye(k, dtype=bool)]
    print(f"\n[{name}] pairwise cosine similarity of selected passages "
          f"(mean off-diagonal = {off_diag.mean():.3f}, max = {off_diag.max():.3f})")
    print("      " + "".join(f"[{j + 1}]".rjust(7) for j in range(k)))
    for i in range(k):
        print(f"  [{i + 1}] " + "".join(f"{S[i, j]:7.3f}" for j in range(k)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=None, help="question index in the results files")
    parser.add_argument("--sample_id", default=None, help="instead of --idx, e.g. 100597_rooms")
    parser.add_argument("--methods", nargs="+", default=["topk", "mmr"])
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--embed_model", default=DEFAULT_MODEL)
    parser.add_argument("--emb_cache", default="results/emb_cache.pkl")
    parser.add_argument("--gen_model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="show outputs from {method}_{model}_gen.json if present")
    parser.add_argument("--width", type=int, default=45, help="column width")
    parser.add_argument("--chars", type=int, default=200, help="passage chars to show")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # Windows GBK console: don't crash on accents

    results = {m: load_json(Path(args.results_dir) / f"{m}_selected.json") for m in args.methods}
    first_data = results[args.methods[0]]["data"]
    if args.sample_id is not None:
        args.idx = next(i for i, it in enumerate(first_data) if it["sample_id"] == args.sample_id)
    elif args.idx is None:
        parser.error("give --idx or --sample_id")
    items = {m: r["data"][args.idx] for m, r in results.items()}

    pool_path = results[args.methods[0]]["args"]["pool"]
    candidates = load_pool(pool_path)[args.idx]["docs"]
    picked = {m: {d["rank"] for d in item["docs"]} for m, item in items.items()}

    if "qa_pairs" in items[args.methods[0]]:
        show_answer_coverage(args, items, candidates, picked, pool_path)
    else:
        show_ratings(args, items, candidates, picked, pool_path)

    show_generations(args)

    embedder = Embedder(args.embed_model, cache_path=args.emb_cache)
    for m, item in items.items():
        print_sim_matrix(m, item["docs"], embedder)
    embedder.save()


def show_generations(args):
    """Print the generated answer/summary of each method, if its *_gen.json exists."""
    name = args.gen_model.split("/")[-1]
    for m in args.methods:
        path = Path(args.results_dir) / f"{m}_{name}_gen.json"
        if not path.exists():
            print(f"\n[{m}] no generation yet ({path})")
            continue
        data = load_json(path)["data"]
        if args.idx < len(data):
            print(f"\n[{m}] {name} output:")
            print(textwrap.fill(data[args.idx]["output"], 100, initial_indent="  ", subsequent_indent="  "))


def show_answer_coverage(args, items, candidates, picked, pool_path):
    """ASQA: which answers (qa_pairs) the candidates and the selected passages contain."""
    first = items[args.methods[0]]
    qa_pairs = first["qa_pairs"]
    n_qa = len(qa_pairs)
    print(f"Q{args.idx}: {first['question']}")
    for i, qa in enumerate(qa_pairs):
        print(f"  [A{i + 1}] {qa['question']}  ->  {qa['short_answers']}")
    print()

    # Candidate pool: which of the candidates contain which answers, and who picked them
    pool_hits = set()
    print(f"Candidate pool ({len(candidates)} docs, {pool_path}):")
    print("  rank  " + "".join(m.upper()[:6].ljust(7) for m in args.methods)
          + "answers   title")
    for d in candidates:
        hits = covered_answers(d, qa_pairs)
        pool_hits.update(hits)
        marks = "".join(("x" if d["rank"] in picked[m] else ".").ljust(7) for m in args.methods)
        print(f"  {d['rank']:>4}  {marks}{fmt_cov(hits):<10}{d['title']}")
    print(f"  -> pool covers {len(pool_hits)}/{n_qa}: {fmt_cov(sorted(pool_hits))}\n")

    columns = []
    for m, item in items.items():
        method_hits = set()
        lines = []
        for i, d in enumerate(item["docs"]):
            hits = covered_answers(d, qa_pairs)
            method_hits.update(hits)
            lines.append(f"[{i + 1}] (rank {d['rank']}) <covers: {fmt_cov(hits)}> "
                         f"{d['title']}: {d['text'][:args.chars]}...")
        missing = sorted(set(range(n_qa)) - method_hits)
        lines.append(f"COVERED {len(method_hits)}/{n_qa}: {fmt_cov(sorted(method_hits))}"
                     f"  | missing: {fmt_cov(missing)}")
        columns.append((m.upper(), lines))
    side_by_side(columns, args.width)


def fmt_dist(counts):
    n = sum(counts) or 1
    return "  ".join(f"{s + 1}*:{c:>2} ({c / n:4.0%})" for s, c in enumerate(counts))


def tvd(p, q):
    """Total variation distance between two rating count vectors."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return 0.5 * np.abs(p / p.sum() - q / q.sum()).sum()


def show_ratings(args, items, candidates, picked, pool_path):
    """SPACE / Yelp: star rating of every sentence, and the rating distribution of
    each method's selection vs. the pool (gold proportions)."""
    first = items[args.methods[0]]
    pool_dist = [0] * 5
    for d in candidates:
        pool_dist[d["rating"] - 1] += 1
    print(f"Q{args.idx} [{first['sample_id']}]: {first['question']}")
    if first.get("summaries"):
        print(f"  reference summary: {first['summaries'][0]}")
    print()

    print(f"Candidate pool ({len(candidates)} sentences, {pool_path}):")
    print("  rank  " + "".join(m.upper()[:6].ljust(7) for m in args.methods) + "stars  sentence")
    for d in candidates:
        marks = "".join(("x" if d["rank"] in picked[m] else ".").ljust(7) for m in args.methods)
        print(f"  {d['rank']:>4}  {marks}{d['rating']:<7}{d['text'][:args.chars]}")
    print()

    name_w = max(len(m) for m in args.methods + ["pool"])
    print("Rating distribution (TVD = total variation distance to the pool, 0 = proportional):")
    print(f"  {'POOL'.ljust(name_w)}  {fmt_dist(pool_dist)}")
    for m, item in items.items():
        dist = [0] * 5
        for d in item["docs"]:
            dist[d["rating"] - 1] += 1
        print(f"  {m.upper().ljust(name_w)}  {fmt_dist(dist)}   TVD = {tvd(dist, pool_dist):.2f}")
    print()

    columns = []
    for m, item in items.items():
        lines = [f"[{i + 1}] (rank {d['rank']}, {d['rating']}*) {d['text'][:args.chars]}"
                 for i, d in enumerate(item["docs"])]
        columns.append((m.upper(), lines))
    side_by_side(columns, args.width)


if __name__ == "__main__":
    main()
