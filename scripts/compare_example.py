"""Show, for one question, the passages each method selected, which answers
(qa_pairs) each passage contains, which of the candidate pool contain an answer,
and the pairwise cosine similarity between the selected passages
(to pick examples for slides).

A passage "covers" interpretation Ai if any of its short_answers appears in the
passage title + text after ALCE-style normalization (lowercase, no punctuation,
no articles, collapsed whitespace).

Usage:
  python scripts/compare_example.py --idx 0
  python scripts/compare_example.py --idx 0 --methods topk mmr --width 50
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
    parser.add_argument("--idx", type=int, required=True, help="question index in the results files")
    parser.add_argument("--methods", nargs="+", default=["topk", "mmr"])
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--embed_model", default=DEFAULT_MODEL)
    parser.add_argument("--emb_cache", default="results/emb_cache.pkl")
    parser.add_argument("--width", type=int, default=45, help="column width")
    parser.add_argument("--chars", type=int, default=200, help="passage chars to show")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # Windows GBK console: don't crash on accents

    items = {m: load_json(Path(args.results_dir) / f"{m}_selected.json")["data"][args.idx]
             for m in args.methods}
    first = items[args.methods[0]]
    qa_pairs = first["qa_pairs"]
    n_qa = len(qa_pairs)
    print(f"Q{args.idx}: {first['question']}")
    for i, qa in enumerate(qa_pairs):
        print(f"  [A{i + 1}] {qa['question']}  ->  {qa['short_answers']}")
    print()

    # Candidate pool: which of the candidates contain which answers, and who picked them
    pool_path = load_json(Path(args.results_dir) / f"{args.methods[0]}_selected.json")["args"]["pool"]
    candidates = load_pool(pool_path)[args.idx]["docs"]
    picked = {m: {d["rank"] for d in item["docs"]} for m, item in items.items()}
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

    embedder = Embedder(args.embed_model, cache_path=args.emb_cache)
    for m, item in items.items():
        print_sim_matrix(m, item["docs"], embedder)
    embedder.save()


if __name__ == "__main__":
    main()
