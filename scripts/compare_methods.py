"""Aggregate comparison of the generated summaries of several selectors on a
review pool (SPACE / Yelp). Exploratory only, not the group's official evaluation.

Columns:
  TVD        rating distribution of the selected sentences vs. the pool (0 = proportional)
  cited%     summaries with at least one [i] citation
  cites/sum  average number of [i] citations per summary
  bad cite   summaries citing a review that does not exist (e.g. [6] with 5 reviews)
  words      average summary length
  ROUGE-L    F1 against the human reference summaries, averaged over the references
             (SPACE only; citations are removed first)

Evidence table (with --evidence):
  rank       average retrieval rank of the selected sentences (0 = most similar to the query)
  relevance  average cosine similarity between the query and the selected sentences
  redundancy average pairwise cosine similarity among the selected sentences
  drop-neg%  queries whose pool has a sentence from a review with <= 3 stars but whose
             selection has none (the minority opinion disappears)

Usage:
  python scripts/compare_methods.py --results_dir results/space_k20
  python scripts/compare_methods.py --results_dir results/yelp_k20 --methods random topk mmr
"""
import argparse
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_json
from src.embeddings import DEFAULT_MODEL, Embedder


def tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(candidate, reference):
    c = tokens(re.sub(r"\[\d+\]", " ", candidate))
    r = tokens(reference)
    lcs = lcs_len(c, r)
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(c), lcs / len(r)
    return 2 * p * rec / (p + rec)


def tvd(docs, pool_dist):
    counts = [0] * 5
    for d in docs:
        counts[d["rating"] - 1] += 1
    n, n_pool = sum(counts), sum(pool_dist)
    return 0.5 * sum(abs(a / n - b / n_pool) for a, b in zip(counts, pool_dist))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/space_k20")
    parser.add_argument("--methods", nargs="+", default=["random", "topk", "mmr"])
    parser.add_argument("--gen_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--evidence", action="store_true", help="also compare the selected evidence")
    parser.add_argument("--embed_model", default=DEFAULT_MODEL)
    parser.add_argument("--emb_cache", default="results/emb_cache.pkl")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # Windows GBK console: don't crash on accents

    name = args.gen_model.split("/")[-1]
    print(f"{'method':8s} {'n':>4s} {'TVD':>6s} {'cited%':>7s} {'cites/sum':>9s} "
          f"{'bad cite':>8s} {'words':>6s} {'ROUGE-L':>8s}")
    for m in args.methods:
        path = Path(args.results_dir) / f"{m}_{name}_gen.json"
        if not path.exists():
            print(f"{m:8s} missing: {path}")
            continue
        data = load_json(path)["data"]
        tvds, n_cites, words, rouge = [], [], [], []
        cited = bad = 0
        for item in data:
            out = item["output"]
            ids = [int(x) for x in re.findall(r"\[(\d+)\]", out)]
            tvds.append(tvd(item["docs"], item["pool_rating_dist"]))
            cited += bool(ids)
            bad += any(i < 1 or i > len(item["docs"]) for i in ids)
            n_cites.append(len(ids))
            words.append(len(out.split()))
            if item.get("summaries"):
                rouge.append(st.mean(rouge_l(out, ref) for ref in item["summaries"]))
        n = len(data)
        rouge_str = f"{st.mean(rouge):8.3f}" if rouge else f"{'-':>8s}"
        print(f"{m:8s} {n:4d} {st.mean(tvds):6.3f} {cited / n:7.1%} {st.mean(n_cites):9.1f} "
              f"{bad:8d} {st.mean(words):6.0f} {rouge_str}")

    if args.evidence:
        compare_evidence(args)


def compare_evidence(args):
    import numpy as np

    embedder = Embedder(args.embed_model, cache_path=args.emb_cache)
    print(f"\n{'method':8s} {'rank':>5s} {'relevance':>9s} {'redundancy':>10s} {'drop-neg%':>9s}")
    for m in args.methods:
        path = Path(args.results_dir) / f"{m}_selected.json"
        if not path.exists():
            print(f"{m:8s} missing: {path}")
            continue
        data = load_json(path)["data"]
        ranks, rel, red = [], [], []
        drop = has_neg = 0
        for item in data:
            docs = item["docs"]
            q = embedder.encode_query(item.get("query", item["question"]))
            E = embedder.encode_docs(docs)
            S = E @ E.T
            ranks.append(np.mean([d["rank"] for d in docs]))
            rel.append(float((E @ q).mean()))
            red.append(float(S[~np.eye(len(docs), dtype=bool)].mean()))
            if sum(item["pool_rating_dist"][:3]) > 0:
                has_neg += 1
                drop += all(d["rating"] > 3 for d in docs)
        print(f"{m:8s} {st.mean(ranks):5.1f} {st.mean(rel):9.3f} {st.mean(red):10.3f} "
              f"{drop / max(has_neg, 1):9.1%}")
    embedder.save()


if __name__ == "__main__":
    main()
