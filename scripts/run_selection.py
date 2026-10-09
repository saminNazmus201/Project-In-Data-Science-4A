"""Run one selector over the candidate pool and save results in ALCE format.

Output: results/{method}_selected.json  =  {"args": {...}, "data": [item, ...]}
Each item keeps the fields ALCE eval.py reads (question, answer, qa_pairs,
annotations) plus `docs` = the selected docs in selection order, so that
citation [i] in a generated answer refers to docs[i-1].

Usage:
  python scripts/run_selection.py --method topk --k 5
  python scripts/run_selection.py --method mmr --k 5 --lam 0.5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import default_pool_path, load_pool, save_json
from src.embeddings import DEFAULT_MODEL, Embedder
from src.selection import get_selector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, help="topk | mmr")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--lam", type=float, default=0.5, help="MMR lambda")
    parser.add_argument("--pool", default=default_pool_path(50, 20))
    parser.add_argument("--n", type=int, default=None, help="only use the first n questions of the pool")
    parser.add_argument("--embed_model", default=DEFAULT_MODEL)
    parser.add_argument("--emb_cache", default="results/emb_cache.pkl")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--out", default=None, help="override output path")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # Windows GBK console: don't crash on accents

    pool = load_pool(args.pool)[:args.n]
    embedder = Embedder(args.embed_model, cache_path=args.emb_cache)
    selector = get_selector(args.method, lam=args.lam, embedder=embedder)

    data = []
    for item in pool:
        selected = selector.select(item["question"], item["docs"], args.k)
        data.append({
            "sample_id": item["sample_id"],
            "question": item["question"],
            "answer": item["answer"],
            "qa_pairs": item["qa_pairs"],
            "annotations": item["annotations"],
            "docs": selected,
        })
        print(f"{item['question'][:60]:60s} ranks={[d['rank'] for d in selected]}")
    embedder.save()

    out = args.out or str(Path(args.out_dir) / f"{args.method}_selected.json")
    save_json({"args": vars(args), "data": data}, out)
    print(f"\nSaved {len(data)} questions -> {out}")


if __name__ == "__main__":
    main()
