"""Run one selector over the candidate pool and save results in ALCE format.

Output: {out_dir}/{method}_selected.json  =  {"args": {...}, "data": [item, ...]}
out_dir defaults to results/ for the ASQA pool and results/{pool name}/ for any
other pool (e.g. results/space_k20/).
Each item keeps every field of the pool item (for ASQA the ones ALCE eval.py
reads: question, answer, qa_pairs, annotations; for SPACE/Yelp also summaries and
rating distributions) plus `docs` = the selected docs in selection order, so that
citation [i] in a generated answer refers to docs[i-1].

Relevance query for MMR: the item's `query` field if present (review pools:
retrieval query without the entity name), otherwise `question`.

Usage:
  python scripts/run_selection.py --method topk --k 5
  python scripts/run_selection.py --method mmr --k 5 --lam 0.5
  python scripts/run_selection.py --method random --k 5 --pool pools/space_k20.json
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
    parser.add_argument("--method", required=True, help="topk | mmr | random")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--lam", type=float, default=0.5, help="MMR lambda")
    parser.add_argument("--seed", type=int, default=0, help="random selector seed")
    parser.add_argument("--pool", default=default_pool_path(50, 20))
    parser.add_argument("--n", type=int, default=None, help="only use the first n questions of the pool")
    parser.add_argument("--embed_model", default=DEFAULT_MODEL)
    parser.add_argument("--emb_cache", default="results/emb_cache.pkl")
    parser.add_argument("--out_dir", default=None, help="default: results/ (ASQA) or results/{pool name}/")
    parser.add_argument("--out", default=None, help="override output path")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # Windows GBK console: don't crash on accents

    pool = load_pool(args.pool)[:args.n]
    embedder = Embedder(args.embed_model, cache_path=args.emb_cache)
    selector = get_selector(args.method, lam=args.lam, embedder=embedder, seed=args.seed)

    data = []
    for item in pool:
        selected = selector.select(item.get("query", item["question"]), item["docs"], args.k)
        data.append({**{key: v for key, v in item.items() if key != "docs"}, "docs": selected})
        print(f"{item['question'][:60]:60s} ranks={[d['rank'] for d in selected]}")
    embedder.save()

    if args.out_dir is None:
        is_asqa = Path(args.pool).as_posix() == default_pool_path(50, 20)
        args.out_dir = "results" if is_asqa else (Path("results") / Path(args.pool).stem).as_posix()
    out = args.out or str(Path(args.out_dir) / f"{args.method}_selected.json")
    save_json({"args": vars(args), "data": data}, out)
    print(f"\nSaved {len(data)} questions -> {out}")


if __name__ == "__main__":
    main()
