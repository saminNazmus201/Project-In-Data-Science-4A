"""Load ASQA from ALCE and build a fixed candidate pool shared by the whole group.

Raw ALCE fields (checked on asqa_eval_gtr_top100.json):
  item: qa_pairs, wikipages, annotations, sample_id, question, docs, answer
  doc:  id, title, text, score, summary, extraction
Docs are already sorted by GTR `score` (descending), i.e. by retrieval rank.

Usage:
  python -m src.data_loader --data_dir data --n 50 --pool_size 20
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_DATA_DIR = "data"
DEFAULT_DATA_FILE = "asqa_eval_gtr_top100.json"
DEFAULT_POOL_DIR = "pools"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def default_pool_path(n, pool_size, pool_dir=DEFAULT_POOL_DIR):
    return (Path(pool_dir) / f"asqa_gtr_n{n}_k{pool_size}.json").as_posix()


def build_pool(data_path, n=50, pool_size=20):
    """Keep the first n questions and the top `pool_size` docs of each."""
    raw = load_json(data_path)
    pool = []
    for item in raw[:n]:
        docs = []
        for rank, doc in enumerate(item["docs"][:pool_size]):
            docs.append({"rank": rank, **doc})  # rank 0 = best GTR score
        pool.append({
            "sample_id": item["sample_id"],
            "question": item["question"],
            "answer": item["answer"],
            "qa_pairs": item["qa_pairs"],
            "annotations": item["annotations"],
            "docs": docs,
        })
    return pool


def load_pool(path):
    return load_json(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--data_file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--n", type=int, default=50, help="number of questions")
    parser.add_argument("--pool_size", type=int, default=20, help="candidate docs per question")
    parser.add_argument("--out", default=None, help="output path (default: pools/asqa_gtr_n{n}_k{pool_size}.json)")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # Windows GBK console: don't crash on accents

    pool = build_pool(Path(args.data_dir) / args.data_file, args.n, args.pool_size)
    out = args.out or default_pool_path(args.n, args.pool_size)
    save_json(pool, out)

    print(f"Saved {len(pool)} questions x {args.pool_size} docs -> {out}")
    for i, item in enumerate(pool[:3]):
        print(f"\n[{i}] {item['question']}")
        print(f"    qa_pairs: {len(item['qa_pairs'])}, docs: {len(item['docs'])}")
        for d in item["docs"][:3]:
            print(f"    rank={d['rank']} score={d['score']:.4f} {d['title']}")


if __name__ == "__main__":
    main()
