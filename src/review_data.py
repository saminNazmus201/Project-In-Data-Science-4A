"""Build candidate pools of review sentences for SPACE (hotels) and Yelp (restaurants).

For every entity (hotel / restaurant) and every query (general + aspects) we
retrieve the top `pool_size` review sentences of that entity with a dense
retriever (BGE, cosine similarity). Each sentence keeps the rating of the review
it came from, so the rating distribution of the pool is the gold proportion.

SPACE (Angelidis et al., TACL 2021), file space_summ.json from
github.com/stangelid/qt (data/json/space_summ.json):
  entity: entity_id, entity_name, reviews, summaries
  review: review_id, sentences (already split), rating (1-5)
  summaries: {general, building, cleanliness, food, location, rooms, service} -> 3 human summaries each

Yelp Open Dataset (yelp.com/dataset, download yourself after accepting the terms):
  yelp_academic_dataset_business.json: business_id, name, city, latitude, longitude, categories, review_count, ...
  yelp_academic_dataset_review.json:   review_id, business_id, stars, text, ...
  One JSON object per line. No reference summaries.

Pool item fields:
  dataset, sample_id, entity_id, entity_name, aspect,
  question            question shown to the LLM (contains the entity name)
  query               retrieval query (no entity name; MMR also uses it as relevance query)
  summaries           human reference summaries for (entity, aspect); [] for Yelp
  entity_rating_dist  counts of ratings 1..5 over all reviews of the entity (index 0 = 1 star)
  pool_rating_dist    counts of ratings 1..5 over the pool sentences (gold proportions)
  latitude, longitude Yelp only
  docs                [{rank, id, review_id, sent_idx, rating, title, text, score}], rank 0 = most similar
                      (`title` is "" so the doc format matches the ASQA pools)

Usage:
  python -m src.review_data --dataset space
  python -m src.review_data --dataset yelp --yelp_dir data/yelp --n_entities 100 --city Philadelphia
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

from .data_loader import save_json
from .embeddings import DEFAULT_MODEL, Embedder

DEFAULT_SPACE_FILE = "data/space/space_summ.json"
DEFAULT_YELP_DIR = "data/yelp"

# aspect -> (phrase for the LLM question, retrieval query)
SPACE_ASPECTS = {
    "general": ("", "What do guests think of this hotel overall?"),
    "building": ("the building", "What do guests say about the hotel building, lobby, architecture and decor?"),
    "cleanliness": ("the cleanliness", "How clean are the hotel and the rooms?"),
    "food": ("the food", "What do guests say about the food, breakfast and restaurant?"),
    "location": ("the location", "What do guests say about the hotel's location and surroundings?"),
    "rooms": ("the rooms", "What do guests say about the rooms, beds and bathrooms?"),
    "service": ("the service", "What do guests say about the staff and service?"),
}
YELP_ASPECTS = {
    "general": ("", "What do customers think of this restaurant overall?"),
    "food": ("the food", "What do customers say about the food, dishes and taste?"),
    "service": ("the service", "What do customers say about the staff, service and waiting time?"),
    "price": ("the prices", "What do customers say about the prices and value for money?"),
    "ambience": ("the ambience", "What do customers say about the atmosphere, decor and noise?"),
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def make_question(dataset, aspect_phrase, name):
    who = "guests" if dataset == "space" else "customers"
    if not aspect_phrase:
        return f"What do {who} say about {name}?"
    return f"What do {who} say about {aspect_phrase} at {name}?"


def rating_dist(ratings):
    counts = [0] * 5
    for r in ratings:
        counts[int(r) - 1] += 1
    return counts


def build_entity_pools(dataset, entity, aspects, embedder, pool_size, min_words, max_words):
    """entity: {entity_id, entity_name, reviews: [{review_id, rating, sentences}], summaries, ...}
    Sentences longer than max_words are mostly unsplit reviews (0.4% in SPACE); dense
    retrieval favours them for general queries, so they are dropped."""
    sents = []
    for review in entity["reviews"]:
        for j, text in enumerate(review["sentences"]):
            if min_words <= len(text.split()) <= max_words:
                sents.append({"review_id": review["review_id"], "sent_idx": j,
                              "rating": int(review["rating"]), "text": text})
    E = embedder.encode([s["text"] for s in sents])
    entity_dist = rating_dist(r["rating"] for r in entity["reviews"])

    items = []
    for aspect, (phrase, query) in aspects.items():
        scores = E @ embedder.encode_query(query)
        top = np.argsort(-scores, kind="stable")[:pool_size]
        docs = [{"rank": rank, "id": f"{s['review_id']}_{s['sent_idx']}", **s, "title": "",
                 "score": float(scores[i])}
                for rank, i in enumerate(top) for s in [sents[i]]]
        item = {
            "dataset": dataset,
            "sample_id": f"{entity['entity_id']}_{aspect}",
            "entity_id": entity["entity_id"],
            "entity_name": entity["entity_name"],
            "aspect": aspect,
            "question": make_question(dataset, phrase, entity["entity_name"]),
            "query": query,
            "summaries": entity.get("summaries", {}).get(aspect, []),
            "entity_rating_dist": entity_dist,
            "pool_rating_dist": rating_dist(d["rating"] for d in docs),
        }
        for key in ("latitude", "longitude"):
            if key in entity:
                item[key] = entity[key]
        item["docs"] = docs
        items.append(item)
    return items


def load_space(path=DEFAULT_SPACE_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_yelp(yelp_dir=DEFAULT_YELP_DIR, n_entities=100, city=None, min_reviews=100,
              max_reviews=500, reviews_per_entity=100, seed=0):
    """Sample `n_entities` restaurants and up to `reviews_per_entity` reviews each
    (100, as in SPACE). The review file is streamed once."""
    yelp_dir = Path(yelp_dir)
    candidates = []
    with open(yelp_dir / "yelp_academic_dataset_business.json", encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            if "Restaurants" not in (b.get("categories") or ""):
                continue
            if city and b.get("city", "").lower() != city.lower():
                continue
            if not min_reviews <= b.get("review_count", 0) <= max_reviews:
                continue
            candidates.append(b)
    candidates.sort(key=lambda b: b["business_id"])
    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(n_entities, len(candidates)))
    by_id = {b["business_id"]: b for b in chosen}

    reviews = {bid: [] for bid in by_id}
    with open(yelp_dir / "yelp_academic_dataset_review.json", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["business_id"] in reviews:
                reviews[r["business_id"]].append(r)

    entities = []
    for bid, b in by_id.items():
        revs = sorted(reviews[bid], key=lambda r: r["review_id"])
        revs = rng.sample(revs, min(reviews_per_entity, len(revs)))
        entities.append({
            "entity_id": bid,
            "entity_name": b["name"],
            "latitude": b["latitude"],
            "longitude": b["longitude"],
            "reviews": [{"review_id": r["review_id"], "rating": int(r["stars"]),
                         "sentences": split_sentences(r["text"])} for r in revs],
        })
    return entities


def default_review_pool_path(dataset, pool_size, pool_dir="pools"):
    return (Path(pool_dir) / f"{dataset}_k{pool_size}.json").as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["space", "yelp"])
    parser.add_argument("--space_file", default=DEFAULT_SPACE_FILE)
    parser.add_argument("--yelp_dir", default=DEFAULT_YELP_DIR)
    parser.add_argument("--n_entities", type=int, default=None,
                        help="number of entities (default: all 50 for SPACE, 100 for Yelp)")
    parser.add_argument("--city", default=None, help="Yelp: only restaurants in this city")
    parser.add_argument("--seed", type=int, default=0, help="Yelp: sampling seed")
    parser.add_argument("--min_reviews", type=int, default=100, help="Yelp: min review_count of a restaurant")
    parser.add_argument("--max_reviews", type=int, default=500, help="Yelp: max review_count of a restaurant")
    parser.add_argument("--pool_size", type=int, default=20, help="candidate sentences per query")
    parser.add_argument("--min_words", type=int, default=4, help="drop shorter sentences")
    parser.add_argument("--max_words", type=int, default=60, help="drop longer sentences (unsplit reviews)")
    parser.add_argument("--embed_model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=None, help="output path (default: pools/{dataset}_k{pool_size}.json)")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # Windows GBK console: don't crash on accents

    if args.dataset == "space":
        entities = load_space(args.space_file)[:args.n_entities]
        aspects = SPACE_ASPECTS
    else:
        entities = load_yelp(args.yelp_dir, args.n_entities or 100, args.city,
                             args.min_reviews, args.max_reviews, seed=args.seed)
        aspects = YELP_ASPECTS

    embedder = Embedder(args.embed_model)  # no disk cache: ~45k sentences would bloat it
    pool = []
    for i, entity in enumerate(entities):
        pool += build_entity_pools(args.dataset, entity, aspects, embedder, args.pool_size,
                                   args.min_words, args.max_words)
        print(f"[{i + 1}/{len(entities)}] {entity['entity_name']}: "
              f"{sum(len(r['sentences']) for r in entity['reviews'])} sentences", flush=True)
        embedder.cache.clear()

    out = args.out or default_review_pool_path(args.dataset, args.pool_size)
    save_json(pool, out)
    print(f"\nSaved {len(pool)} queries x {args.pool_size} sentences -> {out}")
    for item in pool[:2]:
        print(f"\n{item['question']}  (pool ratings 1..5 = {item['pool_rating_dist']})")
        for d in item["docs"][:3]:
            print(f"    rank={d['rank']} rating={d['rating']} score={d['score']:.3f} {d['text'][:90]}")


if __name__ == "__main__":
    main()
