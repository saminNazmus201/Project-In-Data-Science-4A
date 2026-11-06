import json
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = Path(
    "data/space/processed/space_retrieved_reviews.json"
)

OUTPUT_PATH = Path(
    "data/space/processed/space_top_k_results.json"
)

CANDIDATE_K = 20
FINAL_K = 5


# --------------------------------------------------
# Top-K Selection
# --------------------------------------------------

class SpaceTopKSelector:
    """
    Select the top-k most relevant reviews.

    Candidates are ranked by their relevance scores
    in descending order.
    """

    def select(self, candidates, final_k=FINAL_K):

        if final_k < 1:
            raise ValueError("final_k must be positive.")

        if not candidates:
            return []

        # Sort reviews by relevance score: highest first.
        ranked_candidates = sorted(
            candidates,
            key=lambda review: review["relevance_score"],
            reverse=True,
        )

        # Select the top-k reviews.
        selected = ranked_candidates[:final_k]

        results = []

        for rank, review in enumerate(selected, start=1):

            results.append({
                "rank": rank,
                "review_id": review["review_id"],
                "hotel_id": review["hotel_id"],
                "hotel_name": review["hotel_name"],
                "text": review["text"],
                "rating": review["rating"],
                "sentiment": review["sentiment"],
                "distance": review["distance"],
                "relevance_score": review["relevance_score"],
            })

        return results


# --------------------------------------------------
# Run Top-K on All SPACE Hotel-Aspect Cases
# --------------------------------------------------

def run_top_k():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Retrieval results not found: {INPUT_PATH}\n"
            "Run SPACE retrieval first."
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        retrieval_results = json.load(file)

    selector = SpaceTopKSelector()
    all_results = []

    for index, case in enumerate(retrieval_results, start=1):

        candidates = case["reviews"]

        selected_reviews = selector.select(
            candidates=candidates,
            final_k=FINAL_K,
        )

        all_results.append({
            "hotel_id": case["hotel_id"],
            "hotel_name": case["hotel_name"],
            "aspect": case["aspect"],
            "query": case["query"],
            "method": "top_k",
            "candidate_k": CANDIDATE_K,
            "final_k": FINAL_K,
            "candidate_count": len(candidates),
            "selected_count": len(selected_reviews),
            "results": selected_reviews,
        })

        print(
            f"[{index}/{len(retrieval_results)}] "
            f"{case['hotel_name']} | {case['aspect']} | "
            f"Selected: {len(selected_reviews)}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSPACE Top-K selection completed!")
    print(f"Hotel-aspect cases: {len(all_results)}")
    print(f"Final reviews per case: {FINAL_K}")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_top_k()