import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.space.top_k import SpaceTopKSelector
from src.space.mmr import SpaceMMRSelector
from src.space.proportional import SpaceProportionalSelector


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = Path(
    "data/space/processed/space_retrieved_reviews.json"
)

OUTPUT_PATH = Path(
    "data/space/processed/one_hotel_comparison.json"
)

HOTEL_ID = "100597"
ASPECT = "cleanliness"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

FINAL_K = 5
LAMBDA_PARAM = 0.5


def print_results(method_name, results, score_key=None):

    print(f"\n\n########## {method_name} RESULTS ##########")

    for review in results:
        print(f"\nRank: {review['rank']}")
        print(f"Review ID: {review['review_id']}")
        print(f"Rating: {review['rating']}")
        print(f"Sentiment: {review['sentiment']}")
        print(
            f"Relevance: "
            f"{review['relevance_score']:.4f}"
        )

        if score_key and score_key in review:
            print(
                f"{score_key}: "
                f"{review[score_key]:.4f}"
            )

        print(f"Text: {review['text']}")


def main():

    # --------------------------------------------------
    # Load retrieved candidates
    # --------------------------------------------------

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        all_cases = json.load(file)

    case = next(
        (
            item for item in all_cases
            if str(item["hotel_id"]) == HOTEL_ID
            and item["aspect"] == ASPECT
        ),
        None,
    )

    if case is None:
        raise ValueError(
            f"No case found for hotel {HOTEL_ID}, "
            f"aspect {ASPECT}"
        )

    candidates = case["reviews"]

    print("=" * 80)
    print(f"Hotel: {case['hotel_name']}")
    print(f"Hotel ID: {HOTEL_ID}")
    print(f"Aspect: {ASPECT}")
    print(f"Query: {case['query']}")
    print(f"Candidates: {len(candidates)}")
    print(f"Final selection: {FINAL_K}")
    print("=" * 80)

    # --------------------------------------------------
    # 1. Top-K
    # --------------------------------------------------

    top_k_selector = SpaceTopKSelector()

    top_k_results = top_k_selector.select(
        candidates=candidates,
        final_k=FINAL_K,
    )

    print_results(
        method_name="TOP-K",
        results=top_k_results,
    )

    # --------------------------------------------------
    # 2. Load embedding model
    # --------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    # --------------------------------------------------
    # 3. MMR
    # --------------------------------------------------

    mmr_selector = SpaceMMRSelector(
        model=model,
        lambda_param=LAMBDA_PARAM,
    )

    mmr_results = mmr_selector.select(
        query=case["query"],
        candidates=candidates,
        final_k=FINAL_K,
    )

    print_results(
        method_name="MMR",
        results=mmr_results,
        score_key="mmr_score",
    )

    # --------------------------------------------------
    # 4. Proportional Selection
    # --------------------------------------------------

    proportional_selector = SpaceProportionalSelector(
        model_name=MODEL_NAME,
        lambda_param=LAMBDA_PARAM,
    )

    proportional_results = proportional_selector.select(
        query=case["query"],
        candidates=candidates,
        final_k=FINAL_K,
    )

    print_results(
        method_name="PROPORTIONAL",
        results=proportional_results,
        score_key="hpf_score",
    )

    # --------------------------------------------------
    # 5. Save comparison
    # --------------------------------------------------

    comparison = {
        "hotel_id": HOTEL_ID,
        "hotel_name": case["hotel_name"],
        "aspect": ASPECT,
        "query": case["query"],
        "candidate_count": len(candidates),
        "final_k": FINAL_K,
        "lambda_param": LAMBDA_PARAM,
        "top_k": top_k_results,
        "mmr": mmr_results,
        "proportional": proportional_results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            comparison,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("ONE-HOTEL COMPARISON COMPLETED!")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
