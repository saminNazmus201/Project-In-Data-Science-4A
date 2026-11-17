
import json
from pathlib import Path

FILES = {
    "Top-K": Path(
        "data/space/processed/space_user_query_top_k_results.json"
    ),
    "MMR": Path(
        "data/space/processed/space_user_query_mmr_results.json"
    ),
    "Proportional": Path(
        "data/space/processed/space_user_query_proportional_results.json"
    ),
}

for method, file_path in FILES.items():

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    case = data[0]

    # Proportional and Top-K/MMR use different
    # output field names in the existing scripts.
    reviews = case.get(
        "results",
        case.get("selected_reviews", [])
    )

    print("\n" + "=" * 90)
    print(f"{method.upper()} SELECTION")
    print("=" * 90)

    print(f"Hotel: {case['hotel_name']}")
    print(f"Query: {case['query']}")

    for review in reviews:

        print("\n" + "-" * 80)

        print(f"Rank: {review['rank']}")
        print(f"Review ID: {review['review_id']}")
        print(f"Rating: {review['rating']}")
        print(f"Sentiment: {review['sentiment']}")
        print(
            f"Relevance Score: "
            f"{review['relevance_score']:.4f}"
        )

        if "mmr_score" in review:
            print(f"MMR Score: {review['mmr_score']:.4f}")

        if "hpf_score" in review:
            print(f"HPF Score: {review['hpf_score']:.4f}")

        print("\nReview Text:")
        print(review["text"])
