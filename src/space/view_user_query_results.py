
import json
from pathlib import Path

# JSON file path
FILE_PATH = Path(
    "data/space/processed/space_user_query_results.json"
)

# Load results
with FILE_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

# Display experiment information
for result in data:

    print("=" * 80)
    print("USER QUERY RETRIEVAL RESULTS")
    print("=" * 80)

    print(f"\nHotel: {result['hotel_name']}")
    print(f"Query: {result['query']}")
    print(f"Retrieved reviews: {result['retrieved_count']}")

    print("\n" + "=" * 80)
    print("RETRIEVED REVIEWS")
    print("=" * 80)

    for i, review in enumerate(result["reviews"], start=1):

        print(f"\n{'-' * 80}")
        print(f"Review {i}")
        print("-" * 80)

        print(f"Review ID: {review['review_id']}")
        print(f"Rating: {review['rating']}")
        print(f"Sentiment: {review['sentiment']}")
        print(f"Distance: {review['distance']:.4f}")
        print(f"Relevance Score: {review['relevance_score']:.4f}")

        print("\nReview Text:")
        print(review["text"])

    print("\n" + "=" * 80)
    print("END OF RESULTS")
    print("=" * 80)
