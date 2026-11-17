
import json
from pathlib import Path

from retrieve_reviews import SpaceReviewRetriever, TOP_K

PREPARED_DATA_PATH = Path(
    "data/space/processed/space_prepared.json"
)

OUTPUT_PATH = Path(
    "data/space/processed/space_user_query_retrieved_reviews.json"
)


def main():

    # Ask the user to provide a query.
    user_query = input("\nEnter your hotel-related query: ").strip()

    if not user_query:
        raise ValueError("Query cannot be empty.")

    retriever = SpaceReviewRetriever()

    # Load all hotels.
    with PREPARED_DATA_PATH.open("r", encoding="utf-8") as file:
        hotels = json.load(file)

    all_results = []

    print(f"\nUser query: {user_query}")
    print(f"Total hotels: {len(hotels)}\n")

    for index, hotel in enumerate(hotels, start=1):

        hotel_id = str(hotel["entity_id"])
        hotel_name = hotel["entity_name"]

        reviews = retriever.retrieve(
            hotel_id=hotel_id,
            query=user_query,
            k=TOP_K,
        )

        all_results.append({
            "hotel_id": hotel_id,
            "hotel_name": hotel_name,
            "query_type": "user_query",
            "query": user_query,
            "candidate_k": TOP_K,
            "retrieved_count": len(reviews),
            "reviews": reviews,
        })

        print(
            f"[{index}/{len(hotels)}] "
            f"{hotel_name} | Retrieved: {len(reviews)}"
        )

    # Save results separately.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nUser-query retrieval completed!")
    print(f"Hotels processed: {len(hotels)}")
    print(f"Total cases: {len(all_results)}")
    print(f"Maximum reviews per hotel: {TOP_K}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
