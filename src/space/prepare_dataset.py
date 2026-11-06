import json
from pathlib import Path


# Input and output paths
INPUT_PATH = Path("data/space/space_summ.json")
OUTPUT_PATH = Path("data/space/processed/space_prepared.json")


def prepare_dataset():
    """Prepare hotel reviews and reference summaries for experiments."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"SPACE dataset not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        hotels = json.load(file)

    prepared_hotels = []

    for hotel in hotels:
        reviews = []

        for review in hotel.get("reviews", []):
            # Combine the review's sentences into one text.
            review_text = " ".join(review.get("sentences", [])).strip()

            if not review_text:
                continue

            rating = review.get("rating")

            # Preserve the original rating and derive a rating-based label.
            if rating is None:
                sentiment = "unknown"
            elif rating >= 4:
                sentiment = "positive"
            elif rating == 3:
                sentiment = "neutral"
            else:
                sentiment = "negative"

            reviews.append({
                "review_id": review.get("review_id"),
                "text": review_text,
                "rating": rating,
                "sentiment": sentiment,
            })

        # Keep all reference summaries, including general and aspect summaries.
        summaries = hotel.get("summaries", {})

        prepared_hotels.append({
            "entity_id": hotel.get("entity_id"),
            "entity_name": hotel.get("entity_name"),
            "reviews": reviews,
            "summaries": summaries,
        })

    # Create the output directory if necessary.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            prepared_hotels,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # Print validation statistics.
    total_reviews = sum(
        len(hotel["reviews"]) for hotel in prepared_hotels
    )

    sentiment_counts = {
        "positive": 0,
        "neutral": 0,
        "negative": 0,
        "unknown": 0,
    }

    for hotel in prepared_hotels:
        for review in hotel["reviews"]:
            sentiment_counts[review["sentiment"]] += 1

    print("SPACE dataset preparation completed!")
    print(f"Hotels processed: {len(prepared_hotels)}")
    print(f"Reviews processed: {total_reviews}")
    print(f"Rating-based sentiment counts: {sentiment_counts}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    prepare_dataset()
