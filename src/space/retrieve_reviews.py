import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DB_PATH = Path("data/space/processed/space_chroma_db")
OUTPUT_PATH = Path("data/space/processed/space_retrieved_reviews.json")

COLLECTION_NAME = "space_hotel_reviews"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 20

# Fixed queries for the final experiment
ASPECT_QUERIES = {
    "cleanliness": "hotel room cleanliness and bathroom hygiene",
    "service": "hotel staff service friendliness and responsiveness",
}


# --------------------------------------------------
# SPACE Review Retriever
# --------------------------------------------------

class SpaceReviewRetriever:

    def __init__(self):

        if not DB_PATH.exists():
            raise FileNotFoundError(
                f"SPACE vector database not found: {DB_PATH}"
            )

        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(DB_PATH),
                anonymized_telemetry=False,
            )
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        self.model = SentenceTransformer(MODEL_NAME)

    def retrieve(self, hotel_id, query, k=TOP_K):

        hotel_id = str(hotel_id)

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        # Check that this hotel exists in the collection.
        hotel_records = self.collection.get(
            where={"hotel_id": hotel_id},
            include=["metadatas"],
        )

        if not hotel_records["ids"]:
            raise ValueError(
                f"No reviews found for hotel ID: {hotel_id}"
            )

        hotel_review_count = len(hotel_records["ids"])
        k = min(k, hotel_review_count)

        # Embed the fixed query using the same embedding model.
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Retrieve reviews belonging to this hotel only.
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where={"hotel_id": hotel_id},
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved_reviews = []

        for index, review_id in enumerate(results["ids"][0]):

            metadata = results["metadatas"][0][index]
            distance = float(results["distances"][0][index])

            retrieved_reviews.append({
                "review_id": metadata["review_id"],
                "hotel_id": metadata["hotel_id"],
                "hotel_name": metadata["hotel_name"],
                "text": results["documents"][0][index],
                "rating": metadata["rating"],
                "sentiment": metadata["sentiment"],
                "distance": distance,
                "relevance_score": 1.0 - distance,
            })

        return retrieved_reviews


# --------------------------------------------------
# Run Retrieval for All 50 Hotels and Both Aspects
# --------------------------------------------------

def main():

    retriever = SpaceReviewRetriever()

    # Read the prepared dataset to get the complete hotel list.
    prepared_data_path = Path(
        "data/space/processed/space_prepared.json"
    )

    if not prepared_data_path.exists():
        raise FileNotFoundError(
            f"Prepared SPACE dataset not found: "
            f"{prepared_data_path}"
        )

    with prepared_data_path.open("r", encoding="utf-8") as file:
        hotels = json.load(file)

    all_results = []

    total_cases = len(hotels) * len(ASPECT_QUERIES)
    completed_cases = 0

    for hotel in hotels:

        hotel_id = str(hotel["entity_id"])
        hotel_name = hotel["entity_name"]

        for aspect, query in ASPECT_QUERIES.items():

            reviews = retriever.retrieve(
                hotel_id=hotel_id,
                query=query,
                k=TOP_K,
            )

            all_results.append({
                "hotel_id": hotel_id,
                "hotel_name": hotel_name,
                "aspect": aspect,
                "query": query,
                "candidate_k": TOP_K,
                "retrieved_count": len(reviews),
                "reviews": reviews,
            })

            completed_cases += 1

            print(
                f"[{completed_cases}/{total_cases}] "
                f"{hotel_name} | {aspect} | "
                f"Retrieved: {len(reviews)}"
            )

    # Save all retrieval results for the selection experiments.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSPACE retrieval completed!")
    print(f"Hotels processed: {len(hotels)}")
    print(f"Fixed queries per hotel: {len(ASPECT_QUERIES)}")
    print(f"Total hotel-aspect cases: {len(all_results)}")
    print(f"Maximum reviews per case: {TOP_K}")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()