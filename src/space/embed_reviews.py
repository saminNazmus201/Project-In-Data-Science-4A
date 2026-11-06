import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# File paths
INPUT_PATH = Path("data/space/processed/space_prepared.json")
DB_PATH = Path("data/space/processed/space_chroma_db")

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB collection
COLLECTION_NAME = "space_hotel_reviews"

# Batch size
BATCH_SIZE = 100


def embed_reviews():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {INPUT_PATH}. "
            "Run prepare_dataset.py first."
        )

    # Load prepared hotel data
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        hotels = json.load(file)

    # Load embedding model
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Create persistent ChromaDB client (compatible with 0.3.23)
    DB_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(DB_PATH),
            anonymized_telemetry=False,
        )
    )

    # Recreate collection to avoid duplicate records on reruns
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except (ValueError, IndexError):
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_reviews = 0

    # Process all 50 hotels
    for hotel in hotels:
        hotel_id = str(hotel["entity_id"])
        hotel_name = hotel["entity_name"]
        reviews = hotel.get("reviews", [])

        for start in range(0, len(reviews), BATCH_SIZE):
            batch = reviews[start:start + BATCH_SIZE]

            valid_reviews = [
                review
                for review in batch
                if review.get("text", "").strip()
            ]

            if not valid_reviews:
                continue

            texts = [
                review["text"].strip()
                for review in valid_reviews
            ]

            # Generate normalized embeddings
            embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            ids = []
            metadatas = []

            for index, review in enumerate(valid_reviews):
                review_id = review.get("review_id")

                if review_id is None:
                    review_id = f"review_{start + index}"

                ids.append(f"{hotel_id}_{review_id}")

                metadatas.append({
                    "hotel_id": hotel_id,
                    "hotel_name": hotel_name,
                    "review_id": str(review_id),
                    "rating": (
                        float(review["rating"])
                        if review.get("rating") is not None
                        else -1.0
                    ),
                    "sentiment": review.get(
                        "sentiment", "unknown"
                    ),
                })

            # Store review texts, embeddings and metadata
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
            )

            total_reviews += len(texts)

        print(
            f"Processed hotel: {hotel_name} "
            f"({total_reviews} reviews stored so far)"
        )

    # Persist data to disk
    client.persist()

    print("\nSPACE vector store created successfully!")
    print(f"Hotels processed: {len(hotels)}")
    print(f"Reviews stored: {collection.count()}")
    print("Embedding dimensions: 384")
    print(f"Database path: {DB_PATH}")
    print(f"Collection name: {COLLECTION_NAME}")


if __name__ == "__main__":
    embed_reviews()
