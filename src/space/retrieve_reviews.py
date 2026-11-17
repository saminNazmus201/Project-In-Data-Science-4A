
import json
import re
import difflib
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

DB_PATH = Path("data/space/processed/space_chroma_db")

# Original aspect-based experiment output
OUTPUT_PATH = Path(
    "data/space/processed/space_retrieved_reviews.json"
)

# New user-query experiment output
USER_QUERY_OUTPUT_PATH = Path(
    "data/space/processed/space_user_query_results.json"
)

COLLECTION_NAME = "space_hotel_reviews"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 20

# Original predefined aspect queries
ASPECT_QUERIES = {
    "cleanliness": "hotel room cleanliness and bathroom hygiene",
    "service": "hotel staff service friendliness and responsiveness",
}


# ============================================================
# SPACE Review Retriever
# ============================================================

class SpaceReviewRetriever:

    def __init__(self):

        if not DB_PATH.exists():
            raise FileNotFoundError(
                f"ChromaDB not found: {DB_PATH}"
            )

        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(DB_PATH),
                anonymized_telemetry=False
            )
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        self.model = SentenceTransformer(MODEL_NAME)

        print("\nSPACE Review Retriever initialized.")

    def retrieve(self, hotel_id, query, k=TOP_K):

        hotel_id = str(hotel_id)

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # Check the number of reviews for this hotel
        hotel_records = self.collection.get(
            where={"hotel_id": hotel_id},
            include=["metadatas"]
        )

        if not hotel_records["ids"]:
            raise ValueError(
                f"No reviews found for hotel ID: {hotel_id}"
            )

        hotel_review_count = len(hotel_records["ids"])

        k = min(k, hotel_review_count)

        # Encode the query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Retrieve reviews belonging to the selected hotel
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where={"hotel_id": hotel_id},
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved_reviews = []

        for index, review_id in enumerate(results["ids"][0]):

            metadata = results["metadatas"][0][index]

            distance = float(
                results["distances"][0][index]
            )

            retrieved_reviews.append({

                "review_id": metadata["review_id"],

                "hotel_id": metadata["hotel_id"],

                "hotel_name": metadata["hotel_name"],

                "text": results["documents"][0][index],

                "rating": metadata["rating"],

                "sentiment": metadata["sentiment"],

                "distance": distance,

                "relevance_score": 1.0 - distance

            })

        return retrieved_reviews


# ============================================================
# Load Prepared Hotel Dataset
# ============================================================

def load_hotels():

    prepared_data_path = Path(
        "data/space/processed/space_prepared.json"
    )

    if not prepared_data_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {prepared_data_path}"
        )

    with prepared_data_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        hotels = json.load(file)

    return hotels


# ============================================================
# Hotel Name Matching
# ============================================================

def normalize_name(name):

    # Ignore case and punctuation when matching hotel names
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        name.lower()
    ).strip()


def find_hotel(hotels, hotel_name):

    search_name = normalize_name(hotel_name)

    hotel_names = [
        hotel["entity_name"] for hotel in hotels
    ]

    # First, try exact matching
    for hotel in hotels:

        if normalize_name(hotel["entity_name"]) == search_name:
            return hotel

    # Next, try partial matching
    matches = [

        hotel for hotel in hotels

        if search_name in normalize_name(hotel["entity_name"])

        or normalize_name(hotel["entity_name"]) in search_name

    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:

        print("\nMultiple matching hotels found:")

        for index, hotel in enumerate(matches, start=1):
            print(f"{index}. {hotel['entity_name']}")

        while True:

            try:

                choice = int(
                    input("\nSelect hotel number: ")
                )

                if 1 <= choice <= len(matches):
                    return matches[choice - 1]

                print("Invalid selection. Try again.")

            except ValueError:
                print("Please enter a valid number.")

    # If no match, suggest similar hotel names
    normalized_names = {
        normalize_name(name): name for name in hotel_names
    }

    suggestions = difflib.get_close_matches(
        search_name,
        list(normalized_names.keys()),
        n=5,
        cutoff=0.3
    )

    print("\nNo exact matching hotel found.")

    if suggestions:

        print("\nDid you mean one of these hotels?")

        for index, suggestion in enumerate(suggestions, start=1):
            print(
                f"{index}. {normalized_names[suggestion]}"
            )

    else:

        print("\nAvailable hotels in the dataset:")

        for index, name in enumerate(hotel_names, start=1):
            print(f"{index}. {name}")

    return None


# ============================================================
# Save Results
# ============================================================

def save_results(results, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nResults saved to: {output_path}")


# ============================================================
# Experiment 1: Original Aspect-Based Retrieval
# ============================================================

def run_batch_retrieval(retriever, hotels):

    all_results = []

    total_cases = len(hotels) * len(ASPECT_QUERIES)

    completed_cases = 0

    print("\nStarting aspect-based retrieval...")

    for hotel in hotels:

        hotel_id = str(hotel["entity_id"])

        hotel_name = hotel["entity_name"]

        for aspect, query in ASPECT_QUERIES.items():

            reviews = retriever.retrieve(
                hotel_id=hotel_id,
                query=query,
                k=TOP_K
            )

            all_results.append({

                "hotel_id": hotel_id,

                "hotel_name": hotel_name,

                "aspect": aspect,

                "query": query,

                "candidate_k": TOP_K,

                "retrieved_count": len(reviews),

                "reviews": reviews

            })

            completed_cases += 1

            print(
                f"[{completed_cases}/{total_cases}] "
                f"{hotel_name} | {aspect} | "
                f"Retrieved: {len(reviews)}"
            )

    # Save ONLY to the original output file
    save_results(
        all_results,
        OUTPUT_PATH
    )

    print("\nAspect-based retrieval completed!")

    print(f"Hotels processed: {len(hotels)}")

    print(f"Fixed queries per hotel: {len(ASPECT_QUERIES)}")

    print(f"Total hotel-aspect cases: {len(all_results)}")

    print(f"Maximum reviews per case: {TOP_K}")


# ============================================================
# Experiment 2: User-Query-Based Retrieval
# ============================================================

def run_user_query(retriever, hotels):

    print("\n====================================")
    print("       USER QUERY RETRIEVAL")
    print("====================================")

    # Ask the user for a hotel name
    hotel_name = input(
        "\nEnter hotel name: "
    ).strip()

    if not hotel_name:

        print("Hotel name cannot be empty.")

        return

    # Find the hotel in the dataset
    hotel = find_hotel(
        hotels,
        hotel_name
    )

    if hotel is None:

        print("\nPlease try again with a hotel name from the dataset.")

        return

    hotel_id = str(hotel["entity_id"])

    actual_hotel_name = hotel["entity_name"]

    print(f"\nSelected hotel: {actual_hotel_name}")

    # Ask the user for a custom query
    query = input(
        "\nEnter your search query: "
    ).strip()

    if not query:

        print("Query cannot be empty.")

        return

    print("\nRetrieving relevant reviews...")

    # Retrieve top 20 reviews for this hotel and query
    reviews = retriever.retrieve(
        hotel_id=hotel_id,
        query=query,
        k=TOP_K
    )

    result = {

        "hotel_id": hotel_id,

        "hotel_name": actual_hotel_name,

        "aspect": "user_query",

        "query": query,

        "candidate_k": TOP_K,

        "retrieved_count": len(reviews),

        "reviews": reviews

    }

    # Save ONLY to the new user-query output file
    save_results(
        [result],
        USER_QUERY_OUTPUT_PATH
    )

    print("\nUser-query retrieval completed!")

    print(f"Hotel: {actual_hotel_name}")

    print(f"Query: {query}")

    print(f"Retrieved reviews: {len(reviews)}")


# ============================================================
# Main Function
# ============================================================

def main():

    print("\n====================================")
    print("       SPACE REVIEW RETRIEVAL")
    print("====================================")

    print("\n1. Original aspect-based retrieval")

    print("2. User-provided hotel and query")

    choice = input(
        "\nChoose an experiment (1 or 2): "
    ).strip()

    if choice not in ["1", "2"]:

        print("\nInvalid option. Choose 1 or 2.")

        return

    # Initialize retriever and load dataset
    retriever = SpaceReviewRetriever()

    hotels = load_hotels()

    if choice == "1":

        run_batch_retrieval(
            retriever,
            hotels
        )

    elif choice == "2":

        run_user_query(
            retriever,
            hotels
        )


# ============================================================
# Run Script
# ============================================================

if __name__ == "__main__":

    main()
