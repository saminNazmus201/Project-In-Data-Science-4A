import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "asqa_wikipedia"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

QUESTIONS_PATH = "data/results/asqa_50_questions.json"
OUTPUT_PATH = "data/results/top_k_results.json"


class TopKRetriever:
    def __init__(self, db_path=DB_PATH, model_name=MODEL_NAME):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, k: int = 20):
        """
        Retrieve the top-k most relevant chunks for a query.
        Lower cosine distance means greater embedding similarity.
        """
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=[
                "documents",
                "metadatas",
                "distances",
                "embeddings",
            ],
        )

        retrieved = []

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append({
                "text": document,
                "title": metadata["title"],
                "url": metadata["url"],
                "chunk_id": metadata["chunk_id"],
                "distance": float(distance),
                "score": float(1 - distance),
            })

        return retrieved


def evaluate_top_k():
    """Run Top-K retrieval for all 50 selected questions."""

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    retriever = TopKRetriever()
    all_results = []

    for index, item in enumerate(questions, start=1):
        # Use the question field from the JSON file.
        question = item["question"]
        question_id = item.get("id", index)

        # Retrieve the top 20 candidates.
        candidates = retriever.retrieve(
            query=question,
            k=20,
        )

        # Select the 5 most relevant candidates.
        retrieved = candidates[:5]

        all_results.append({
            "id": question_id,
            "question": question,
            "method": "top_k",
            "candidate_k": 20,
            "results": retrieved,
        })

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nTop-K evaluation completed!")
    print(f"Questions processed: {len(all_results)}")
    print(f"Results saved to: {output_path}")


# --------------------------------------------------
# OLD: Single-question test
# Kept as comments for future reference
# --------------------------------------------------

# if __name__ == "__main__":
#     retriever = TopKRetriever()
#
#     question = "Who has scored the most goals in international football?"
#     results = retriever.retrieve(question, k=20)
#
#     print("Question:", question)
#     print("Retrieved candidates:", len(results))
#
#     for rank, result in enumerate(results[:5], start=1):
#         print(f"\n--- Result {rank} ---")
#         print("Title:", result["title"])
#         print("Chunk ID:", result["chunk_id"])
#         print("Cosine distance:", round(result["distance"], 4))
#         print("Similarity score:", round(result["score"], 4))
#         print("Text:", result["text"][:300].replace("\n", " "))


if __name__ == "__main__":
    evaluate_top_k()