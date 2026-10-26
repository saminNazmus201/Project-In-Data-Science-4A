import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.top_k import TopKRetriever


class MMRSelector:
    def __init__(self, retriever=None, lambda_param=0.5):
        self.retriever = retriever or TopKRetriever()
        self.lambda_param = lambda_param

    def select(self, query: str, candidate_k: int = 20, final_k: int = 5):
        """
        Retrieve candidate chunks and select final_k chunks using MMR.

        lambda_param balances relevance and diversity:
        1.0 = relevance-focused
        0.0 = diversity-focused
        """
        if not 0 <= self.lambda_param <= 1:
            raise ValueError("lambda_param must be between 0 and 1.")

        if final_k < 1 or candidate_k < 1:
            raise ValueError("candidate_k and final_k must be positive.")

        query_embedding = self.retriever.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        results = self.retriever.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=candidate_k,
            include=[
                "documents",
                "metadatas",
                "distances",
                "embeddings",
            ],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        embeddings = np.asarray(results["embeddings"][0])

        if len(documents) == 0:
            return []

        final_k = min(final_k, len(documents))

        # Query-to-document cosine similarity.
        relevance_scores = cosine_similarity(
            query_embedding.reshape(1, -1),
            embeddings,
        )[0]

        # Document-to-document cosine similarity.
        document_similarities = cosine_similarity(embeddings)

        selected_indices = []
        remaining_indices = list(range(len(documents)))

        while remaining_indices and len(selected_indices) < final_k:
            best_index = None
            best_score = -float("inf")

            for idx in remaining_indices:
                if not selected_indices:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        document_similarities[idx][selected_idx]
                        for selected_idx in selected_indices
                    )

                mmr_score = (
                    self.lambda_param * relevance_scores[idx]
                    - (1 - self.lambda_param) * redundancy
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = idx

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

        selected = []

        for idx in selected_indices:
            metadata = metadatas[idx]
            distance = float(distances[idx])

            selected.append({
                "text": documents[idx],
                "title": metadata["title"],
                "url": metadata["url"],
                "chunk_id": metadata["chunk_id"],
                "distance": distance,
                "score": float(1 - distance),
                "mmr_relevance": float(relevance_scores[idx]),
            })

        return selected


def evaluate_mmr():
    """
    Run MMR selection for all questions in the selected ASQA dataset
    and save the results to a JSON file.
    """

    questions_path = Path("data/results/asqa_50_questions.json")
    output_path = Path("data/results/mmr_results.json")

    # Load the 50 selected questions.
    with open(questions_path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    # Initialize the retriever/model once and reuse it.
    retriever = TopKRetriever()

    selector = MMRSelector(
        retriever=retriever,
        lambda_param=0.5,
    )

    all_results = []

    for index, item in enumerate(questions, start=1):

        # Support either question key used in the JSON file.
        question = item.get(
            "question",
            item.get("ambiguous_question"),
        )

        question_id = item.get("id", index)

        if not question:
            raise ValueError(
                f"Question missing in item {index}: {item}"
            )

        # Retrieve 20 candidates and select the final 5.
        selected = selector.select(
            query=question,
            candidate_k=20,
            final_k=5,
        )

        all_results.append({
            "id": question_id,
            "question": question,
            "method": "mmr",
            "candidate_k": 20,
            "final_k": 5,
            "lambda_param": 0.5,
            "results": selected,
        })

        print(
            f"[{index}/{len(questions)}] "
            f"Processed question ID: {question_id}; "
            f"selected chunks: {len(selected)}"
        )

    # Create the output directory if it does not exist.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save all results.
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nMMR evaluation completed!")
    print(f"Questions processed: {len(all_results)}")
    print(f"Results saved to: {output_path}")


# --------------------------------------------------
# OLD: Single-question test
# Kept for future reference
# --------------------------------------------------

# if __name__ == "__main__":
#     selector = MMRSelector(lambda_param=0.5)
#
#     question = "Who has scored the most goals in international football?"
#
#     results = selector.select(
#         question,
#         candidate_k=20,
#         final_k=5,
#     )
#
#     print("Question:", question)
#     print("MMR selected chunks:", len(results))
#
#     for rank, result in enumerate(results, start=1):
#         print(f"\n--- MMR Result {rank} ---")
#         print("Title:", result["title"])
#         print("Chunk ID:", result["chunk_id"])
#         print("Cosine distance:", round(result["distance"], 4))
#         print("Text:", result["text"][:300].replace("\n", " "))


if __name__ == "__main__":
    evaluate_mmr()