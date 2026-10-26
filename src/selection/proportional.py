
import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.top_k import TopKRetriever


class ProportionalSelector:
    def __init__(self, retriever=None, lambda_param=0.5):
        self.retriever = retriever or TopKRetriever()
        self.lambda_param = lambda_param

    def select(self, query: str, candidate_k: int = 20, final_k: int = 5):
        """
        Select final_k chunks using the professor-inspired
        contextual proportionality formula.
        """

        if not 0 <= self.lambda_param <= 1:
            raise ValueError("lambda_param must be between 0 and 1.")

        if candidate_k < 1 or final_k < 1:
            raise ValueError("candidate_k and final_k must be positive.")

        # 1. Encode the question.
        query_embedding = self.retriever.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # 2. Retrieve the candidate pool from ChromaDB.
        results = self.retriever.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        embeddings = np.asarray(results["embeddings"][0])

        if not documents:
            return []

        # Actual candidate count and requested final selection size.
        K = len(documents)
        k = min(final_k, K)

        # 3. Relevance: cosine similarity between query and each chunk.
        relevance_scores = cosine_similarity(
            query_embedding.reshape(1, -1),
            embeddings,
        )[0]

        # 4. Contextual similarity between candidate chunks.
        similarity_matrix = cosine_similarity(embeddings)

        selected_indices = []
        remaining_indices = list(range(K))
        selection_details = {}

        # 5. Greedy selection using the HPF formula.
        while remaining_indices and len(selected_indices) < k:
            current_scores = {}

            for i in remaining_indices:

                # Similarity to all other candidate chunks: p_S
                p_S = sum(
                    similarity_matrix[i, j]
                    for j in range(K)
                    if j != i
                )

                # Similarity to already-selected chunks: p_R
                p_R = sum(
                    similarity_matrix[i, j]
                    for j in selected_indices
                )

                # Contextual proportionality.
                p_T = p_S - p_R

                # HPF formula using p_S - 2*p_R.
                hpf_score = (
                    (1 - self.lambda_param)
                    * (K - k)
                    * relevance_scores[i]
                    + self.lambda_param * (p_S - 2 * p_R)
                )

                current_scores[i] = hpf_score

                selection_details[i] = {
                    "p_S": float(p_S),
                    "p_R": float(p_R),
                    "p_T": float(p_T),
                    "hpf_score": float(hpf_score),
                }

            # Select the remaining chunk with the highest HPF score.
            best_index = max(current_scores, key=current_scores.get)

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

        # 6. Return selected chunks with scores and metadata.
        selected = []

        for i in selected_indices:
            metadata = metadatas[i]
            details = selection_details[i]

            selected.append({
                "text": documents[i],
                "title": metadata["title"],
                "url": metadata["url"],
                "chunk_id": metadata["chunk_id"],
                "distance": float(distances[i]),
                "score": float(relevance_scores[i]),
                "p_S": details["p_S"],
                "p_R": details["p_R"],
                "p_T": details["p_T"],
                "hpf_score": details["hpf_score"],
            })

        return selected


# ============================================================
# ORIGINAL SINGLE-QUESTION TEST
# Kept here for future use; commented out, NOT deleted.
# ============================================================

# if __name__ == "__main__":
#     selector = ProportionalSelector(lambda_param=0.5)
#
#     question = "Who has scored the most goals in international football?"
#     results = selector.select(
#         query=question,
#         candidate_k=20,
#         final_k=5,
#     )
#
#     print("Question:", question)
#     print("Selected chunks:", len(results))
#
#     for rank, result in enumerate(results, start=1):
#         print(f"\n--- Proportional Result {rank} ---")
#         print("Title:", result["title"])
#         print("Chunk ID:", result["chunk_id"])
#         print("Relevance:", round(result["score"], 4))
#         print("p_S:", round(result["p_S"], 4))
#         print("p_R:", round(result["p_R"], 4))
#         print("p_T:", round(result["p_T"], 4))
#         print("HPF score:", round(result["hpf_score"], 4))
#         print("Text:", result["text"][:300].replace("\n", " "))


# ============================================================
# BATCH EVALUATION: 50 ASQA QUESTIONS
# ============================================================

def evaluate_proportional():
    questions_path = Path("data/results/asqa_50_questions.json")
    output_path = Path("data/results/proportional_2pr_results.json")

    if not questions_path.exists():
        raise FileNotFoundError(
            f"Questions file not found: {questions_path}"
        )

    # Load the 50 selected ASQA questions.
    with questions_path.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    selector = ProportionalSelector(lambda_param=0.5)

    all_results = []

    for index, item in enumerate(questions, start=1):
        # Support either question field used in the input JSON.
        question = item.get("question") or item.get("ambiguous_question")

        if not question:
            print(f"Skipping item {index}: no question field found.")
            continue

        question_id = item.get("id", index)

        # Apply proportional selection: 20 candidates -> 5 chunks.
        selected_chunks = selector.select(
            query=question,
            candidate_k=20,
            final_k=5,
        )

        all_results.append({
            "id": question_id,
            "question": question,
            "results": selected_chunks,
        })

        print(
            f"Processed {index}/{len(questions)} "
            f"| Selected chunks: {len(selected_chunks)}"
        )

    # Save the results for later comparison with Top-K and MMR.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nProportional evaluation completed!")
    print("Questions processed:", len(all_results))
    print("Results saved to:", output_path)


if __name__ == "__main__":
    evaluate_proportional()
