import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = Path(
    "data/space/processed/space_retrieved_reviews.json"
)

OUTPUT_PATH = Path(
    "data/space/processed/space_mmr_results.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CANDIDATE_K = 20
FINAL_K = 5
LAMBDA_PARAM = 0.5


# --------------------------------------------------
# SPACE MMR Selector
# --------------------------------------------------

class SpaceMMRSelector:
    """
    Select reviews using Maximal Marginal Relevance (MMR).

    MMR balances query relevance and diversity.

    lambda_param = 1.0: relevance-focused
    lambda_param = 0.0: diversity-focused
    """

    def __init__(self, model, lambda_param=0.5):

        if not 0 <= lambda_param <= 1:
            raise ValueError(
                "lambda_param must be between 0 and 1."
            )

        self.model = model
        self.lambda_param = lambda_param

    def select(self, query, candidates, final_k=FINAL_K):

        if final_k < 1:
            raise ValueError("final_k must be positive.")

        if not candidates:
            return []

        # Encode query and candidate reviews using the same model.
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        documents = [review["text"] for review in candidates]

        document_embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Query-to-review relevance.
        relevance_scores = cosine_similarity(
            query_embedding.reshape(1, -1),
            document_embeddings,
        )[0]

        # Review-to-review similarity.
        similarity_matrix = cosine_similarity(
            document_embeddings
        )

        final_k = min(final_k, len(candidates))

        selected_indices = []
        remaining_indices = list(range(len(candidates)))

        # Greedy MMR selection.
        while remaining_indices and len(selected_indices) < final_k:

            best_index = None
            best_score = -float("inf")

            for index in remaining_indices:

                if not selected_indices:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        similarity_matrix[index, selected_index]
                        for selected_index in selected_indices
                    )

                mmr_score = (
                    self.lambda_param * relevance_scores[index]
                    - (1 - self.lambda_param) * redundancy
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = index

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

        selected_reviews = []

        for rank, index in enumerate(selected_indices, start=1):

            review = candidates[index]

            selected_reviews.append({
                "rank": rank,
                "review_id": review["review_id"],
                "hotel_id": review["hotel_id"],
                "hotel_name": review["hotel_name"],
                "text": review["text"],
                "rating": review["rating"],
                "sentiment": review["sentiment"],
                "distance": review["distance"],
                "relevance_score": float(relevance_scores[index]),
                "mmr_score": float(
                    self.lambda_param * relevance_scores[index]
                    - (
                        (1 - self.lambda_param)
                        * max(
                            (
                                similarity_matrix[index, selected_index]
                                for selected_index in selected_indices[:rank - 1]
                            ),
                            default=0.0,
                        )
                    )
                ),
            })

        return selected_reviews


# --------------------------------------------------
# Run MMR on All SPACE Hotel-Aspect Cases
# --------------------------------------------------

def run_mmr():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Retrieval results not found: {INPUT_PATH}\n"
            "Run SPACE retrieval first."
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        retrieval_results = json.load(file)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    selector = SpaceMMRSelector(
        model=model,
        lambda_param=LAMBDA_PARAM,
    )

    all_results = []

    for index, case in enumerate(retrieval_results, start=1):

        selected_reviews = selector.select(
            query=case["query"],
            candidates=case["reviews"],
            final_k=FINAL_K,
        )

        all_results.append({
            "hotel_id": case["hotel_id"],
            "hotel_name": case["hotel_name"],
            "aspect": case["aspect"],
            "query": case["query"],
            "method": "mmr",
            "candidate_k": CANDIDATE_K,
            "final_k": FINAL_K,
            "lambda_param": LAMBDA_PARAM,
            "candidate_count": len(case["reviews"]),
            "selected_count": len(selected_reviews),
            "results": selected_reviews,
        })

        print(
            f"[{index}/{len(retrieval_results)}] "
            f"{case['hotel_name']} | {case['aspect']} | "
            f"Selected: {len(selected_reviews)}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSPACE MMR selection completed!")
    print(f"Hotel-aspect cases: {len(all_results)}")
    print(f"Lambda: {LAMBDA_PARAM}")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_mmr()
