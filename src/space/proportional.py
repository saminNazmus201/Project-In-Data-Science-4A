import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = Path("data/space/processed/space_retrieved_reviews.json")

OUTPUT_PATH = Path(
    "data/space/processed/space_proportional_results.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CANDIDATE_K = 20
FINAL_K = 5
LAMBDA_PARAM = 0.5


# --------------------------------------------------
# Proportional Selector
# --------------------------------------------------

class SpaceProportionalSelector:

    def __init__(
        self,
        model_name=MODEL_NAME,
        lambda_param=LAMBDA_PARAM,
    ):
        self.model = SentenceTransformer(model_name)
        self.lambda_param = lambda_param

    def select(self, query, candidates, final_k=FINAL_K):

        if not candidates:
            return []

        # Use the same candidate relevance scores
        # produced by the SPACE retrieval pipeline.
        relevance_scores = np.array(
            [
                float(candidate["relevance_score"])
                for candidate in candidates
            ],
            dtype=float,
        )

        # Encode candidate reviews to calculate
        # candidate-to-candidate cosine similarity.
        texts = [
            candidate["text"]
            for candidate in candidates
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        similarity_matrix = cosine_similarity(embeddings)

        # Remove self-similarity from the matrix.
        np.fill_diagonal(similarity_matrix, 0.0)

        K = len(candidates)
        k = min(final_k, K)

        # p_S: total similarity of each candidate
        # to all other candidates.
        p_S = np.sum(similarity_matrix, axis=1)

        selected_indices = []
        remaining_indices = list(range(K))
        selected_results = []

        # --------------------------------------------------
        # Greedy HPF selection
        # --------------------------------------------------

        for rank in range(1, k + 1):

            best_index = None
            best_score = -float("inf")
            best_details = None

            for i in remaining_indices:

                # p_R: similarity of candidate i
                # to reviews already selected.
                if selected_indices:
                    p_R = float(
                        np.sum(
                            similarity_matrix[
                                i, selected_indices
                            ]
                        )
                    )
                else:
                    p_R = 0.0

                # p_T: similarity to candidates
                # not yet selected, excluding self.
                p_T = float(p_S[i] - p_R)

                # Hypergraph Proportionality Function.
                hpf_score = (
                    (1 - self.lambda_param)
                    * (K - k)
                    * relevance_scores[i]
                    + self.lambda_param
                    * (p_S[i] - 2 * p_R)
                )

                if hpf_score > best_score:
                    best_score = float(hpf_score)
                    best_index = i

                    best_details = {
                        "p_S": float(p_S[i]),
                        "p_R": p_R,
                        "p_T": p_T,
                        "hpf_score": float(hpf_score),
                    }

            if best_index is None:
                break

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

            candidate = candidates[best_index]

            selected_review = {
                "rank": rank,
                "review_id": candidate["review_id"],
                "hotel_id": candidate["hotel_id"],
                "hotel_name": candidate["hotel_name"],
                "text": candidate["text"],
                "rating": candidate["rating"],
                "sentiment": candidate["sentiment"],
                "distance": candidate["distance"],
                "relevance_score": float(
                    relevance_scores[best_index]
                ),
                **best_details,
            }

            selected_results.append(selected_review)

        return selected_results


# --------------------------------------------------
# Run selection for all hotel-aspect cases
# --------------------------------------------------

def run_proportional():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    selector = SpaceProportionalSelector()

    all_results = []

    for case_number, case in enumerate(cases, start=1):

        candidates = case.get("reviews", [])

        selected_reviews = selector.select(
            query=case["query"],
            candidates=candidates,
            final_k=FINAL_K,
        )

        result = {
            "hotel_id": case["hotel_id"],
            "hotel_name": case["hotel_name"],
            "aspect": case["aspect"],
            "query": case["query"],
            "candidate_k": case.get(
                "candidate_k", CANDIDATE_K
            ),
            "retrieved_count": len(candidates),
            "selected_count": len(selected_reviews),
            "method": "Proportional",
            "lambda_param": LAMBDA_PARAM,
            "selected_reviews": selected_reviews,
        }

        all_results.append(result)

        print(
            f"[{case_number}/{len(cases)}] "
            f"{case['hotel_name']} | "
            f"{case['aspect']} | "
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

    print("\nSPACE Proportional selection completed!")
    print(f"Hotel-aspect cases: {len(all_results)}")
    print(f"Final reviews per case: {FINAL_K}")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_proportional()