
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from top_k import SpaceTopKSelector
from mmr import SpaceMMRSelector
from proportional import SpaceProportionalSelector


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = Path(
    "data/space/processed/space_user_query_results.json"
)

OUTPUT_DIR = Path(
    "data/space/processed"
)

TOP_K_OUTPUT = OUTPUT_DIR / "space_user_query_top_k_results.json"

MMR_OUTPUT = OUTPUT_DIR / "space_user_query_mmr_results.json"

PROPORTIONAL_OUTPUT = (
    OUTPUT_DIR / "space_user_query_proportional_results.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

FINAL_K = 5

LAMBDA_PARAM = 0.5


# ============================================================
# Load User Query Retrieval Results
# ============================================================

def load_retrieval_results():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Run user-query retrieval first."
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        retrieval_results = json.load(file)

    if not retrieval_results:
        raise ValueError("No retrieval results found.")

    return retrieval_results


# ============================================================
# Save Selection Results
# ============================================================

def save_results(results, output_path):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open("w", encoding="utf-8") as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved: {output_path}")


# ============================================================
# Main Selection Experiment
# ============================================================

def run_user_query_selection():

    print("\n====================================")
    print("   USER QUERY SELECTION EXPERIMENT")
    print("====================================")

    # Load retrieved candidates
    retrieval_results = load_retrieval_results()

    if len(retrieval_results) != 1:
        raise ValueError(
            "Expected one user-query retrieval case. "
            f"Found {len(retrieval_results)}."
        )

    case = retrieval_results[0]

    candidates = case["reviews"]

    query = case["query"]

    hotel_name = case["hotel_name"]

    if not candidates:
        raise ValueError("No candidate reviews available.")

    final_k = min(FINAL_K, len(candidates))

    print(f"\nHotel: {hotel_name}")
    print(f"Query: {query}")
    print(f"Candidate reviews: {len(candidates)}")
    print(f"Final reviews to select: {final_k}")

    # --------------------------------------------------------
    # Initialize selectors
    # --------------------------------------------------------

    print("\nInitializing selection algorithms...")

    # Load the embedding model once.
    # Reuse it for both MMR and Proportional.
    proportional_selector = SpaceProportionalSelector(
        model_name=MODEL_NAME,
        lambda_param=LAMBDA_PARAM
    )

    model = proportional_selector.model

    top_k_selector = SpaceTopKSelector()

    mmr_selector = SpaceMMRSelector(
        model=model,
        lambda_param=LAMBDA_PARAM
    )

    print("All selectors initialized.")

    # --------------------------------------------------------
    # 1. Top-K Selection
    # --------------------------------------------------------

    print("\nRunning Top-K selection...")

    top_k_reviews = top_k_selector.select(
        candidates=candidates,
        final_k=final_k
    )

    top_k_result = {

        "hotel_id": case["hotel_id"],
        "hotel_name": hotel_name,
        "query": query,
        "method": "top_k",

        "candidate_k": len(candidates),
        "final_k": final_k,

        "candidate_count": len(candidates),
        "selected_count": len(top_k_reviews),

        "results": top_k_reviews
    }

    save_results(
        [top_k_result],
        TOP_K_OUTPUT
    )

    # --------------------------------------------------------
    # 2. MMR Selection
    # --------------------------------------------------------

    print("\nRunning MMR selection...")

    mmr_reviews = mmr_selector.select(
        query=query,
        candidates=candidates,
        final_k=final_k
    )

    mmr_result = {

        "hotel_id": case["hotel_id"],
        "hotel_name": hotel_name,
        "query": query,
        "method": "mmr",

        "candidate_k": len(candidates),
        "final_k": final_k,

        "lambda_param": LAMBDA_PARAM,

        "candidate_count": len(candidates),
        "selected_count": len(mmr_reviews),

        "results": mmr_reviews
    }

    save_results(
        [mmr_result],
        MMR_OUTPUT
    )

    # --------------------------------------------------------
    # 3. Proportional Selection
    # --------------------------------------------------------

    print("\nRunning Proportional selection...")

    proportional_reviews = proportional_selector.select(
        query=query,
        candidates=candidates,
        final_k=final_k
    )

    proportional_result = {

        "hotel_id": case["hotel_id"],
        "hotel_name": hotel_name,
        "query": query,
        "method": "proportional",

        "candidate_k": len(candidates),
        "final_k": final_k,

        "lambda_param": LAMBDA_PARAM,

        "candidate_count": len(candidates),
        "selected_count": len(proportional_reviews),

        "results": proportional_reviews
    }

    save_results(
        [proportional_result],
        PROPORTIONAL_OUTPUT
    )

    # --------------------------------------------------------
    # Experiment Summary
    # --------------------------------------------------------

    print("\n====================================")
    print("       SELECTION COMPLETED")
    print("====================================")

    print(f"\nHotel: {hotel_name}")

    print(f"Query: {query}")

    print(f"Candidate reviews: {len(candidates)}")

    print("\nSelected reviews:")

    print(f"Top-K: {len(top_k_reviews)}")

    print(f"MMR: {len(mmr_reviews)}")

    print(f"Proportional: {len(proportional_reviews)}")

    print("\nOutput files:")

    print(TOP_K_OUTPUT)

    print(MMR_OUTPUT)

    print(PROPORTIONAL_OUTPUT)

    print("\nUser-query selection experiment completed!")


# ============================================================
# Run Experiment
# ============================================================

if __name__ == "__main__":

    run_user_query_selection()
