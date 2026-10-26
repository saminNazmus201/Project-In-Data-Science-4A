import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path("data/results")
OUTPUT_FILE = RESULTS_DIR / "comparison_summary.csv"
QUESTION_RESULTS_FILE = RESULTS_DIR / "comparison_per_question.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

METHOD_FILES = {
    "Top-K": RESULTS_DIR / "top_k_results.json",
    "MMR": RESULTS_DIR / "mmr_results.json",
    "Proportional": RESULTS_DIR / "proportional_2pr_results.json",
}


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results(file_path):
    """Load retrieval results from a JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a list of question results in {file_path}"
        )

    return data


# ============================================================
# CALCULATE METRICS FOR ONE QUESTION
# ============================================================

def calculate_metrics(question, chunks, model):
    """
    Calculate:
    1. Mean relevance
    2. Mean pairwise redundancy
    3. Unique article count
    """

    if not chunks:
        return {
            "mean_relevance": np.nan,
            "mean_pairwise_redundancy": np.nan,
            "unique_articles": 0,
            "selected_chunks": 0,
        }

    texts = [chunk.get("text", "") for chunk in chunks]

    if any(not text.strip() for text in texts):
        raise ValueError(
            "An empty chunk was found. Check the JSON results."
        )

    # Encode the question and selected chunks using the same model.
    embeddings = model.encode(
        [question] + texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_embedding = embeddings[0:1]
    chunk_embeddings = embeddings[1:]

    # --------------------------------------------------------
    # 1. Mean relevance
    # --------------------------------------------------------

    relevance_scores = cosine_similarity(
        query_embedding,
        chunk_embeddings,
    )[0]

    mean_relevance = float(np.mean(relevance_scores))

    # --------------------------------------------------------
    # 2. Mean pairwise redundancy
    # --------------------------------------------------------

    if len(chunks) < 2:
        mean_redundancy = np.nan
    else:
        similarity_matrix = cosine_similarity(chunk_embeddings)

        # Extract the upper triangle without the diagonal.
        upper_triangle = similarity_matrix[
            np.triu_indices(len(chunks), k=1)
        ]

        mean_redundancy = float(np.mean(upper_triangle))

    # --------------------------------------------------------
    # 3. Unique article count
    # --------------------------------------------------------

    unique_titles = {
        chunk.get("title")
        for chunk in chunks
        if chunk.get("title")
    }

    return {
        "mean_relevance": mean_relevance,
        "mean_pairwise_redundancy": mean_redundancy,
        "unique_articles": len(unique_titles),
        "selected_chunks": len(chunks),
    }


# ============================================================
# MAIN COMPARISON
# ============================================================

def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    per_question_rows = []
    summary_rows = []

    # Load all three methods.
    all_results = {
        method: load_results(file_path)
        for method, file_path in METHOD_FILES.items()
    }

    # Check that all methods contain the same number of questions.
    question_counts = {
        method: len(results)
        for method, results in all_results.items()
    }

    print("\nQuestion counts:")
    for method, count in question_counts.items():
        print(f"{method}: {count}")

    if len(set(question_counts.values())) != 1:
        raise ValueError(
            "The methods have different question counts. "
            "Check the input JSON files before comparing."
        )

    # Check that question order and text match across methods.
    reference_results = all_results["Top-K"]

    for method, results in all_results.items():
        for index, (reference, current) in enumerate(
            zip(reference_results, results),
            start=1,
        ):
            if reference.get("question") != current.get("question"):
                raise ValueError(
                    f"Question mismatch at position {index} "
                    f"between Top-K and {method}. "
                    "Align the questions before comparison."
                )

    total_questions = len(reference_results)

    # Evaluate every method on every question.
    for method, results in all_results.items():
        print(f"\nEvaluating {method}...")

        for index, item in enumerate(results, start=1):
            question = item.get("question")

            if not question:
                raise ValueError(
                    f"Missing question text in {method}, item {index}."
                )

            chunks = item.get("results", [])

            metrics = calculate_metrics(
                question=question,
                chunks=chunks,
                model=model,
            )

            per_question_rows.append({
                "question_id": item.get("id", index),
                "question": question,
                "method": method,
                **metrics,
            })

            print(
                f"\r{method}: {index}/{total_questions}",
                end="",
                flush=True,
            )

        print()

    # --------------------------------------------------------
    # SAVE PER-QUESTION RESULTS
    # --------------------------------------------------------

    per_question_df = pd.DataFrame(per_question_rows)

    per_question_df.to_csv(
        QUESTION_RESULTS_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # AGGREGATE RESULTS ACROSS QUESTIONS
    # --------------------------------------------------------

    for method in METHOD_FILES:
        method_df = per_question_df[
            per_question_df["method"] == method
        ]

        summary_rows.append({
            "method": method,
            "questions_evaluated": len(method_df),
            "mean_relevance": method_df["mean_relevance"].mean(),
            "mean_pairwise_redundancy": (
                method_df["mean_pairwise_redundancy"].mean()
            ),
            "mean_unique_articles": method_df["unique_articles"].mean(),
            "mean_selected_chunks": method_df["selected_chunks"].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRIEVAL COMPARISON SUMMARY")
    print("=" * 70)

    print(summary_df.round(4).to_string(index=False))

    print("\nFiles saved:")
    print(f"Per-question results: {QUESTION_RESULTS_FILE}")
    print(f"Summary:              {OUTPUT_FILE}")

    print("\nComparison completed successfully!")


if __name__ == "__main__":
    main()
