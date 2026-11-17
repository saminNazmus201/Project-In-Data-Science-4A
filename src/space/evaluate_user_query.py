
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "space" / "processed"

CANDIDATES_FILE = DATA_DIR / "space_user_query_results.json"

METHOD_FILES = {
    "Top-K": DATA_DIR / "space_user_query_top_k_results.json",
    "MMR": DATA_DIR / "space_user_query_mmr_results.json",
    "Proportional": DATA_DIR / "space_user_query_proportional_results.json",
}

CSV_OUTPUT = DATA_DIR / "space_user_query_evaluation.csv"
JSON_OUTPUT = DATA_DIR / "space_user_query_evaluation.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SENTIMENT_LABELS = ["positive", "neutral", "negative"]


# ============================================================
# 2. LOAD AND NORMALIZE JSON STRUCTURES
# ============================================================

def load_json(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_cases(data):
    """
    Recursively extract case dictionaries.

    Supports:
    - A single case dictionary
    - A list of cases
    - Nested lists
    - Dictionaries containing cases/results/data

    A candidate case must contain hotel information and reviews.
    A selection case must contain hotel information and a
    selected-review list.
    """

    cases = []

    if isinstance(data, list):

        for item in data:
            cases.extend(extract_cases(item))

        return cases

    if isinstance(data, dict):

        # Candidate case
        if (
            "hotel_id" in data
            and isinstance(data.get("reviews"), list)
        ):
            return [data]

        # Selection case
        selection_keys = (
            "results",
            "selected_reviews",
            "selected",
        )

        if (
            "hotel_id" in data
            and any(
                isinstance(data.get(key), list)
                for key in selection_keys
            )
        ):
            return [data]

        # Wrapper dictionary
        for key in ("cases", "results", "data"):
            if key in data:
                cases.extend(extract_cases(data[key]))

        return cases

    return cases


def normalize_query(query):

    return " ".join(str(query).strip().lower().split())


def get_selected_reviews(case):

    for key in (
        "selected_reviews",
        "results",
        "selected",
    ):

        if isinstance(case.get(key), list):
            return case[key]

    raise ValueError(
        f"No selected reviews found. Available keys: "
        f"{list(case.keys())}"
    )


# ============================================================
# 3. SENTIMENT DISTRIBUTION
# ============================================================

def sentiment_distribution(reviews):

    counts = Counter()

    for review in reviews:

        sentiment = str(
            review.get("sentiment", "unknown")
        ).lower()

        if sentiment in SENTIMENT_LABELS:
            counts[sentiment] += 1

    total = sum(counts.values())

    if total == 0:
        return {
            label: None
            for label in SENTIMENT_LABELS
        }

    return {
        label: counts[label] / total
        for label in SENTIMENT_LABELS
    }


def sentiment_js_distance(
    candidate_reviews,
    selected_reviews
):

    candidate_dist = sentiment_distribution(
        candidate_reviews
    )

    selected_dist = sentiment_distribution(
        selected_reviews
    )

    if (
        any(candidate_dist[label] is None
            for label in SENTIMENT_LABELS)
        or
        any(selected_dist[label] is None
            for label in SENTIMENT_LABELS)
    ):
        return None

    candidate_values = np.array(
        [candidate_dist[label] for label in SENTIMENT_LABELS]
    )

    selected_values = np.array(
        [selected_dist[label] for label in SENTIMENT_LABELS]
    )

    return float(
        jensenshannon(
            candidate_values,
            selected_values,
            base=2,
        )
    )


# ============================================================
# 4. AVERAGE RELEVANCE
# ============================================================

def average_relevance(
    selected_reviews,
    candidate_reviews
):
    """
    Use original retrieval relevance scores for every method.

    This ensures a fair comparison between Top-K, MMR
    and Proportional.
    """

    candidate_scores = {}

    for review in candidate_reviews:

        review_id = str(review["review_id"])

        score = review.get("relevance_score")

        if score is not None:
            candidate_scores[review_id] = float(score)

    scores = []

    for review in selected_reviews:

        review_id = str(review["review_id"])

        if review_id not in candidate_scores:
            raise ValueError(
                f"Review {review_id} not found in original "
                "candidate relevance lookup."
            )

        scores.append(candidate_scores[review_id])

    if not scores:
        return None

    return float(np.mean(scores))


# ============================================================
# 5. SEMANTIC DIVERSITY
# ============================================================

def semantic_diversity(
    selected_reviews,
    embedding_cache
):
    """
    Mean pairwise cosine distance between selected reviews.
    Higher values generally indicate greater diversity.
    """

    texts = [
        str(review.get("text", "")).strip()
        for review in selected_reviews
    ]

    texts = [text for text in texts if text]

    if len(texts) < 2:
        return None

    embeddings = np.array([
        embedding_cache[text]
        for text in texts
    ])

    similarity_matrix = cosine_similarity(embeddings)

    distance_matrix = 1.0 - similarity_matrix

    upper_triangle = np.triu_indices(
        len(texts),
        k=1,
    )

    return float(
        np.mean(distance_matrix[upper_triangle])
    )


# ============================================================
# 6. VALIDATE CASES
# ============================================================

def validate_same_case(candidate_case, method_case, method_name):

    candidate_hotel = str(candidate_case["hotel_id"])
    method_hotel = str(method_case["hotel_id"])

    candidate_query = normalize_query(
        candidate_case["query"]
    )

    method_query = normalize_query(
        method_case["query"]
    )

    if candidate_hotel != method_hotel:
        raise ValueError(
            f"{method_name}: Hotel ID mismatch."
        )

    if candidate_query != method_query:
        raise ValueError(
            f"{method_name}: Query mismatch."
        )


# ============================================================
# 7. MAIN EVALUATION
# ============================================================

def main():

    print("\nLoading user-query evaluation data...")

    # --------------------------------------------------------
    # Load candidate reviews
    # --------------------------------------------------------

    candidate_data = load_json(CANDIDATES_FILE)

    candidate_cases = extract_cases(candidate_data)

    if len(candidate_cases) != 1:
        raise ValueError(
            f"Expected exactly one user-query candidate case, "
            f"but found {len(candidate_cases)}. "
            "Please inspect the candidate JSON structure."
        )

    candidate_case = candidate_cases[0]

    candidate_reviews = candidate_case.get("reviews", [])

    if not candidate_reviews:
        raise ValueError("No candidate reviews found.")

    print(f"Hotel: {candidate_case.get('hotel_name')}")
    print(f"Query: {candidate_case.get('query')}")
    print(f"Candidate reviews: {len(candidate_reviews)}")

    # --------------------------------------------------------
    # Load selection outputs
    # --------------------------------------------------------

    method_cases = {}

    for method_name, file_path in METHOD_FILES.items():

        data = load_json(file_path)

        cases = extract_cases(data)

        if len(cases) != 1:
            raise ValueError(
                f"{method_name}: Expected exactly one "
                f"selection case, but found {len(cases)}."
            )

        method_case = cases[0]

        validate_same_case(
            candidate_case,
            method_case,
            method_name,
        )

        selected_reviews = get_selected_reviews(
            method_case
        )

        if not selected_reviews:
            raise ValueError(
                f"{method_name}: No selected reviews found."
            )

        if len(selected_reviews) != 5:
            print(
                f"WARNING: {method_name} selected "
                f"{len(selected_reviews)} reviews instead of 5."
            )

        method_cases[method_name] = {
            "case": method_case,
            "reviews": selected_reviews,
        }

        print(
            f"{method_name}: "
            f"{len(selected_reviews)} selected reviews loaded."
        )

    # --------------------------------------------------------
    # Collect unique selected texts
    # --------------------------------------------------------

    unique_texts = set()

    for method_data in method_cases.values():

        for review in method_data["reviews"]:

            text = str(review.get("text", "")).strip()

            if text:
                unique_texts.add(text)

    unique_texts = sorted(unique_texts)

    print(
        f"\nUnique selected review texts: {len(unique_texts)}"
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    embedding_cache = {}

    if unique_texts:

        embeddings = model.encode(
            unique_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        embedding_cache = {
            text: embedding
            for text, embedding in zip(
                unique_texts,
                embeddings,
            )
        }

    # --------------------------------------------------------
    # Calculate evaluation metrics
    # --------------------------------------------------------

    print("\nCalculating evaluation metrics...")

    rows = []

    candidate_dist = sentiment_distribution(
        candidate_reviews
    )

    for method_name, method_data in method_cases.items():

        selected_reviews = method_data["reviews"]

        selected_dist = sentiment_distribution(
            selected_reviews
        )

        js_distance = sentiment_js_distance(
            candidate_reviews,
            selected_reviews,
        )

        relevance = average_relevance(
            selected_reviews,
            candidate_reviews,
        )

        diversity = semantic_diversity(
            selected_reviews,
            embedding_cache,
        )

        row = {
            "hotel_id": candidate_case["hotel_id"],
            "hotel_name": candidate_case.get("hotel_name", ""),
            "query": candidate_case["query"],
            "method": method_name,
            "candidate_count": len(candidate_reviews),
            "selected_count": len(selected_reviews),

            "candidate_positive_pct": (
                candidate_dist["positive"] * 100
                if candidate_dist["positive"] is not None
                else None
            ),

            "candidate_neutral_pct": (
                candidate_dist["neutral"] * 100
                if candidate_dist["neutral"] is not None
                else None
            ),

            "candidate_negative_pct": (
                candidate_dist["negative"] * 100
                if candidate_dist["negative"] is not None
                else None
            ),

            "selected_positive_pct": (
                selected_dist["positive"] * 100
                if selected_dist["positive"] is not None
                else None
            ),

            "selected_neutral_pct": (
                selected_dist["neutral"] * 100
                if selected_dist["neutral"] is not None
                else None
            ),

            "selected_negative_pct": (
                selected_dist["negative"] * 100
                if selected_dist["negative"] is not None
                else None
            ),

            "sentiment_js_distance": js_distance,
            "average_relevance": relevance,
            "semantic_diversity": diversity,
        }

        rows.append(row)

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        CSV_OUTPUT,
        index=False,
    )

    print(f"\nCSV saved to: {CSV_OUTPUT}")

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------

    summary = {
        "hotel_id": candidate_case["hotel_id"],
        "hotel_name": candidate_case.get("hotel_name"),
        "query": candidate_case["query"],
        "candidate_count": len(candidate_reviews),
        "methods": {},
        "evaluation_notes": {
            "sentiment_js_distance": (
                "Measures how closely the selected sentiment "
                "distribution matches the candidate distribution. "
                "Lower means closer, not necessarily more accurate."
            ),
            "average_relevance": (
                "Mean original retrieval relevance score of selected "
                "reviews. The same candidate scores are used for all methods."
            ),
            "semantic_diversity": (
                "Mean pairwise cosine distance between selected "
                "review embeddings. Higher generally indicates more diversity."
            ),
            "scope": (
                "Selection-only evaluation for one user-provided "
                "hotel and custom query. No generated answer or "
                "summary is evaluated."
            ),
        },
    }

    for row in rows:

        method_name = row["method"]

        summary["methods"][method_name] = {
            "selected_count": row["selected_count"],
            "sentiment_js_distance": row["sentiment_js_distance"],
            "average_relevance": row["average_relevance"],
            "semantic_diversity": row["semantic_diversity"],
            "selected_sentiment_distribution": {
                "positive_pct": row["selected_positive_pct"],
                "neutral_pct": row["selected_neutral_pct"],
                "negative_pct": row["selected_negative_pct"],
            },
        }

    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"JSON saved to: {JSON_OUTPUT}")

    # --------------------------------------------------------
    # Print comparison table
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("USER-QUERY EVALUATION RESULTS")
    print("=" * 90)

    print(f"Hotel: {candidate_case.get('hotel_name')}")
    print(f"Query: {candidate_case.get('query')}")

    print("-" * 90)

    print(
        f"{'Method':<15}"
        f"{'JS Distance':>16}"
        f"{'Avg Relevance':>18}"
        f"{'Semantic Diversity':>23}"
    )

    print("-" * 90)

    for row in rows:

        print(
            f"{row['method']:<15}"
            f"{row['sentiment_js_distance']:>16.4f}"
            f"{row['average_relevance']:>18.4f}"
            f"{row['semantic_diversity']:>23.4f}"
        )

    print("=" * 90)
    print("USER-QUERY EVALUATION COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()
