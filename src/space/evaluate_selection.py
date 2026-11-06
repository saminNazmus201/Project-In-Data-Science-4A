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

CANDIDATES_FILE = DATA_DIR / "space_retrieved_reviews.json"

METHOD_FILES = {
    "Top-K": DATA_DIR / "space_top_k_results.json",
    "MMR": DATA_DIR / "space_mmr_results.json",
    "Proportional": DATA_DIR / "space_proportional_results.json",
}

PER_CASE_OUTPUT = DATA_DIR / "space_evaluation_per_case.csv"
SUMMARY_OUTPUT = DATA_DIR / "space_evaluation_summary.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SENTIMENT_LABELS = ["positive", "neutral", "negative"]


# ============================================================
# 2. LOAD JSON FILES
# ============================================================

def load_json(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"\nFile not found: {file_path}\n"
            "Please run the corresponding SPACE pipeline first."
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_case_list(data, file_path):
    """
    Supports either a list of cases or a dictionary containing
    a list under 'cases' or 'results'.
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("cases", "results"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        f"Unexpected JSON structure in {file_path}. "
        "Expected a list of cases or a dictionary containing "
        "'cases' or 'results'."
    )


def normalize_id(value):
    return str(value)


def case_key(case):
    return (
        normalize_id(case["hotel_id"]),
        str(case["aspect"]).strip().lower(),
    )



def get_selected_reviews(case):
    """
    Supports the actual result structures of SPACE methods:
    Top-K and MMR use 'results'.
    Proportional uses 'selected_reviews'.
    """
    for key in ("selected_reviews", "results", "selected", "reviews"):
        if isinstance(case.get(key), list):
            return case[key]

    raise ValueError(
        f"No selected-review list found for hotel_id="
        f"{case.get('hotel_id')}, aspect={case.get('aspect')}. "
        f"Available keys: {list(case.keys())}"
    )


# ============================================================
# 3. SENTIMENT DISTRIBUTION
# ============================================================

def sentiment_distribution(reviews):
    """
    Calculate positive, neutral and negative proportions.
    Reviews with unknown or missing sentiment are excluded.
    """
    counts = Counter()

    for review in reviews:
        sentiment = str(review.get("sentiment", "unknown")).lower()

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


def sentiment_js_distance(candidate_reviews, selected_reviews):
    """
    Compare the selected sentiment distribution with the
    candidate sentiment distribution.

    Lower JS distance means the distributions are more similar.
    """
    candidate_dist = sentiment_distribution(candidate_reviews)
    selected_dist = sentiment_distribution(selected_reviews)

    if (
        any(candidate_dist[label] is None for label in SENTIMENT_LABELS)
        or any(selected_dist[label] is None for label in SENTIMENT_LABELS)
    ):
        return None

    candidate_values = np.array(
        [candidate_dist[label] for label in SENTIMENT_LABELS],
        dtype=float,
    )

    selected_values = np.array(
        [selected_dist[label] for label in SENTIMENT_LABELS],
        dtype=float,
    )

    # scipy returns Jensen-Shannon distance, not squared distance.
    distance = jensenshannon(
        candidate_values,
        selected_values,
        base=2,
    )

    return float(distance)


# ============================================================
# 4. FAIR AVERAGE RELEVANCE
# ============================================================

def average_relevance(selected_reviews, candidate_reviews):
    """
    Use relevance scores from the original retrieved candidates
    for every method, including MMR.

    This avoids comparing scores recalculated differently by
    different selection methods.
    """
    candidate_scores = {}

    for review in candidate_reviews:
        review_id = normalize_id(review["review_id"])

        if review.get("relevance_score") is not None:
            candidate_scores[review_id] = float(
                review["relevance_score"]
            )

    scores = []

    for review in selected_reviews:
        review_id = normalize_id(review["review_id"])

        if review_id not in candidate_scores:
            raise ValueError(
                f"Selected review {review_id} was not found in the "
                "retrieved candidate list. Check that all methods "
                "select reviews from the same candidate set."
            )

        scores.append(candidate_scores[review_id])

    if not scores:
        return None

    return float(np.mean(scores))


# ============================================================
# 5. SEMANTIC DIVERSITY
# ============================================================

def semantic_diversity(selected_reviews, embedding_cache):
    """
    Calculate mean pairwise cosine distance among selected reviews.

    Higher values generally indicate greater semantic diversity.
    """
    texts = [
        str(review.get("text", "")).strip()
        for review in selected_reviews
    ]

    texts = [text for text in texts if text]

    if len(texts) < 2:
        return None

    embeddings = np.array(
        [embedding_cache[text] for text in texts]
    )

    similarity_matrix = cosine_similarity(embeddings)

    # Convert cosine similarity into cosine distance.
    distance_matrix = 1.0 - similarity_matrix

    # Only use each unique pair, excluding the diagonal.
    upper_triangle = np.triu_indices(
        len(texts),
        k=1,
    )

    distances = distance_matrix[upper_triangle]

    return float(np.mean(distances))


# ============================================================
# 6. MAIN EVALUATION
# ============================================================

def main():
    print("\nLoading SPACE evaluation data...")

    candidate_data = load_json(CANDIDATES_FILE)
    candidate_cases = get_case_list(
        candidate_data,
        CANDIDATES_FILE,
    )

    candidate_map = {
        case_key(case): case
        for case in candidate_cases
    }

    if len(candidate_map) != len(candidate_cases):
        raise ValueError(
            "Duplicate hotel-aspect cases found in "
            "space_retrieved_reviews.json."
        )

    print(f"Retrieved candidate cases: {len(candidate_map)}")

    # --------------------------------------------------------
    # Load all three selection methods
    # --------------------------------------------------------

    method_maps = {}

    for method_name, file_path in METHOD_FILES.items():
        data = load_json(file_path)
        cases = get_case_list(data, file_path)

        case_map = {
            case_key(case): case
            for case in cases
        }

        if len(case_map) != len(cases):
            raise ValueError(
                f"Duplicate hotel-aspect cases in {file_path}"
            )

        if set(case_map.keys()) != set(candidate_map.keys()):
            missing = set(candidate_map.keys()) - set(case_map.keys())
            extra = set(case_map.keys()) - set(candidate_map.keys())

            raise ValueError(
                f"\nCase mismatch for {method_name}."
                f"\nMissing cases: {list(missing)[:5]}"
                f"\nUnexpected cases: {list(extra)[:5]}"
            )

        method_maps[method_name] = case_map

        print(f"{method_name}: {len(case_map)} cases loaded")

    # --------------------------------------------------------
    # Collect unique selected texts and encode them in one batch
    # --------------------------------------------------------

    unique_texts = set()

    for method_name, case_map in method_maps.items():
        for case in case_map.values():
            for review in get_selected_reviews(case):
                text = str(review.get("text", "")).strip()

                if text:
                    unique_texts.add(text)

    unique_texts = sorted(unique_texts)

    print(f"\nUnique selected review texts: {len(unique_texts)}")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    if unique_texts:
        embeddings = model.encode(
            unique_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        embedding_cache = {
            text: embedding
            for text, embedding in zip(unique_texts, embeddings)
        }
    else:
        embedding_cache = {}

    # --------------------------------------------------------
    # Evaluate each case for each method
    # --------------------------------------------------------

    rows = []

    for method_name, case_map in method_maps.items():
        print(f"\nEvaluating {method_name}...")

        for key, method_case in case_map.items():
            candidate_case = candidate_map[key]

            candidate_reviews = candidate_case.get("reviews", [])
            selected_reviews = get_selected_reviews(method_case)

            candidate_dist = sentiment_distribution(
                candidate_reviews
            )

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
                "aspect": str(candidate_case["aspect"]).lower(),
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
    # Save per-case CSV
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        PER_CASE_OUTPUT,
        index=False,
    )

    print(f"\nPer-case results saved to:\n{PER_CASE_OUTPUT}")

    # --------------------------------------------------------
    # Overall and aspect-specific summaries
    # --------------------------------------------------------

    metrics = [
        "sentiment_js_distance",
        "average_relevance",
        "semantic_diversity",
    ]

    summary = {
        "evaluation_notes": {
            "total_hotel_aspect_cases": int(len(candidate_map)),
            "methods": list(METHOD_FILES.keys()),
            "expected_rows": int(len(candidate_map) * len(METHOD_FILES)),
            "sentiment_labels": SENTIMENT_LABELS,
            "sentiment_source": (
                "Rating-based sentiment labels from the prepared SPACE dataset."
            ),
            "sentiment_js_distance": (
                "Measures how closely selected-review sentiment "
                "distribution matches the retrieved-candidate distribution. "
                "Lower is closer; the candidate distribution is a baseline, "
                "not ground-truth hotel opinion distribution."
            ),
            "average_relevance": (
                "Mean original retrieval relevance score of selected reviews. "
                "The same candidate scores are used for all methods."
            ),
            "semantic_diversity": (
                "Mean pairwise cosine distance between selected review "
                "embeddings. Higher generally indicates more diversity."
            ),
            "scope": (
                "Selection-only evaluation; no generated summaries are "
                "evaluated in this script."
            ),
        },
        "overall": {},
        "by_aspect": {},
    }

    def summarize(group_df):
        result = {
            "number_of_cases": int(len(group_df)),
        }

        for method_name in METHOD_FILES:
            method_df = group_df[
                group_df["method"] == method_name
            ]

            result[method_name] = {
                metric: (
                    float(method_df[metric].mean())
                    if method_df[metric].notna().any()
                    else None
                )
                for metric in metrics
            }

            result[method_name]["number_of_cases"] = int(
                len(method_df)
            )

        return result

    summary["overall"] = summarize(df)

    for aspect in ("cleanliness", "service"):
        aspect_df = df[df["aspect"] == aspect]

        if aspect_df.empty:
            print(
                f"\nWARNING: No cases found for aspect '{aspect}'."
            )
            continue

        summary["by_aspect"][aspect] = summarize(aspect_df)

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Print readable metric tables
    # --------------------------------------------------------

    def print_summary(title, group_df):
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)

        print(
            f"{'Method':<15}"
            f"{'JS Distance':>16}"
            f"{'Avg Relevance':>18}"
            f"{'Semantic Diversity':>23}"
        )

        print("-" * 78)

        for method_name in METHOD_FILES:
            method_df = group_df[
                group_df["method"] == method_name
            ]

            values = []

            for metric in metrics:
                mean_value = method_df[metric].mean()

                values.append(
                    f"{mean_value:.4f}"
                    if pd.notna(mean_value)
                    else "N/A"
                )

            print(
                f"{method_name:<15}"
                f"{values[0]:>16}"
                f"{values[1]:>18}"
                f"{values[2]:>23}"
            )

    print_summary("OVERALL RESULTS", df)

    for aspect in ("cleanliness", "service"):
        aspect_df = df[df["aspect"] == aspect]

        if not aspect_df.empty:
            print_summary(
                f"{aspect.upper()} RESULTS",
                aspect_df,
            )

    print("\n" + "=" * 78)
    print("EVALUATION COMPLETED")
    print("=" * 78)
    print(f"Cases evaluated: {len(candidate_map)}")
    print(f"Methods compared: {len(METHOD_FILES)}")
    print(f"Expected per-case rows: {len(candidate_map) * len(METHOD_FILES)}")
    print(f"CSV output: {PER_CASE_OUTPUT}")
    print(f"Summary JSON: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
