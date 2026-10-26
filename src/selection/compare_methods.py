import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.top_k import TopKRetriever
from src.selection.mmr import MMRSelector
from src.selection.proportional import ProportionalSelector


QUESTION = "Who has scored the most goals in international football?"
CANDIDATE_K = 20
FINAL_K = 5


def calculate_metrics(results, retriever):
    """Calculate mean relevance and pairwise redundancy for selected chunks."""
    if not results:
        return {
            "mean_relevance": 0.0,
            "mean_redundancy": 0.0,
            "unique_articles": 0,
        }

    embeddings = retriever.model.encode(
        [item["text"] for item in results],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    similarities = cosine_similarity(embeddings)

    n = len(results)
    pairwise_similarities = [
        similarities[i, j]
        for i in range(n)
        for j in range(i + 1, n)
    ]

    mean_relevance = np.mean([item["score"] for item in results])
    mean_redundancy = (
        float(np.mean(pairwise_similarities))
        if pairwise_similarities
        else 0.0
    )

    unique_articles = len({item["title"] for item in results})

    return {
        "mean_relevance": float(mean_relevance),
        "mean_redundancy": mean_redundancy,
        "unique_articles": unique_articles,
    }


def print_results(method_name, results, retriever):
    metrics = calculate_metrics(results, retriever)

    print(f"\n{'=' * 70}")
    print(f"{method_name}")
    print("=" * 70)

    for rank, item in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Title: {item['title']}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Relevance score: {item['score']:.4f}")
        print(f"Text: {item['text'][:350].replace(chr(10), ' ')}")

    print("\nMetrics:")
    print(f"Mean relevance: {metrics['mean_relevance']:.4f}")
    print(f"Mean pairwise redundancy: {metrics['mean_redundancy']:.4f}")
    print(f"Unique articles: {metrics['unique_articles']}")

    return metrics


def main():
    print("Question:", QUESTION)
    print(f"Candidate pool: Top-{CANDIDATE_K}")
    print(f"Final selection: {FINAL_K}")

    retriever = TopKRetriever()

    # All methods use the same retriever and candidate-pool size.
    top_k_results = retriever.retrieve(QUESTION, k=CANDIDATE_K)[:FINAL_K]

    mmr = MMRSelector(retriever=retriever, lambda_param=0.5)
    mmr_results = mmr.select(
        QUESTION,
        candidate_k=CANDIDATE_K,
        final_k=FINAL_K,
    )

    proportional = ProportionalSelector(
        retriever=retriever,
        lambda_param=0.5,
    )
    proportional_results = proportional.select(
        QUESTION,
        candidate_k=CANDIDATE_K,
        final_k=FINAL_K,
    )

    all_metrics = {}

    all_metrics["Top-k"] = print_results(
        "TOP-K RESULTS", top_k_results, retriever
    )
    all_metrics["MMR"] = print_results(
        "MMR RESULTS", mmr_results, retriever
    )
    all_metrics["Proportional"] = print_results(
        "PROPORTIONAL RESULTS", proportional_results, retriever
    )

    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(
        f"{'Method':<16}"
        f"{'Mean relevance':>18}"
        f"{'Redundancy':>15}"
        f"{'Unique articles':>18}"
    )

    for method, metrics in all_metrics.items():
        print(
            f"{method:<16}"
            f"{metrics['mean_relevance']:>18.4f}"
            f"{metrics['mean_redundancy']:>15.4f}"
            f"{metrics['unique_articles']:>18}"
        )


if __name__ == "__main__":
    main()
