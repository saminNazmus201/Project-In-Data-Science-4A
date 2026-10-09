"""Maximal Marginal Relevance.

score(p) = lam * rel(p) - (1 - lam) * max_{s in selected} sim(p, s)
rel(p)   = cos(question, p)
sim(p,s) = cos(p, s)
Embeddings are L2-normalized, so cosine = dot product.
"""
import numpy as np

from ..embeddings import Embedder
from .base import Selector


class MMRSelector(Selector):
    name = "mmr"

    def __init__(self, lam=0.5, embedder=None):
        self.lam = lam
        self.embedder = embedder or Embedder()

    def select(self, question: str, docs: list[dict], k: int) -> list[dict]:
        if not docs:
            return []
        q = self.embedder.encode_query(question)
        P = self.embedder.encode_docs(docs)
        rel = P @ q            # (n,)
        sim = P @ P.T          # (n, n)

        selected = []
        remaining = list(range(len(docs)))
        while remaining and len(selected) < k:
            if selected:
                redundancy = sim[np.ix_(remaining, selected)].max(axis=1)
            else:
                redundancy = np.zeros(len(remaining))
            scores = self.lam * rel[remaining] - (1 - self.lam) * redundancy
            best = remaining[int(np.argmax(scores))]  # ties -> earlier retrieval rank
            selected.append(best)
            remaining.remove(best)
        return [docs[i] for i in selected]
