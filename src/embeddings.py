"""Sentence-transformers embedder with a text -> vector cache.

The same text is never encoded twice. The cache can optionally be saved to
disk so that later scripts (e.g. compare_example.py) reuse it.
"""
import pickle
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
# BGE recommends this prefix for queries (not for passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def doc_to_text(doc):
    # review sentences (SPACE / Yelp) have an empty title
    return f"{doc['title']}\n{doc['text']}" if doc.get("title") else doc["text"]


class Embedder:
    def __init__(self, model_name=DEFAULT_MODEL, cache_path=None, device=None, query_prefix=None):
        self.model_name = model_name
        self.cache_path = cache_path
        self.device = device
        if query_prefix is None:
            query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        self.query_prefix = query_prefix
        self._model = None
        self.cache = {}
        if cache_path and Path(cache_path).exists():
            with open(cache_path, "rb") as f:
                saved = pickle.load(f)
            if saved.get("model_name") == model_name:
                self.cache = saved["vectors"]

    @property
    def model(self):
        if self._model is None:  # load lazily so cached runs don't need the model
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts):
        """Return an (n, d) array of L2-normalized embeddings."""
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            vecs = self.model.encode(missing, normalize_embeddings=True, batch_size=32)
            self.cache.update(zip(missing, vecs))
        return np.stack([self.cache[t] for t in texts])

    def encode_query(self, question):
        return self.encode([self.query_prefix + question])[0]

    def encode_docs(self, docs):
        return self.encode([doc_to_text(d) for d in docs])

    def save(self):
        if self.cache_path:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump({"model_name": self.model_name, "vectors": self.cache}, f)
