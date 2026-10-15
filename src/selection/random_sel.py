import random
import zlib

from .base import Selector


class RandomSelector(Selector):
    """k docs uniformly at random from the pool, returned in pool (rank) order.
    The seed is combined with the question, so every question gets a different
    but reproducible draw."""
    name = "random"

    def __init__(self, seed=0):
        self.seed = seed

    def select(self, question: str, docs: list[dict], k: int) -> list[dict]:
        rng = random.Random(self.seed * 1_000_003 + zlib.crc32(question.encode("utf-8")))
        idx = sorted(rng.sample(range(len(docs)), min(k, len(docs))))
        return [docs[i] for i in idx]
