from .base import Selector


class TopKSelector(Selector):
    """Take the first k docs (the pool is already sorted by retrieval score)."""
    name = "topk"

    def select(self, question: str, docs: list[dict], k: int) -> list[dict]:
        return docs[:k]
