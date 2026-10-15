"""Common interface for all evidence selectors.

Every selector implements:
    select(question: str, docs: list[dict], k: int) -> list[dict]
returning the chosen docs in selection order. `docs` is the candidate pool
(already sorted by retrieval rank); the returned dicts are the original doc
dicts, unmodified.
"""
from abc import ABC, abstractmethod


class Selector(ABC):
    name = "base"

    @abstractmethod
    def select(self, question: str, docs: list[dict], k: int) -> list[dict]:
        raise NotImplementedError
