"""Selector registry. To add a new method (e.g. proportional), implement
Selector in its own file and add one line to `get_selector`."""
from .base import Selector
from .topk import TopKSelector
from .mmr import MMRSelector
from .random_sel import RandomSelector


def get_selector(method, lam=0.5, embedder=None, seed=0):
    if method == "topk":
        return TopKSelector()
    if method == "mmr":
        return MMRSelector(lam=lam, embedder=embedder)
    if method == "random":
        return RandomSelector(seed=seed)
    # if method == "proportional":
    #     from .proportional import ProportionalSelector
    #     return ProportionalSelector(embedder=embedder)
    raise ValueError(f"Unknown method: {method}")
