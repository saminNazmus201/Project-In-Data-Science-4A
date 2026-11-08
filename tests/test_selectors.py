# Author: William Degrer
import itertools
import random

import numpy as np
import pytest

from src.eval import core
from src.eval.selection_stats import case_metrics, js_from_counts, oracle_js
from src.space.mmr import SpaceMMRSelector
from src.space.proportional import SpaceProportionalSelector
from src.space.top_k import SpaceTopKSelector


class FakeModel:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return self.vectors[texts]
        return np.array([self.vectors[t] for t in texts])


def random_case(K=20, d=8, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(K, d))
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    q = rng.normal(size=d)
    q = q / np.linalg.norm(q)
    rel = E @ q
    cands = []
    for i in range(K):
        rating = int(rng.integers(1, 6))
        if rating >= 4:
            sentiment = "positive"
        elif rating == 3:
            sentiment = "neutral"
        else:
            sentiment = "negative"
        cands.append({"review_id": "r" + str(i), "hotel_id": "h", "hotel_name": "H", "text": "text " + str(i),
                      "rating": rating, "distance": 1 - rel[i], "relevance_score": float(rel[i]),
                      "sentiment": sentiment})
    vectors = {"query": q}
    for i in range(K):
        vectors[cands[i]["text"]] = E[i]
    return cands, E, rel, FakeModel(vectors)


def make_proportional(model, lam=0.5):
    sel = SpaceProportionalSelector.__new__(SpaceProportionalSelector)
    sel.model = model
    sel.lambda_param = lam
    return sel


def ids(selected):
    return [r["review_id"] for r in selected]


# K=4, k=2, lambda=0.5 so (1-lambda)(K-k) = 1. Item 0 and 1 is almost the same, 3 is far away.
# p_S = [1.55, 1.55, 1.10, 0.30]
# step 1: r + 0.5 p_S           = [1.675, 1.625, 1.35, 0.45]  -> 0
# step 2: r + 0.5 (p_S - 2 p_R) = [-, 0.675, 0.85, 0.35]       -> 2
HAND_REL = [0.9, 0.85, 0.8, 0.3]
HAND_SIM = np.array([[1.0, 0.95, 0.5, 0.1],
                     [0.95, 1.0, 0.5, 0.1],
                     [0.5, 0.5, 1.0, 0.1],
                     [0.1, 0.1, 0.1, 1.0]])


def test_hand_example_proportional():
    assert core.proportional(HAND_REL, HAND_SIM, 2) == [0, 2]


def test_hand_example_penalty_matters():
    assert core.proportional(HAND_REL, HAND_SIM, 2, penalty=0) == [0, 1]


def test_hand_example_topk_and_mmr():
    assert core.topk(HAND_REL, 2) == [0, 1]
    assert core.mmr(HAND_REL, HAND_SIM, 2) == [0, 2]


@pytest.mark.parametrize("seed", range(5))
def test_greedy_score_is_exact_hpf_gain(seed):
    _, E, rel, _ = random_case(seed=seed)
    sim = E @ E.T
    K = len(rel)
    k = 5
    lam = 0.5
    S = sim.copy()
    np.fill_diagonal(S, 0)
    p_S = S.sum(axis=1)
    R = [3, 7]
    for x in range(K):
        if x in R:
            continue
        gain = (1 - lam) * (K - k) * rel[x] + lam * (p_S[x] - 2 * S[x, R].sum())
        diff = core.hpf_objective(R + [x], rel, sim, k=k, lam=lam) - core.hpf_objective(R, rel, sim, k=k, lam=lam)
        assert gain == pytest.approx(diff)


def test_lambda_zero_is_topk():
    _, E, rel, _ = random_case(seed=1)
    assert core.proportional(rel, E @ E.T, 5, lam=0.0) == core.topk(rel, 5)


def test_greedy_is_close_to_brute_force():
    _, E, rel, _ = random_case(K=10, seed=2)
    sim = E @ E.T
    greedy = core.hpf_objective(core.proportional(rel, sim, 4), rel, sim, k=4)
    best = -999
    for c in itertools.combinations(range(10), 4):
        best = max(best, core.hpf_objective(list(c), rel, sim, k=4))
    assert greedy <= best + 1e-9
    assert greedy >= 0.9 * best


@pytest.mark.parametrize("seed", range(5))
def test_space_selectors_match_core(seed):
    cands, E, rel, model = random_case(seed=seed)
    sim = E @ E.T
    assert ids(SpaceTopKSelector().select(cands, 5)) == ["r" + str(i) for i in core.topk(rel, 5)]
    assert ids(SpaceMMRSelector(model, 0.5).select("query", cands, 5)) == ["r" + str(i) for i in core.mmr(rel, sim, 5)]
    assert ids(make_proportional(model).select("query", cands, 5)) == ["r" + str(i) for i in core.proportional(rel, sim, 5)]


def run_selector(name, cands, model):
    if name == "topk":
        return SpaceTopKSelector().select(cands, 5)
    if name == "mmr":
        return SpaceMMRSelector(model, 0.5).select("query", cands, 5)
    return make_proportional(model).select("query", cands, 5)


@pytest.mark.parametrize("name", ["topk", "mmr", "prop"])
def test_selectors_k_unique_and_deterministic(name):
    cands, _, _, model = random_case(seed=3)
    first = ids(run_selector(name, cands, model))
    assert len(first) == 5
    assert len(set(first)) == 5
    assert ids(run_selector(name, cands, model)) == first


@pytest.mark.parametrize("name", ["topk", "mmr", "prop"])
def test_selectors_dont_use_ratings(name):
    cands, _, _, model = random_case(seed=4)
    shuffled = [dict(c) for c in cands]
    ratings = [c["rating"] for c in cands]
    random.Random(0).shuffle(ratings)
    for c, r in zip(shuffled, ratings):
        c["rating"] = r
        if r >= 4:
            c["sentiment"] = "positive"
        elif r == 3:
            c["sentiment"] = "neutral"
        else:
            c["sentiment"] = "negative"
    assert ids(run_selector(name, cands, model)) == ids(run_selector(name, shuffled, model))


def test_selectors_small_pool():
    cands, _, _, model = random_case(K=3, seed=5)
    for name in ["topk", "mmr", "prop"]:
        assert len(run_selector(name, cands, model)) == 3


def test_js_zero_for_same_distribution():
    assert js_from_counts(np.array([4.0, 1, 0]), np.array([16.0, 4, 0])) == pytest.approx(0.0)


def test_js_one_for_disjoint():
    assert js_from_counts(np.array([5.0, 0, 0]), np.array([0.0, 0, 20])) == pytest.approx(1.0)


def test_oracle_is_lower_bound():
    cands, E, _, _ = random_case(seed=6)
    best = oracle_js(cands, 5)
    for idx in itertools.islice(itertools.combinations(range(20), 5), 3000):
        assert case_metrics(list(idx), cands, E)["js"] >= best - 1e-12


def test_minority_lost():
    cands, E, _, _ = random_case(seed=7)
    pos = [i for i in range(len(cands)) if cands[i]["sentiment"] == "positive"][:5]
    other = [i for i in range(len(cands)) if cands[i]["sentiment"] != "positive"][:1]
    if len(pos) == 5 and len(other) > 0:
        assert case_metrics(pos, cands, E)["minority_lost"] == 1.0
        assert case_metrics(pos[:4] + other, cands, E)["minority_lost"] == 0.0
