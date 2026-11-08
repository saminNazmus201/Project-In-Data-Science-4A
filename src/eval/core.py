# Author: William Degrer
import numpy as np


def topk(rel, k):
    rel = list(rel)
    order = sorted(range(len(rel)), key=lambda i: -rel[i])
    return order[:k]


def mmr(rel, sim, k, lam=0.5):
    sim = np.array(sim, dtype=float)
    selected = []
    remaining = list(range(len(rel)))
    while len(remaining) > 0 and len(selected) < k:
        best = None
        best_score = -999999
        for i in remaining:
            red = 0
            if len(selected) > 0:
                red = max([sim[i][j] for j in selected])
            score = lam * rel[i] - (1 - lam) * red
            if score > best_score:
                best = i
                best_score = score
        selected.append(best)
        remaining.remove(best)
    return selected


def hpf_objective(selected, rel, sim, k=None, lam=0.5):
    sim = np.array(sim, dtype=float)
    K = len(rel)
    if k is None:
        k = len(selected)
    total = 0
    for i in selected:
        p_S = 0
        for j in range(K):
            if j != i:
                p_S += sim[i][j]
        p_R = 0
        for j in selected:
            if j != i:
                p_R += sim[i][j]
        total += (1 - lam) * (K - k) * rel[i] + lam * (p_S - p_R)
    return total


# greedy HPF, penalty 2 = exact gain of HPF(R) (same as src/space/proportional.py)
def proportional(rel, sim, k, lam=0.5, penalty=2, first="hpf"):
    sim = np.array(sim, dtype=float)
    K = len(rel)
    k = min(k, K)
    p_S = []
    for i in range(K):
        s = 0
        for j in range(K):
            if j != i:
                s += sim[i][j]
        p_S.append(s)

    selected = []
    remaining = list(range(K))
    while len(remaining) > 0 and len(selected) < k:
        best = None
        best_score = -999999
        for i in remaining:
            if len(selected) == 0 and first == "relevance":
                score = rel[i]
            else:
                p_R = 0
                for j in selected:
                    p_R += sim[i][j]
                score = (1 - lam) * (K - k) * rel[i] + lam * (p_S[i] - penalty * p_R)
            if score > best_score:
                best = i
                best_score = score
        selected.append(best)
        remaining.remove(best)
    return selected
