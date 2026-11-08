# Author: William Degrer
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon

from . import core

DATA_DIR = "data/space/processed"
CANDIDATES_FILE = DATA_DIR + "/space_retrieved_reviews.json"
METHOD_FILES = {
    "Top-K": DATA_DIR + "/space_top_k_results.json",
    "MMR": DATA_DIR + "/space_mmr_results.json",
    "Proportional": DATA_DIR + "/space_proportional_results.json",
}
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_FINAL = 5

METRICS = ["js", "tvd_stars", "minority_lost", "pos_bias", "relevance", "diversity"]
LABELS = {
    "js": "Sentiment JS distance",
    "tvd_stars": "TVD over 1-5 stars",
    "minority_lost": "Minority lost",
    "pos_bias": "Positive bias (sel - pool)",
    "relevance": "Relevance",
    "diversity": "Semantic diversity",
}


def sent_counts(reviews):
    c = np.zeros(3)
    for r in reviews:
        if r["sentiment"] == "positive":
            c[0] += 1
        elif r["sentiment"] == "neutral":
            c[1] += 1
        elif r["sentiment"] == "negative":
            c[2] += 1
    return c


def star_counts(reviews):
    c = np.zeros(5)
    for r in reviews:
        stars = int(round(float(r["rating"])))
        c[stars - 1] += 1
    return c


def js_from_counts(sel, pool):
    return float(jensenshannon(sel / sel.sum(), pool / pool.sum(), base=2))


def case_metrics(idx, cands, emb):
    sel = []
    for i in idx:
        sel.append(cands[i])
    pool_s = sent_counts(cands)
    sel_s = sent_counts(sel)
    pool_r = star_counts(cands)
    sel_r = star_counts(sel)

    if pool_s[1] + pool_s[2] > 0:
        if sel_s[1] + sel_s[2] == 0:
            minority_lost = 1.0
        else:
            minority_lost = 0.0
    else:
        minority_lost = np.nan

    tvd = 0
    for a, b in zip(sel_r / sel_r.sum(), pool_r / pool_r.sum()):
        tvd += abs(a - b)
    tvd = tvd / 2

    rel = []
    for c in sel:
        rel.append(c["relevance_score"])

    dists = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            dists.append(1 - float(np.dot(emb[idx[a]], emb[idx[b]])))

    return {
        "js": js_from_counts(sel_s, pool_s),
        "tvd_stars": float(tvd),
        "minority_lost": minority_lost,
        "pos_bias": sel_s[0] / sel_s.sum() - pool_s[0] / pool_s.sum(),
        "relevance": float(np.mean(rel)),
        "diversity": float(np.mean(dists)),
    }


# best JS any k-subset can get, only the sentiment counts matter
def oracle_js(cands, k):
    pool = sent_counts(cands)
    best = 999
    for a in range(int(min(k, pool[0])) + 1):
        for b in range(int(min(k - a, pool[1])) + 1):
            c = k - a - b
            if c <= pool[2]:
                js = js_from_counts(np.array([a, b, c], dtype=float), pool)
                if js < best:
                    best = js
    return best


def bootstrap_ci(x, n_boot=10000, seed=0):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        means.append(rng.choice(x, size=len(x), replace=True).mean())
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_test(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    if len(d) == 0:
        return {"n": 0, "mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": 1.0}
    mean, lo, hi = bootstrap_ci(d)
    if np.all(d == 0):
        p = 1.0
    else:
        p = float(wilcoxon(d).pvalue)
    return {"n": int(len(d)), "mean_diff": mean, "ci_low": lo, "ci_high": hi, "p": p}


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0] * m
    running = 0
    for rank in range(m):
        i = order[rank]
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return np.array(adj)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_selected(case):
    if "selected_reviews" in case:
        return case["selected_reviews"]
    return case["results"]


def fmt(x):
    if x is None or np.isnan(x):
        return "-"
    return "%.3f" % x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results/eval")
    parser.add_argument("--n_random", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    cases = load(CANDIDATES_FILE)
    methods = {}
    for m in METHOD_FILES:
        methods[m] = {}
        for c in load(METHOD_FILES[m]):
            methods[m][(str(c["hotel_id"]), c["aspect"])] = c

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    texts = []
    for c in cases:
        for r in c["reviews"]:
            if r["text"] not in texts:
                texts.append(r["text"])
    vecs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    text_vec = {}
    for t, v in zip(texts, vecs):
        text_vec[t] = v

    rng = np.random.default_rng(args.seed)
    rows = []
    ablation_rows = []
    agreement = {"Top-K": 0, "MMR": 0, "Proportional": 0}

    for c in cases:
        key = (str(c["hotel_id"]), c["aspect"])
        cands = c["reviews"]
        emb = np.array([text_vec[r["text"]] for r in cands])
        id_to_idx = {}
        for i, r in enumerate(cands):
            id_to_idx[str(r["review_id"])] = i
        base = {"hotel_id": key[0], "hotel_name": c["hotel_name"], "aspect": key[1], "K": len(cands)}

        for m in methods:
            idx = [id_to_idx[str(r["review_id"])] for r in get_selected(methods[m][key])]
            row = dict(base)
            row["method"] = m
            row.update(case_metrics(idx, cands, emb))
            rows.append(row)

        # random baseline = mean over many random draws
        draws = []
        for _ in range(args.n_random):
            idx = list(rng.choice(len(cands), K_FINAL, replace=False))
            draws.append(case_metrics(idx, cands, emb))
        row = dict(base)
        row["method"] = "Random"
        for name in METRICS:
            vals = [d[name] for d in draws]
            if all(np.isnan(v) for v in vals):
                row[name] = np.nan
            else:
                row[name] = float(np.nanmean(vals))
        rows.append(row)

        row = dict(base)
        row["method"] = "Oracle (best JS)"
        row["js"] = oracle_js(cands, K_FINAL)
        rows.append(row)

        rel = np.array([r["relevance_score"] for r in cands])
        sim = emb @ emb.T
        ref = {
            "Top-K": core.topk(rel, K_FINAL),
            "MMR": core.mmr(rel, sim, K_FINAL),
            "Proportional": core.proportional(rel, sim, K_FINAL),
        }
        for m in methods:
            got = [id_to_idx[str(r["review_id"])] for r in get_selected(methods[m][key])]
            if set(got) == set(ref[m]):
                agreement[m] += 1

        for lam, pen, first in itertools.product([0.0, 0.25, 0.5, 0.75, 1.0], [0, 1, 2], ["hpf", "relevance"]):
            idx = core.proportional(rel, sim, K_FINAL, lam=lam, penalty=pen, first=first)
            row = dict(base)
            row["lam"] = lam
            row["penalty"] = pen
            row["first"] = first
            row.update(case_metrics(idx, cands, emb))
            ablation_rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out + "/per_case.csv", index=False)
    method_order = ["Top-K", "MMR", "Proportional", "Random", "Oracle (best JS)"]

    summary = []
    for aspect in ["all", "cleanliness", "service"]:
        if aspect == "all":
            sub = df
        else:
            sub = df[df.aspect == aspect]
        for m in method_order:
            row = {"aspect": aspect, "method": m}
            for name in METRICS:
                vals = sub[sub.method == m][name].to_numpy(dtype=float)
                if np.all(np.isnan(vals)):
                    row[name] = np.nan
                    row[name + "_lo"] = np.nan
                    row[name + "_hi"] = np.nan
                else:
                    row[name], row[name + "_lo"], row[name + "_hi"] = bootstrap_ci(vals)
            summary.append(row)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out + "/summary.csv", index=False)

    pairs = [("Proportional", "Top-K"), ("Proportional", "MMR"), ("Proportional", "Random"),
             ("Top-K", "Random"), ("MMR", "Random"), ("MMR", "Top-K")]
    tests = []
    for aspect in ["all", "cleanliness", "service"]:
        if aspect == "all":
            sub = df
        else:
            sub = df[df.aspect == aspect]
        for name in METRICS:
            piv = sub.pivot_table(index=["hotel_id", "aspect"], columns="method", values=name, dropna=False)
            block = []
            for a, b in pairs:
                t = paired_test(piv[a].to_numpy(), piv[b].to_numpy())
                t.update({"aspect": aspect, "metric": name, "a": a, "b": b})
                block.append(t)
            adj = holm(np.array([t["p"] for t in block]))
            for i in range(len(block)):
                block[i]["p_holm"] = float(adj[i])
            tests += block
    tdf = pd.DataFrame(tests)
    tdf.to_csv(out + "/paired_tests.csv", index=False)

    adf = pd.DataFrame(ablation_rows)
    abl = adf.groupby(["lam", "penalty", "first"])[METRICS].mean().reset_index()
    abl.to_csv(out + "/ablation_proportional.csv", index=False)

    n_cases = len(cases)
    lines = []
    lines.append("# SPACE selection, statistical evaluation")
    lines.append("")
    lines.append("%d hotel-aspect cases (50 hotels x cleanliness and service), up to 20 candidates, k = %d. "
                 "Random is the mean of %d random draws per case. Numbers is mean with 95%% bootstrap CI "
                 "(10000 resamples over the cases)." % (n_cases, K_FINAL, args.n_random))
    lines.append("")
    lines.append("Our own implementations in src/eval/core.py picks the same reviews as src/space/ in: "
                 + ", ".join("%s %d/%d" % (m, agreement[m], n_cases) for m in agreement) + ".")
    lines.append("")
    for aspect in ["all", "cleanliness", "service"]:
        lines.append("## Aspect: " + aspect)
        lines.append("")
        lines.append("| Method | " + " | ".join(LABELS[n] for n in METRICS) + " |")
        lines.append("|---|" + "---|" * len(METRICS))
        for m in method_order:
            r = sdf[(sdf.aspect == aspect) & (sdf.method == m)].iloc[0]
            cells = []
            for n in METRICS:
                if np.isnan(r[n]):
                    cells.append("-")
                else:
                    cells.append("%s [%s, %s]" % (fmt(r[n]), fmt(r[n + "_lo"]), fmt(r[n + "_hi"])))
            lines.append("| " + m + " | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("## Paired differences (a - b), all cases")
    lines.append("")
    lines.append("| Metric | a vs b | Mean diff [95% CI] | Wilcoxon p | Holm p |")
    lines.append("|---|---|---|---|---|")
    for t in tests:
        if t["aspect"] == "all":
            lines.append("| %s | %s vs %s | %+.3f [%+.3f, %+.3f] | %.3g | %.3g |" % (
                LABELS[t["metric"]], t["a"], t["b"], t["mean_diff"], t["ci_low"], t["ci_high"], t["p"], t["p_holm"]))
    lines.append("")
    lines.append("## Ablation of the proportional score (mean over all cases)")
    lines.append("")
    lines.append("penalty 2 = p_S - 2p_R (current code, exact HPF gain), 1 = p_S - p_R, 0 = only p_S. "
                 "first: hpf = current code, relevance = first pick on relevance only like IAdU (Eq. 17).")
    lines.append("")
    lines.append("| lambda | penalty | first pick | JS | minority lost | pos. bias | relevance | diversity |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in abl.iterrows():
        lines.append("| %s | %d | %s | %s | %s | %+.3f | %s | %s |" % (
            r["lam"], int(r["penalty"]), r["first"], fmt(r["js"]), fmt(r["minority_lost"]),
            r["pos_bias"], fmt(r["relevance"]), fmt(r["diversity"])))

    with open(out + "/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(out + "/meta.json", "w") as f:
        json.dump({"agreement": agreement, "n_cases": n_cases}, f)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
