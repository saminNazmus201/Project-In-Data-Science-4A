# Author: William Degrer
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from .selection_stats import bootstrap_ci, holm, paired_test

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
SENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
METHODS = ["Top-K", "MMR", "Proportional", "Random"]
METRICS = ["cited", "bad_cite", "faithful", "contradicted", "cite_precision", "polarity_gap",
           "evidence_gap", "mentions_neg", "minority_lost", "rougeL", "words"]
LABELS = {
    "cited": "Has citation", "bad_cite": "Invalid citation", "faithful": "Faithful (NLI)",
    "contradicted": "Contradicted", "cite_precision": "Citation precision",
    "polarity_gap": "Polarity gap to pool", "evidence_gap": "Polarity gap to evidence",
    "mentions_neg": "Mentions a complaint", "minority_lost": "Minority lost",
    "rougeL": "ROUGE-L", "words": "Words",
}


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", text.strip())
    out = []
    for p in parts:
        if re.sub(r"\[\d+\]", "", p).strip(" .") == "" and len(out) > 0:
            out[-1] = out[-1] + " " + p
        elif p.strip() != "":
            out.append(p.strip())
    return out


def strip_cites(s):
    s = re.sub(r"\[\d+\]", "", s)
    return re.sub(r"\s+", " ", s).strip()


class NLI:
    def __init__(self):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(NLI_MODEL, max_length=512)
        labels = self.model.model.config.id2label
        for i in labels:
            if labels[i].lower().startswith("entail"):
                self.ent = int(i)
            if labels[i].lower().startswith("contra"):
                self.con = int(i)

    def probs(self, pairs):
        if len(pairs) == 0:
            return np.zeros((0, 3))
        logits = np.array(self.model.predict(pairs, batch_size=16, show_progress_bar=False))
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


class Sentiment:
    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=SENT_MODEL, top_k=None, truncation=True)

    def scores(self, sentences):
        out = []
        for res in self.pipe(sentences, batch_size=16):
            pos = 0
            neg = 0
            for r in res:
                if r["label"].lower() == "positive":
                    pos = r["score"]
                if r["label"].lower() == "negative":
                    neg = r["score"]
            out.append([pos, neg])
        return np.array(out)


def polarity_of(sentiments):
    s = list(sentiments)
    return (s.count("positive") - s.count("negative")) / len(s)


def evaluate(items, nli, senti, rouge):
    rows = []
    for n, it in enumerate(items):
        out = it["output"]
        sents = split_sentences(out)
        clean = [strip_cites(s) for s in sents]
        docs = [d["text"] for d in it["docs"]]
        cites = []
        for s in sents:
            cites.append([int(x) for x in re.findall(r"\[(\d+)\]", s)])
        all_cites = []
        for cs in cites:
            all_cites += cs

        pairs = []
        for h in clean:
            for d in docs:
                pairs.append((d, h))
        P = nli.probs(pairs).reshape(len(clean), len(docs), 3)
        ent = P[:, :, nli.ent]
        con = P[:, :, nli.con]
        supported = ent.max(axis=1) >= 0.5
        contradicted = (con.max(axis=1) >= 0.5) & ~supported

        # sentence counts as correct cited if one of its cited reviews entails it
        prec = []
        for i in range(len(sents)):
            valid = [c for c in cites[i] if 1 <= c <= len(docs)]
            if len(valid) > 0:
                prec.append(max(ent[i, c - 1] for c in valid) >= 0.5)
        cite_prec = np.mean(prec) if len(prec) > 0 else np.nan

        S = senti.scores(clean)
        pol = float(np.mean(S[:, 0] - S[:, 1]))
        pool_pol = polarity_of(it["pool_sentiment"])
        ev_pol = polarity_of([d["sentiment"] for d in it["docs"]])
        has_minority = any(s != "positive" for s in it["pool_sentiment"])
        mentions_neg = bool((S[:, 1] > 0.5).any())

        refs = it.get("references") or []
        if len(refs) > 0:
            rl = float(np.mean([rouge.score(r, strip_cites(out))["rougeL"].fmeasure for r in refs]))
        else:
            rl = np.nan

        bad = 0.0
        for c in all_cites:
            if c < 1 or c > len(docs):
                bad = 1.0
        rows.append({
            "id": it["id"], "hotel_id": it["hotel_id"], "aspect": it["aspect"], "method": it["method"],
            "n_sentences": len(sents), "words": len(strip_cites(out).split()),
            "cited": 1.0 if len(all_cites) > 0 else 0.0,
            "bad_cite": bad,
            "faithful": float(supported.mean()),
            "contradicted": float(contradicted.mean()),
            "cite_precision": cite_prec,
            "polarity": pol, "pool_polarity": pool_pol, "evidence_polarity": ev_pol,
            "polarity_gap": abs(pol - pool_pol),
            "evidence_gap": abs(pol - ev_pol),
            "mentions_neg": 1.0 if mentions_neg else 0.0,
            "minority_lost": (0.0 if mentions_neg else 1.0) if has_minority else np.nan,
            "rougeL": rl,
        })
        if (n + 1) % 40 == 0:
            print("[%d/%d]" % (n + 1, len(items)), flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", default="results/eval")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    items = [it for it in data["data"] if it.get("output")]

    # only cases where all methods have a summary
    per_case = {}
    for it in items:
        key = (it["hotel_id"], it["aspect"])
        if key not in per_case:
            per_case[key] = set()
        per_case[key].add(it["method"])
    items = [it for it in items if set(METHODS).issubset(per_case[(it["hotel_id"], it["aspect"])])]

    tag = os.path.basename(args.input).replace(".json", "").replace("summaries_", "")
    out = args.out_dir
    per_item = out + "/summary_per_item_" + tag + ".csv"
    if os.path.exists(per_item) and not args.redo:
        df = pd.read_csv(per_item, dtype={"hotel_id": str})
    else:
        from rouge_score import rouge_scorer
        df = evaluate(items, NLI(), Sentiment(), rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True))
        df.to_csv(per_item, index=False)

    n_cases = len(df[["hotel_id", "aspect"]].drop_duplicates())
    lines = []
    lines.append("# Summary level evaluation (" + data.get("model", tag) + ")")
    lines.append("")
    lines.append("%d hotel-aspect cases x %d methods. NLI: %s. Sentiment: %s. "
                 "Mean with 95%% bootstrap CI over the cases." % (n_cases, df.method.nunique(), NLI_MODEL, SENT_MODEL))
    lines.append("")
    lines.append("| Method | " + " | ".join(LABELS[m] for m in METRICS) + " |")
    lines.append("|---|" + "---|" * len(METRICS))
    for m in METHODS:
        sub = df[df.method == m]
        cells = []
        for k in METRICS:
            vals = sub[k].to_numpy(dtype=float)
            if np.all(np.isnan(vals)):
                cells.append("-")
            elif k == "words":
                cells.append("%.0f" % np.mean(vals))
            else:
                mean, lo, hi = bootstrap_ci(vals)
                cells.append("%.3f [%.3f, %.3f]" % (mean, lo, hi))
        lines.append("| " + m + " | " + " | ".join(cells) + " |")

    tests = []
    for k in METRICS:
        piv = df.pivot_table(index=["hotel_id", "aspect"], columns="method", values=k, dropna=False)
        block = []
        for b in ["Top-K", "MMR", "Random"]:
            if piv[b].isna().all():
                continue
            t = paired_test(piv["Proportional"].to_numpy(), piv[b].to_numpy())
            t.update({"metric": k, "a": "Proportional", "b": b})
            block.append(t)
        if len(block) > 0:
            adj = holm(np.array([t["p"] for t in block]))
            for i in range(len(block)):
                block[i]["p_holm"] = float(adj[i])
            tests += block
    pd.DataFrame(tests).to_csv(out + "/summary_paired_tests_" + tag + ".csv", index=False)

    lines.append("")
    lines.append("## Proportional minus the other method (paired)")
    lines.append("")
    lines.append("| Metric | vs | Mean diff [95% CI] | Holm p |")
    lines.append("|---|---|---|---|")
    for t in tests:
        if t["n"] == 0:
            continue
        lines.append("| %s | %s | %+.3f [%+.3f, %+.3f] | %.3g |" % (
            LABELS[t["metric"]], t["b"], t["mean_diff"], t["ci_low"], t["ci_high"], t["p_holm"]))
    with open(out + "/summary_report_" + tag + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
