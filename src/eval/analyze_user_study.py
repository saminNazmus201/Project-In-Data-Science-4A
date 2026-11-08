# Author: William Degrer
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

from .selection_stats import bootstrap_ci, holm, paired_test

METHODS = ["Top-K", "MMR", "Proportional"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    parser.add_argument("--answers", nargs="+", required=True)
    args = parser.parse_args()

    with open(args.key) as f:
        key_data = json.load(f)
    key = {}
    for c in key_data["cases"]:
        key[c["case"]] = c["letters"]

    frames = []
    for p in args.answers:
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["score"])
    df["method"] = [key[c][l] for c, l in zip(df["case"], df["letter"])]
    print(df.evaluator.nunique(), "evaluators,", df.case.nunique(), "cases,", len(df), "scores")
    print()

    for crit in sorted(df.criterion.unique()):
        sub = df[df.criterion == crit]
        piv = sub.pivot_table(index=["evaluator", "case"], columns="method", values="score").dropna()
        print("==", crit, "(n = %d)" % len(piv))
        for m in METHODS:
            mean, lo, hi = bootstrap_ci(piv[m].to_numpy())
            print("  %-13s %5.2f  [%.2f, %.2f]" % (m, mean, lo, hi))
        if len(piv) >= 3:
            p = friedmanchisquare(piv["Top-K"], piv["MMR"], piv["Proportional"]).pvalue
            print("  Friedman p = %.3g" % p)
            tests = []
            for b in ["Top-K", "MMR"]:
                tests.append(paired_test(piv["Proportional"].to_numpy(), piv[b].to_numpy()))
            adj = holm(np.array([t["p"] for t in tests]))
            for b, t, pa in zip(["Top-K", "MMR"], tests, adj):
                print("  Proportional - %s: %+.2f [%+.2f, %+.2f], Holm p = %.3g" % (
                    b, t["mean_diff"], t["ci_low"], t["ci_high"], pa))
        print()


if __name__ == "__main__":
    main()
