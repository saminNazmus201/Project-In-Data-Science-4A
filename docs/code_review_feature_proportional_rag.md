# Code review of `feature/proportional-rag` (SPACE experiment)

William, 8 Nov 2026. Branch at commit `a00614e` (6 Nov).

## Summary

The SPACE pipeline works end to end and the results can be reproduced. I ran every step again from
`space_summ.json` (prepare, embed, retrieve, Top-K / MMR / Proportional, evaluate) on a clean machine.

| | JS (lower better) | Relevance | Diversity |
|---|---|---|---|
| Mid-term slides, Proportional | 0.2308 | 0.5286 | 0.3539 |
| My re-run, Proportional | 0.2303 | 0.5288 | 0.3543 |

Cleanliness is exactly the same. Service differ with max 0.003, probably because of the vector index and
different library versions. I also wrote own versions of the three selectors (`src/eval/core.py`) and they pick
exactly the same five reviews as Top-K, MMR and Proportional in 100 of 100 cases.
Unit tests: `python -m pytest -q` (26 tests, all passes).

The big problem is how we interpret the result. When I add a Random baseline and paired tests
(`src/eval/selection_stats.py`) the claim "Proportional best preserves the sentiment mix" dont hold:

* Proportional vs Top-K, JS distance: -0.019, 95% CI [-0.044, +0.004], Wilcoxon p = 0.17 (Holm 0.93).
* Proportional vs Random: -0.002 [-0.020, +0.017], p = 0.61. So Proportional is not better than picking five reviews random.
* What Proportional actually do: Top-K and MMR picks to few positive reviews compared to the pool
  (-9 and -14 percentage points, both p < 0.001). Proportional removes this negative skew (+0.7 points, same as Random).
* Proportional have the lowest semantic diversity of all methods, even lower then Random (0.354 vs 0.409, p < 0.001),
  and it loses the minority opinion (no neutral or negative review picked even if the pool has one) in 23.5% of the cases,
  compared to 17.3% for Top-K and 9.2% for MMR.
* The best JS you can get with five reviews is 0.113 on average, so there is a lot of room for a method that actually aims for the sentiment mix.

All tables is in `results/eval/report.md`.

## Findings

### 1. Reproducibility (high)
* `README.md` and `requirements.txt` is empty, so nobody else can run the code.
* The code needs **ChromaDB 0.3.23** (old `Settings(chroma_db_impl="duckdb+parquet")`, `retrieve_reviews.py:43`,
  `embed_reviews.py:43`). On Python 3.13 it also needs `pydantic<2`. I added a `requirements.txt` with this.
* No result files is in git (`data/` is ignored). I added the small outputs (`space_*_results.json`,
  `space_evaluation_*`) so the slides can be checked. The ASQA files (`comparison_*.csv`) is still missing.
* The scripts is run with `python -m src.space.<name>` from the repo root, the README should say that.

### 2. First pick in the proportional selector (medium)
In `src/space/proportional.py:99-121` the score when nothing is selected yet is
`(1-lambda)(K-k)*r + lambda*p_S` = 7.5*r + 0.5*p_S. In IAdU (SIGMOD 2021, Eq. 17) the first item is picked only on relevance.
Our way is fine (it is the exact gain of HPF(R) when R is empty) but it is not the same as the paper, so we should
either change it or write it in the report. The ablation (`results/eval/ablation_proportional.csv`) shows that with
lambda = 0.5 the difference is small (JS 0.230 vs 0.232), but with lambda 0.75 or higher the relevance first pick is clearly better.

### 3. The -2*p_R term is correct (can go in the report)
When x is added to R the sum of p_R(i) over R grows with 2 * sum_{j in R} s(x, j), one time for x and one time for
every j that is already in R. So the greedy score `(1-lambda)(K-k)*r(x) + lambda*(p_S(x) - 2*p_R(x))` is the
**exact marginal gain** of HPF(R) (Eq. 10/16). There is a test for this in
`tests/test_selectors.py::test_greedy_score_is_exact_hpf_gain`. I think we can remove the "requires further theoretical
justification" sentence from the report and write this instead (we should check it with Georgios).

### 4. Long reviews gets cut by the embedding model (medium)
all-MiniLM-L6-v2 only reads 256 tokens. 18.7% of all SPACE reviews and 21.8% of the retrieved candidates are
longer, so their embeddings (and then also relevance, p_S and p_R) only see the start of the review.
Options: embed sentences and combine them per review, use a model with longer context (for example BGE, 512 tokens)
or select sentences instead of reviews like the pools on `master` does.

### 5. The similarity measures topic, not opinion (high, method)
p_S rewards reviews that is similar to many other reviews. With cosine on MiniLM embeddings that means "typical
wording" and not "typical opinion". This explains why Proportional is the least diverse method and dont beat Random
on the sentiment mix. To make the method proportional in *opinion* the context needs some opinion signal (for example a
predicted sentiment or aspect sentiment per review, never the gold rating) in s(i, j).

### 6. Evaluation (medium)
`src/space/evaluate_selection.py` is clean and the numbers is exactly the same as mine. What was missing, and that I
have added in `src/eval/`: Random baseline, oracle lower bound, minority loss, positive bias, TVD over the stars,
bootstrap CIs, Wilcoxon and Holm correction.

### 7. Smaller things
* In MMR (`src/space/mmr.py:77`) the relevance is computed again from the embeddings, while Top-K and Proportional uses the
  stored `relevance_score`. It is the same number (Chroma cosine distance = 1 - cos) so the results agree, but one source is safer.
* `rating` is saved as float in the Chroma metadata, so code that uses it as stars has to round it.
* We have two different code bases (`master` with the shared `Selector` interface and BGE sentence pools, and this
  branch with ChromaDB and whole reviews). We need to agree on one before the final experiments.
* ASQA: Top-K takes the 5 nearest chunks directly while MMR and Proportional select from the top 20. But top 5 of a
  top 20 pool is the same five chunks, so this is not a fairness problem and the caveat can be removed.

## Next steps for the method
1. Add an opinion signal to the context similarity (finding 5) and run the ablation again.
2. Decide sentences or whole reviews (finding 4).
3. Use the IAdU first pick or write about our version (finding 2).
4. Always report Random and the paired tests next to the results.
