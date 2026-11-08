# Evaluation (William)

Everything is run from the repo root, after the SPACE pipeline in `src/space/`.

## 0. Setup

```bash
pip install -r requirements.txt
mkdir -p data/space
curl -L -o data/space/space_summ.json https://raw.githubusercontent.com/stangelid/qt/main/data/json/space_summ.json
```

## 1. SPACE pipeline (Nazmus code, not changed)

```bash
python -m src.space.prepare_dataset
python -m src.space.embed_reviews
python -m src.space.retrieve_reviews
python -m src.space.top_k
python -m src.space.mmr
python -m src.space.proportional
python -m src.space.evaluate_selection
```

## 2. Unit tests

```bash
python -m pytest -q          # 26 tests, no model download needed
```

The tests checks the hand example (K = 4, k = 2), that the greedy score is the same as the exact gain of HPF(R),
that lambda = 0 gives Top-K and that greedy is within 10% of the brute force optimum. They also check that the
selectors in `src/space/` picks the same reviews as our own versions in `src/eval/core.py`, that they return k
unique reviews, is deterministic, dont use the ratings and works when the pool is smaller then k. Last the metrics
(JS is 0 and 1 in the extreme cases, the oracle is a lower bound, minority loss).

## 3. Selection level statistics (Random, oracle, ablation)

```bash
python -m src.eval.selection_stats
# results/eval/report.md, summary.csv, paired_tests.csv, per_case.csv, ablation_proportional.csv
```

## 4. Summaries and evaluation of the answers

```bash
# CPU pilot, around 2.5 h for 400 summaries. This model almost never cites so citation metrics is mostly empty
python -m src.eval.generate_summaries --model Qwen/Qwen2.5-1.5B-Instruct
# GPU / Colab, the model we freezed in the project plan, see notebooks/colab_summaries.ipynb
python -m src.eval.generate_summaries --model Qwen/Qwen2.5-3B-Instruct

python -m src.eval.summary_eval --input results/eval/summaries_Qwen2.5-1.5B-Instruct.json
# results/eval/summary_report_<model>.md, summary_per_item_<model>.csv, summary_paired_tests_<model>.csv
```

Metrics: if the summary has citations and if they are valid, NLI faithfulness (cross-encoder/nli-deberta-v3-base),
citation precision, polarity gap to the pool and to the evidence (cardiffnlp/twitter-roberta-base-sentiment-latest),
if a complaint is mentioned, minority loss and ROUGE-L against the SPACE reference summaries.

### Pilot result (Qwen2.5-1.5B on CPU, 87 complete cases, 8 Nov 2026)

See `results/eval/summary_report_Qwen2.5-1.5B-Instruct.md`. The run stopped at 350 of 400 summaries because of a
time limit, so we only compare the cases where all four methods have a summary.

* Polarity gap to the pool: Proportional 0.274, Top-K 0.321, MMR 0.380, Random 0.279. Proportional is significant
  closer than MMR (-0.106, Holm p = 0.01) but not significant closer than Top-K (p = 0.72), and the same as Random.
* Faithfulness, citations, minority loss and ROUGE-L: no significant differences between the methods.
* The 1.5B model only cites in around 40% of the summaries so citation precision is based on few cases. We should use Qwen2.5-3B on Colab.

Note about the NLI numbers: a sentence like "Guests generally praised the location" is checked against every review
one by one, and no single review entails "guests generally". So the absolute faithfulness (around 15%) is way too low
and only the comparison between methods means something. Better options for the final run is NLI against all the
cited reviews together, splitting sentences into smaller claims, or a LLM judge that we check on 50 sentences we label by hand.

## 5. User study

```bash
python -m src.eval.build_user_study --summaries results/eval/summaries_Qwen2.5-3B-Instruct.json
# send results/user_study/user_study.html to 5-10 evaluators, keep key.json private
# put their csv files in results/user_study/answers/ and run
python -m src.eval.analyze_user_study --key results/user_study/key.json --answers results/user_study/answers/*.csv
```

10 hotel-aspect cases (5 cleanliness, 5 service, different hotels). Top-K, MMR and Proportional is shown as
A/B/C in random order for every case and gets a score 1-10 for representativeness, informativeness and trust.
Analysis: mean with bootstrap CI, Friedman test and Wilcoxon with Holm correction (Proportional vs the other ones).
