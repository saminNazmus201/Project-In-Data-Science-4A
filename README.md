 # Proportional Evidence Selection for Fair and Trustworthy RAG

Compares evidence selectors (Random, Top-k, MMR, Proportional) for RAG over
reviews. For every query (e.g. "What do guests say about the rooms at hotel X?")
we retrieve the 20 most relevant review sentences of that hotel/restaurant, each
selector picks k=5, and the same LLM writes a short summary with citations.

- **SPACE** (hotels, main benchmark): 50 hotels × 7 queries (general + building,
  cleanliness, food, location, rooms, service) = 350 queries, 3 human reference
  summaries per query, and a 1–5 rating per review (gold proportions).
- **Yelp** (restaurants, second): ~100 restaurants × 5 queries (general + food,
  service, price, ambience), star ratings and coordinates, no reference summaries.
- **ASQA** (optional, the earlier setup): ALCE pools, see the ASQA section below.

```
data/        raw data (not in git; download it yourself, see below)
pools/       shared candidate pools (in git, so everyone uses the same ones)
prompts/     reviews_default.json (SPACE/Yelp), asqa_default.json (copied from ALCE)
src/         review_data (SPACE/Yelp pools), data_loader (ASQA pools), embeddings,
             selection/, prompt_builder, generation
scripts/     run_selection.py, compare_example.py
results/     outputs (not in git); review results go to results/{pool name}/
```

All commands are run from the project root. Every path is a command-line argument,
so nothing depends on where your copy lives.

## Setup

```bash
pip install -r requirements.txt
```

## SPACE / Yelp pipeline

**Data.** SPACE: only the evaluation file is needed (5 MB), it is in the QT repo:

```bash
mkdir -p data/space
curl -L -o data/space/space_summ.json https://raw.githubusercontent.com/stangelid/qt/main/data/json/space_summ.json
```

Yelp: download the Yelp Open Dataset from https://www.yelp.com/dataset (you have
to accept their terms), extract it so that `data/yelp/yelp_academic_dataset_business.json`
and `data/yelp/yelp_academic_dataset_review.json` exist.

**1. Build the pools** (`pools/space_k20.json` is already there; ~1 min per hotel on a laptop CPU):

```bash
python -m src.review_data --dataset space
# -> pools/space_k20.json  (350 queries x 20 sentences)
python -m src.review_data --dataset yelp --city Philadelphia --n_entities 100
# -> pools/yelp_k20.json   (500 queries x 20 sentences)
```

How a pool is built: all sentences of the entity's reviews (100 reviews per
hotel; for Yelp up to 100 sampled reviews per restaurant) with 4–60 words are
embedded with BGE and ranked by cosine similarity to the retrieval query of the
aspect (`SPACE_ASPECTS` / `YELP_ASPECTS` in `src/review_data.py`). Each sentence
keeps the rating of its review. Fields per pool item: see the docstring of
`src/review_data.py` (`question`, `query`, `summaries`, `pool_rating_dist`,
`entity_rating_dist`, coordinates for Yelp, `docs`).

**2. Run selectors:**

```bash
python scripts/run_selection.py --method random --pool pools/space_k20.json
python scripts/run_selection.py --method topk   --pool pools/space_k20.json
python scripts/run_selection.py --method mmr    --pool pools/space_k20.json --lam 0.5
# -> results/space_k20/{method}_selected.json
```

**3. Inspect one example (for slides)** — star rating of every candidate, who picked it,
and the rating distribution of each selection vs. the pool:

```bash
python scripts/compare_example.py --results_dir results/space_k20 --sample_id 100597_rooms --methods random topk mmr
```

**4. Generate summaries** (the review prompt is chosen automatically; zero-shot,
it does not tell the LLM to be balanced and does not show ratings, so differences
come from the selected evidence):

```bash
python -m src.generation --input results/space_k20/topk_selected.json --dry_run
python -m src.generation --input results/space_k20/topk_selected.json --backend hf --model Qwen/Qwen2.5-3B-Instruct
# -> results/space_k20/topk_Qwen2.5-3B-Instruct_gen.json
```

Every `*_gen.json` item has `output` (summary with [i] citations), `docs`
(selected sentences with `rating`), `summaries` (references) and
`pool_rating_dist`, which is everything the evaluation metrics need.

---

# ASQA (optional)

ALCE data: download it (`ALCE-data.tar` from the ALCE repo's `download_data.sh`)
and extract it into `data/`, so that `data/asqa_eval_gtr_top100.json` exists. If
it is somewhere else, pass `--data_dir`.

## Pipeline

**1. Build the candidate pool** (already in `pools/`, so only rerun this to change it):

```bash
python -m src.data_loader --data_dir data --n 50 --pool_size 20
# -> pools/asqa_gtr_n50_k20.json
```

**2. Run selectors:**

```bash
python scripts/run_selection.py --method topk --k 5
python scripts/run_selection.py --method mmr --k 5 --lam 0.5
# -> results/{method}_selected.json
```

MMR options: `--embed_model` (default `BAAI/bge-base-en-v1.5`) and `--lam`.
Embeddings are cached in `results/emb_cache.pkl`. Use `--n 3` for a quick test.

**3. Inspect one example (for slides):**

```bash
python scripts/compare_example.py --idx 0 --methods topk mmr
```

**4. Generate answers:**

```bash
python -m src.generation --input results/topk_selected.json --dry_run      # print prompts only
python -m src.generation --input results/topk_selected.json --backend hf --model Qwen/Qwen2.5-3B-Instruct
python -m src.generation --input results/topk_selected.json --backend api --model gpt-4o-mini   # needs OPENAI_API_KEY
# -> results/topk_<model>_gen.json
```

`--backend api` works with any OpenAI-compatible endpoint (pass `--api_base`).
Defaults follow ALCE: 2-shot, seed 42, temperature 0.5, top_p 1.0, max 300 new tokens.

**5. Evaluate** with ALCE's own `eval.py`. Clone the ALCE repo *outside* this one
and follow its README for dependencies:

```bash
python eval.py --f <path>/results/topk_<model>_gen.json --citations --qa --mauve
```

## Running on Colab

```python
!git clone <our-repo-url> proportional-rag
%cd proportional-rag
!pip install -q -r requirements.txt
!python scripts/run_selection.py --method topk --k 5
!python scripts/run_selection.py --method mmr --k 5 --lam 0.5
!python -m src.generation --input results/topk_selected.json --backend hf --model Qwen/Qwen2.5-3B-Instruct
!python -m src.generation --input results/mmr_selected.json  --backend hf --model Qwen/Qwen2.5-3B-Instruct
```

`data/` isn't needed on Colab because the pool is in `pools/`. Choose a GPU runtime
(Runtime → Change runtime type). `results/` isn't in git, so download the
`*_gen.json` files (or copy them to Drive) before the session ends.

## Output format (what ALCE eval.py reads)

Every results file is `{"args": {...}, "data": [item, ...]}`. Fields per item:

| field | used by eval.py for |
|---|---|
| `output` | everything (generated answer; eval.py keeps only its first line) |
| `question` | MAUVE |
| `qa_pairs[].question`, `qa_pairs[].short_answers` | STR-EM, QA |
| `annotations[0/1].long_answer` | ROUGE-L |
| `answer` | MAUVE |
| `docs[].title`, `docs[].text` | citation recall/precision (`[i]` → `docs[i-1]`) |

`docs` holds **only the selected passages, in selection order**, matching the
[1]–[k] numbering in the prompt.

## Adding a new selector (e.g. Proportional)

1. Write `src/selection/proportional.py` with a class that subclasses `Selector`
   (`src/selection/base.py`) and implements
   `select(question: str, docs: list[dict], k: int) -> list[dict]`.
   Use `src.embeddings.Embedder` to get cached, normalized embeddings.
2. Register it in `get_selector` in `src/selection/__init__.py`.
3. Run `python scripts/run_selection.py --method proportional --k 5 --pool pools/space_k20.json`.

`question` is the item's `query` (retrieval query without the entity name) when
the pool has one, otherwise its `question`. In review pools every doc also has
`rating` (1–5), `review_id` and `score` (retrieval similarity). Selectors should
not read `rating`: it is the gold label used for evaluation.

## Windows notes

- Paths with spaces need quotes: `cd "D:\project in data science\proportional-rag"`.
- If downloading the HuggingFace model fails with `CERTIFICATE_VERIFY_FAILED`, an
  antivirus is intercepting HTTPS (e.g. Avast web shield). Turn off HTTPS scanning,
  or point `SSL_CERT_FILE` at a CA bundle that includes the Windows root certificates.
