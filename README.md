# Proportional Evidence Selection for RAG

Compares evidence selectors (Top-k, MMR, Proportional) on ASQA from the
[ALCE benchmark](https://github.com/princeton-nlp/ALCE). Each selector picks k=5
passages from a fixed pool of the top-20 GTR-retrieved passages per question;
an LLM then writes a cited answer, scored with ALCE `eval.py`.

```
data/        ALCE data (not in git; download it yourself, see below)
pools/       shared candidate pool (in git, so everyone uses the same one)
prompts/     asqa_default.json, copied from ALCE
src/         data_loader, embeddings, selection/, prompt_builder, generation
scripts/     run_selection.py, compare_example.py
results/     outputs (not in git)
```

All commands are run from the project root. Every path is a command-line argument,
so nothing depends on where your copy lives.

## Setup

```bash
pip install -r requirements.txt
```

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
3. Run `python scripts/run_selection.py --method proportional --k 5`.

## Windows notes

- Paths with spaces need quotes: `cd "D:\project in data science\proportional-rag"`.
- If downloading the HuggingFace model fails with `CERTIFICATE_VERIFY_FAILED`, an
  antivirus is intercepting HTTPS (e.g. Avast web shield). Turn off HTTPS scanning,
  or point `SSL_CERT_FILE` at a CA bundle that includes the Windows root certificates.
