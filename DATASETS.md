# Datasets: download and preprocess

Paper experiments **do not ship raw datasets**. This document describes every
source, how to download it, and how TSS turns it into `DecodeItem`s.

After download, loaders in `spec_exp/benchmark_datasets.py` read from `data/`
(or `$TSS_DATA_DIR`). Model **weights** are separate (see the main README).

## One-command setup

```bash
pip install -r requirements.txt   # includes `datasets`
python scripts/download_and_preprocess_datasets.py
```

Optional flags:

```bash
python scripts/download_and_preprocess_datasets.py --skip-eagle
python scripts/download_and_preprocess_datasets.py --force
TSS_DATA_DIR=/mnt/data python scripts/download_and_preprocess_datasets.py
```

China / HF mirror (optional):

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$PWD/.hf-cache
```

## What the preprocessor writes

```
data/
  spec_bench/question.jsonl     # Spec-Bench (translation, rag, …)
  eagle/gsm8k/question.jsonl    # from EAGLE repo
  eagle/humaneval/question.jsonl
  eagle/sum/question.jsonl      # summarization
  nq_open/validation.jsonl      # Natural Questions open (QA)
  mmlu/test.jsonl               # cais/mmlu test split
  MANIFEST.json
```

Git clones `Spec-Bench-repo/` and `EAGLE-repo/` under `data/` as intermediates;
only the jsonl copies above are required at eval time.

## Paper domains → source → preprocess

| Domain | Metric | Source | Preprocess |
|--------|--------|--------|------------|
| Translation | BLEU | Spec-Bench `category=translation` | copy jsonl; take `turns[0]` as prompt, `reference` as gold |
| RAG | F1 | Spec-Bench `category=rag` | same |
| Summarization | ROUGE-L | EAGLE `eagle/data/sum/question.jsonl` | copy jsonl; prefix `Summarize:` if missing |
| QA | F1 | HuggingFace `nq_open` validation | keep `{question, reference}`; cap 5000 rows |
| MMLU | Acc. | HuggingFace `cais/mmlu` (`all`, test) | dump `{subject, question, choices, answer}` |
| GSM8K (optional) | Acc. | EAGLE `eagle/data/gsm8k/question.jsonl` | copy |
| HumanEval (optional) | pass@1 | EAGLE `eagle/data/humaneval/question.jsonl` | copy |

Official URLs:

- Spec-Bench: https://github.com/hemingkx/Spec-Bench
- EAGLE data jsonl: https://github.com/SafeAILab/EAGLE (`eagle/data/`)
- NQ-Open: https://huggingface.co/datasets/nq_open
- MMLU: https://huggingface.co/datasets/cais/mmlu

## Train / test split used in the paper

Implemented in `spec_exp/benchmark_datasets.py::load_dataset_split`:

1. Load the full domain list.
2. Shuffle with **seed = 42**.
3. Keep the first **80** items in several pipelines (`[:80]`).
4. **Train (skip selection)** = first `train_size` (paper default **16**; Llama-2-13B QA uses **32**).
5. **Held-out test** = the remainder (**64** in the paper).
6. Max new tokens **96**, temperature **0**.

Loaders wrap user text with the Vicuna / Llama-2 chat template at eval time, not
during this preprocess step (preprocess stores raw prompts / references).

## Environment variables

| Variable | Meaning |
|----------|---------|
| `TSS_DATA_DIR` | Root for `spec_bench/`, `eagle/`, `nq_open/`, `mmlu/` (default: `<repo>/data`) |
| `EAGLE_DATA_DIR` | Directory that contains `gsm8k/question.jsonl` etc. |
| `HF_HOME` | Hugging Face cache |
| `HF_ENDPOINT` | Optional mirror |

## Manual download (if git/HF is blocked)

1. Download Spec-Bench `data/spec_bench/question.jsonl` into `data/spec_bench/question.jsonl`.
2. From the EAGLE repo, copy `eagle/data/{gsm8k,humaneval,sum}/question.jsonl` into `data/eagle/...`.
3. Export NQ-Open validation to jsonl with fields `question` and `reference`.
4. Export MMLU test to jsonl with fields `subject`, `question`, `choices`, `answer` (int 0–3).

Then point `TSS_DATA_DIR` at that tree.
