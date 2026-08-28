# TSS: Target-Layer Skipping for Speculative Decoding

Official code and **paper experiment logs** for Target-Layer Skipping (TSS).

Repository: [https://github.com/DanileHaibo/TSS](https://github.com/DanileHaibo/TSS.git)

This repo contains:

1. **Core selection + skip code** for Vicuna-7B (EAGLE) and Llama-2-13B (SAMD Token-Recycle)
2. **Experiment logs / tables** used in the paper (main results, ablations, qualitative cases, sensitivity)

Model weights and large datasets are **not** shipped; download them separately (see below).

---

## Repository layout

```
spec_exp/                 # search algorithms + skip runtime
  sleb_skip_search.py     # max_skip_latter_search, tolerances
  tri_objective_search.py # triple-win selection
  pareto_bridge_search.py
  skip_llama_ctx.py       # Llama layer-skip context manager
scripts/                  # 7B / 13B drivers
  run_vicuna13_eagle3_skip_sweep.py   # EAGLE-7B eval + skip
  run_tss_max_toks_pipeline.py
  run_hydra_samd_skip_greedy.py       # SAMD/Hydra skip eval
  run_samd_target_skip_search.py      # 13B SAMD search
  run_eagle7_score_tol_search.py
  run_eagle7_search_method_comparison.py
configs/
  triple_win_summary.json
  published_skip_sets.json
experiment_logs/          # paper-facing logs & tables
  paper_tables/
  triple_win/
  score_tol_sensitivity/
  search_method_comparison/
  case_studies/
  ablations/
paper/                    # LaTeX fragments for tables / setup
third_party/              # eagle / samd / hydra Python (no weights)
```

---

## Models (download yourself)

| Stack | Target | Draft |
|-------|--------|-------|
| 7B | [`lmsys/vicuna-7b-v1.3`](https://huggingface.co/lmsys/vicuna-7b-v1.3) | `EAGLE-Vicuna-7B-v1.3` |
| 13B | [`meta-llama/Llama-2-13b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | SAMD Token-Recycle (`token_recycle_4_15.json`) |

Default local paths in scripts point under `/root/autodl-tmp/models` or HF cache; edit presets in
`scripts/run_vicuna13_eagle3_skip_sweep.py` and `scripts/run_hydra_samd_skip_greedy.py` as needed.

---

## Hardware / protocol (paper)

- **GPU**: NVIDIA RTX 4090 48GB (reported in paper)
- **Split**: train 16 / held-out 64, seed 42, max new tokens 96, temperature 0
- **Search**: `max_skip_latter`, beam 3, early barrier 2, accept hard ≥ native; score drop tol 5% (published)

Published skip sets: `configs/published_skip_sets.json`.

---

## Reproduce main tables (high level)

```bash
# 1) Install deps (PyTorch + transformers + eagle/samd stacks as in your env)
pip install -r requirements.txt

# 2) 7B EAGLE TSS search / held-out (example)
python scripts/run_tss_max_toks_pipeline.py \
  --preset vicuna7 --dataset translation \
  --train-size 16 --output-len 96 --seed 42

# 3) 13B SAMD TSS search (example)
python scripts/run_samd_target_skip_search.py \
  --target llama2_13b --dataset translation \
  --train-size 16 --max-skip-layers 6

# 4) Score-tolerance sensitivity (7B)
python scripts/run_eagle7_score_tol_search.py \
  --datasets translation,qa,rag,mmlu --train-size 8

# 5) Search-method comparison (7B)
python scripts/run_eagle7_search_method_comparison.py
```

Exact CLI flags may vary by script; see each file’s `argparse` help.
Paper numbers corresponding to these runs are under `experiment_logs/`.

---

## Experiment logs (paper)

| Folder | Content |
|--------|---------|
| `experiment_logs/triple_win/` | Final triple-win summary (7B/13B) |
| `experiment_logs/paper_tables/` | Main / ablation tables (csv/json/md/tex) |
| `experiment_logs/score_tol_sensitivity/` | Score-tol search tables + per-domain search JSON + run log |
| `experiment_logs/search_method_comparison/` | SLEB / Random / Greedy / Accept / Metric vs TSS |
| `experiment_logs/case_studies/` | Native vs TSS qualitative cases + raw held-out hypotheses |
| `experiment_logs/ablations/` | Accept / reject / verify mechanism summaries |
| `paper/` | LaTeX for setup + qualitative cases + main tables |

---

## Citation

If you use this code or logs, please cite the TSS paper (to appear).
