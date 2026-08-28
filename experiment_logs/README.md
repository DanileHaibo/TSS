# Experiment logs (paper)

These artifacts support the tables/figures discussed in the TSS paper.
They are **outputs** of the selection / evaluation pipelines in `scripts/` and `spec_exp/`.

## Index

| Subdir | Paper role | Key files |
|--------|------------|-----------|
| `triple_win/` | Main triple-win skip sets & held-out metrics (7B EAGLE, 13B SAMD) | `triple_win_summary.json` |
| `paper_tables/` | Export of `paper_figure` tables (csv/json/md/tex) | `tab_main_tss.*`, `tab_search_method_comparison_7b.*`, … |
| `score_tol_sensitivity/` | Metric drop/gain sensitivity (0/10/20/+5%) real search | `sensitivity_table.*`, `search/*.json`, `logs/run.log` |
| `search_method_comparison/` | SLEB / Random / Greedy / Accept-only / Metric-only vs TSS | summary json/csv/md |
| `case_studies/` | Qualitative Native vs TSS cases | `case_gallery.*`, `tab_tss_qualitative_cases.*`, `raw/*_outputs.json` |
| `ablations/` | Accept / reject / first-verify mechanism summaries | `*_summary.json`, `verification_mechanism.json` |

## Notes

- Large PNG/PDF figures are omitted from git to keep the repo small; regenerate with the plotting scripts if needed.
- Model weights are not included.
- `case_studies/raw/` stores per-request hypotheses used to build qualitative examples.
