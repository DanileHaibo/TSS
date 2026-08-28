#!/usr/bin/env python3
"""Independently re-run the selected EAGLE-7B held-out configuration."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_tss_max_toks_pipeline import load_split, run_7b_eagle_heldout
from spec_exp.transformers_compat import install_transformers_compat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO / "results" / "tss_tri_objective_v3_train16_20260714",
    )
    args = parser.parse_args()
    install_transformers_compat()
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    source = (
        args.root
        / "candidate_heldout"
        / args.dataset
        / "candidate_heldout.json"
    )
    selected = json.loads(source.read_text())["selected"]
    skip = list(selected["skip_layers"])
    output_len = 256 if args.dataset == "humaneval" else 96
    items = load_split(
        args.dataset, "test", train_size=16, seed=42, output_len=output_len
    )
    output = args.root / "final_validation" / f"{args.dataset}_7b_eagle_heldout.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = run_7b_eagle_heldout(
        args.dataset,
        items,
        skip,
        out_json=output,
        train_result={"_path": str(source), "skip_layers": skip},
    )
    payload["schema_version"] = "eagle7_tri_objective_final_validation_v1"
    payload["selection_source"] = str(source.relative_to(REPO))
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    baseline, best = payload["test_eval"]["baseline"], payload["test_eval"]["best"]
    print(
        f"[{args.dataset}] FINAL skip={skip} "
        f"score={baseline['task_score']:.4f}->{best['task_score']:.4f} "
        f"accept={baseline['mean_accepted_per_step']:.3f}->"
        f"{best['mean_accepted_per_step']:.3f} "
        f"speedup={best['tok_per_s']/baseline['tok_per_s']:.2f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
