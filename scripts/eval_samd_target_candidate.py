#!/usr/bin/env python3
"""Evaluate one fixed SAMD target skip set on the held-out 64 examples."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_samd_target_skip_search import TARGETS, extract_user, render_prompt
from spec_exp.benchmark_config import SCORE_CATEGORY
from spec_exp.benchmark_datasets import load_dataset_split
from spec_exp.io import ensure_dir, write_json
from spec_exp.self_spec_decode import DecodeItem
from spec_exp.transformers_compat import install_transformers_compat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--skip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-len", type=int, default=96)
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--save-hypotheses", action="store_true")
    args = parser.parse_args()

    install_transformers_compat()
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    cfg = TARGETS[args.target]
    skip = {int(value) for value in args.skip.split(",") if value.strip()}

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    raw = load_dataset_split(
        args.dataset,
        split="all",
        train_size=args.train_size,
        seed=42,
        output_len=args.output_len,
    )[:80][args.train_size :]
    items = [
        DecodeItem(
            request_id=item.request_id,
            prompt=render_prompt(
                tokenizer, extract_user(item.prompt), args.target
            ),
            max_tokens=item.max_tokens,
            category=item.category,
            reference=item.reference,
        )
        for item in raw
    ]

    from scripts.run_hydra_samd_skip_greedy import eval_samd, load_samd_model

    model = load_samd_model(
        cfg["path"],
        eagle_path=None,
        size=cfg["size"],
        tree_method="token_recycle",
        use_safetensors=False if args.target == "llama2_13b" else None,
        token_tree_path=(
            "token_recycle_4_15.json" if args.target == "llama2_13b" else None
        ),
    )
    try:
        baseline_raw = eval_samd(
            model,
            items,
            set(),
            baseline_hypotheses=None,
            domain=SCORE_CATEGORY[args.dataset],
        )
        baseline_hypotheses = baseline_raw.get("hypotheses")
        if not args.save_hypotheses:
            baseline_raw.pop("hypotheses", None)
        candidate = eval_samd(
            model,
            items,
            skip,
            baseline_hypotheses=baseline_hypotheses,
            domain=SCORE_CATEGORY[args.dataset],
        )
        if not args.save_hypotheses:
            candidate.pop("hypotheses", None)
    finally:
        del model
        import torch

        torch.cuda.empty_cache()

    keys = ("task_score", "mean_accepted_per_step", "tok_per_s")
    result = {
        "target": args.target,
        "dataset": args.dataset,
        "split": "test64",
        "skip_layers": sorted(skip),
        "baseline": baseline_raw,
        "candidate": candidate,
        "three_win": all(candidate[key] >= baseline_raw[key] for key in keys),
        "delta": {
            key: candidate[key] - baseline_raw[key]
            for key in keys
        },
    }
    ensure_dir(args.output.parent)
    write_json(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
