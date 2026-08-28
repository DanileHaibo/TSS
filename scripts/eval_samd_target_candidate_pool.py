#!/usr/bin/env python3
"""Evaluate a fixed pool of SAMD target skip sets with one model load."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_samd_target_skip_search import TARGETS, extract_user, render_prompt
from spec_exp.benchmark_config import SCORE_CATEGORY
from spec_exp.benchmark_datasets import load_dataset_split
from spec_exp.io import ensure_dir, write_json
from spec_exp.self_spec_decode import DecodeItem
from spec_exp.transformers_compat import install_transformers_compat


def three_win(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return all(
        float(candidate[key]) >= float(baseline[key])
        for key in ("task_score", "mean_accepted_per_step", "tok_per_s")
    )


def tri_ratio(candidate: dict[str, Any], baseline: dict[str, Any]) -> float:
    ratios = [
        float(candidate[key]) / float(baseline[key])
        for key in ("task_score", "mean_accepted_per_step", "tok_per_s")
    ]
    return math.prod(max(value, 1e-12) for value in ratios) ** (1.0 / 3.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidates-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-len", type=int, default=96)
    parser.add_argument("--train-size", type=int, default=16)
    args = parser.parse_args()

    install_transformers_compat()
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    cfg = TARGETS[args.target]
    raw_candidates = json.loads(args.candidates_json.read_text())
    candidates = list(
        dict.fromkeys(
            tuple(sorted({int(layer) for layer in entry}))
            for entry in raw_candidates["candidates"]
            if entry
        )
    )

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

    ensure_dir(args.output.parent)
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
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        baseline_raw = eval_samd(
            model,
            items,
            set(),
            baseline_hypotheses=None,
            domain=SCORE_CATEGORY[args.dataset],
        )
        baseline_hypotheses = baseline_raw.pop("hypotheses", None)
        for index, skip in enumerate(candidates, 1):
            print(
                f"[{args.dataset}] candidate {index}/{len(candidates)} "
                f"skip={list(skip)}",
                flush=True,
            )
            candidate = eval_samd(
                model,
                items,
                set(skip),
                baseline_hypotheses=baseline_hypotheses,
                domain=SCORE_CATEGORY[args.dataset],
            )
            candidate.pop("hypotheses", None)
            row = {
                "skip_layers": list(skip),
                "candidate": candidate,
                "three_win": three_win(candidate, baseline_raw),
                "tri_geomean_ratio": tri_ratio(candidate, baseline_raw),
                "delta": {
                    key: float(candidate[key]) - float(baseline_raw[key])
                    for key in (
                        "task_score",
                        "mean_accepted_per_step",
                        "tok_per_s",
                    )
                },
            }
            results.append(row)
            write_json(
                {
                    "schema_version": "samd_candidate_pool_v1",
                    "target": args.target,
                    "dataset": args.dataset,
                    "protocol": {
                        "train_size": args.train_size,
                        "test_size": len(items),
                        "seed": 42,
                        "output_len": args.output_len,
                    },
                    "baseline": baseline_raw,
                    "results": results,
                    "complete": False,
                },
                args.output,
            )
    finally:
        del model
        import torch

        torch.cuda.empty_cache()

    ranked = sorted(
        (row for row in results if row["three_win"]),
        key=lambda row: row["tri_geomean_ratio"],
        reverse=True,
    )
    payload = {
        "schema_version": "samd_candidate_pool_v1",
        "target": args.target,
        "dataset": args.dataset,
        "protocol": {
            "train_size": args.train_size,
            "test_size": len(items),
            "seed": 42,
            "output_len": args.output_len,
        },
        "baseline": baseline_raw,
        "results": results,
        "selected": ranked[0] if ranked else None,
        "complete": True,
        "wall_s": time.perf_counter() - started,
    }
    write_json(payload, args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
