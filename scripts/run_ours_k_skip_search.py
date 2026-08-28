#!/usr/bin/env python3
"""Joint search over K and target skip layers (accept + task_score)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spec_exp.benchmark_config import DRAFT_MODEL, K_CANDIDATES_OURS, OFFICIAL_K, TARGET_MODEL
from spec_exp.benchmark_datasets import load_dataset_items
from spec_exp.io import ensure_dir, write_json
from spec_exp.self_spec_decode import build_baseline_texts, evaluate_self_spec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-size", type=int, default=8)
    p.add_argument("--test-size", type=int, default=12)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--max-skip-layers", type=int, default=6)
    p.add_argument("--accept-drop-tol", type=float, default=0.08)
    p.add_argument("--score-drop-tol", type=float, default=0.05)
    p.add_argument("--candidate-layer-step", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def search(
    *,
    train_items,
    test_items,
    tokenizer,
    draft_model,
    target_model,
    dataset: str,
    domain: str,
    num_layers: int,
    device: str,
    accept_drop_tol: float,
    score_drop_tol: float,
    max_rounds: int,
    max_skip_layers: int,
    candidate_layer_step: int,
) -> dict:
    baseline_texts = build_baseline_texts(
        items=train_items + test_items,
        tokenizer=tokenizer,
        draft_model=draft_model,
        target_model=target_model,
        k=OFFICIAL_K.get(dataset, 2),
        device=device,
    )

    best_k = OFFICIAL_K.get(dataset, 2)
    skip_layers: set[int] = set()
    history: list[dict] = []

    def eval_split(items, k, skip):
        return evaluate_self_spec(
            items=items,
            tokenizer=tokenizer,
            draft_model=draft_model,
            target_model=target_model,
            skip_layers=skip,
            k=k,
            device=device,
            domain=domain,
            baseline_texts=baseline_texts,
        )

    current_train = eval_split(train_items, best_k, skip_layers)
    current_test = eval_split(test_items, best_k, skip_layers)
    init_score = current_test["task_score"]
    history.append(
        {
            "round": 0,
            "k": best_k,
            "skip_layers": [],
            "train_accept": current_train["accept_rate"],
            "test_accept": current_test["accept_rate"],
            "test_score": current_test["task_score"],
        }
    )

    for rnd in range(1, max_rounds + 1):
        if len(skip_layers) >= max_skip_layers:
            break
        best_layer = None
        best_k_new = best_k
        best_train = None
        best_accept = -1.0

        for k_try in K_CANDIDATES_OURS:
            for layer in range(num_layers):
                if layer in skip_layers or layer % candidate_layer_step != 0:
                    continue
                trial_skip = set(skip_layers)
                trial_skip.add(layer)
                trial = eval_split(train_items, k_try, trial_skip)
                accept_ok = trial["accept_rate"] >= current_train["accept_rate"] - accept_drop_tol
                score_ok = trial["task_score"] >= init_score - score_drop_tol
                if accept_ok and score_ok and trial["accept_rate"] >= best_accept:
                    best_accept = trial["accept_rate"]
                    best_layer = layer
                    best_k_new = k_try
                    best_train = trial

        if best_layer is None:
            break
        skip_layers.add(best_layer)
        best_k = best_k_new
        current_train = best_train
        current_test = eval_split(test_items, best_k, skip_layers)
        history.append(
            {
                "round": rnd,
                "k": best_k,
                "layer": best_layer,
                "skip_layers": sorted(skip_layers),
                "train_accept": current_train["accept_rate"],
                "test_accept": current_test["accept_rate"],
                "test_score": current_test["task_score"],
            }
        )

    return {
        "best_k": best_k,
        "skip_layers": sorted(skip_layers),
        "train_metrics": current_train,
        "test_metrics": current_test,
        "init_test_score": init_score,
        "history": history,
    }


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from spec_exp.benchmark_config import SCORE_CATEGORY

    domain = SCORE_CATEGORY[args.dataset]
    all_items = load_dataset_items(args.dataset, num_requests=args.train_size + args.test_size, seed=args.seed, output_len=args.output_len)
    train = all_items[: args.train_size]
    test = all_items[args.train_size :]

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    num_layers = int(target.config.num_hidden_layers)

    result = search(
        train_items=train,
        test_items=test,
        tokenizer=tokenizer,
        draft_model=draft,
        target_model=target,
        dataset=args.dataset,
        domain=domain,
        num_layers=num_layers,
        device="cuda",
        accept_drop_tol=args.accept_drop_tol,
        score_drop_tol=args.score_drop_tol,
        max_rounds=args.max_rounds,
        max_skip_layers=args.max_skip_layers,
        candidate_layer_step=args.candidate_layer_step,
    )
    result["dataset"] = args.dataset
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    write_json(result, out_dir / f"ours_search_{args.dataset}.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
