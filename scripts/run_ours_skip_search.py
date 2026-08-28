#!/usr/bin/env python3
"""Greedy skip-layer search with fixed official K (no K search)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spec_exp.benchmark_config import DRAFT_MODEL, OFFICIAL_K, TARGET_MODEL
from spec_exp.benchmark_datasets import load_dataset_items
from spec_exp.dynamic_k_decode import build_vanilla_texts
from spec_exp.io import ensure_dir, write_json
from spec_exp.self_spec_decode import evaluate_self_spec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-size", type=int, default=6)
    p.add_argument("--test-size", type=int, default=12)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--max-rounds", type=int, default=12)
    p.add_argument("--max-skip-layers", type=int, default=12)
    p.add_argument("--accept-drop-tol", type=float, default=0.10)
    p.add_argument("--score-drop-tol", type=float, default=0.08)
    p.add_argument("--candidate-layer-step", type=int, default=2)
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
    k: int,
    accept_drop_tol: float,
    score_drop_tol: float,
    max_rounds: int,
    max_skip_layers: int,
    candidate_layer_step: int,
) -> dict:
    # Quality baseline = full target AR (not skip=0 spec), fixes MT-Bench stuck at score=1.0
    quality_baseline = build_vanilla_texts(
        items=train_items + test_items,
        tokenizer=tokenizer,
        target_model=target_model,
        device=device,
    )
    qa_baseline = quality_baseline if domain == "qa" else None

    skip_layers: set[int] = set()
    history: list[dict] = []

    def eval_split(items, skip):
        return evaluate_self_spec(
            items=items,
            tokenizer=tokenizer,
            draft_model=draft_model,
            target_model=target_model,
            skip_layers=skip,
            k=k,
            device=device,
            domain=domain,
            baseline_texts=qa_baseline,
        )

    def task_score_on(items, skip):
        from spec_exp.task_score import mean_task_score

        metrics = eval_split(items, skip)
        hyps = {}
        refs = {it.request_id: it.reference for it in items}
        for it in items:
            prompt_ids = tokenizer(it.prompt, add_special_tokens=True).input_ids
            from spec_exp.self_spec_decode import greedy_generate

            out_ids, _ = greedy_generate(
                prompt_ids=prompt_ids,
                max_new_tokens=it.max_tokens,
                k=k,
                draft_model=draft_model,
                target_model=target_model,
                skip_layers=skip,
                device=device,
            )
            hyps[it.request_id] = tokenizer.decode(out_ids[len(prompt_ids) :], skip_special_tokens=True)
        score = mean_task_score(
            category=domain,
            hypotheses=hyps,
            references=refs,
            baseline_hypotheses=quality_baseline,
        )
        metrics["task_score_vs_vanilla"] = score
        return metrics

    current_train = task_score_on(train_items, skip_layers)
    current_test = task_score_on(test_items, skip_layers)
    init_train_score = current_train["task_score_vs_vanilla"]
    init_test_score = current_test["task_score_vs_vanilla"]
    history.append(
        {
            "round": 0,
            "k": k,
            "skip_layers": [],
            "train_accept": current_train["accept_rate"],
            "test_accept": current_test["accept_rate"],
            "test_score": init_test_score,
        }
    )

    for rnd in range(1, max_rounds + 1):
        if len(skip_layers) >= max_skip_layers:
            break
        best_layer = None
        best_train = None
        best_accept = -1.0

        for layer in range(num_layers):
            if layer in skip_layers or layer % candidate_layer_step != 0:
                continue
            trial_skip = set(skip_layers)
            trial_skip.add(layer)
            trial = task_score_on(train_items, trial_skip)
            accept_ok = trial["accept_rate"] >= current_train["accept_rate"] - accept_drop_tol
            score_ok = trial["task_score_vs_vanilla"] >= init_train_score - score_drop_tol
            if accept_ok and score_ok and trial["accept_rate"] >= best_accept:
                best_accept = trial["accept_rate"]
                best_layer = layer
                best_train = trial

        if best_layer is None:
            break
        skip_layers.add(best_layer)
        current_train = best_train
        current_test = task_score_on(test_items, skip_layers)
        history.append(
            {
                "round": rnd,
                "k": k,
                "layer": best_layer,
                "skip_layers": sorted(skip_layers),
                "train_accept": current_train["accept_rate"],
                "test_accept": current_test["accept_rate"],
                "test_score": current_test["task_score_vs_vanilla"],
            }
        )

    return {
        "best_k": k,
        "skip_layers": sorted(skip_layers),
        "train_metrics": current_train,
        "test_metrics": current_test,
        "init_test_score": init_test_score,
        "history": history,
    }


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from spec_exp.benchmark_config import SCORE_CATEGORY

    domain = SCORE_CATEGORY[args.dataset]
    k = OFFICIAL_K[args.dataset]
    all_items = load_dataset_items(
        args.dataset, num_requests=args.train_size + args.test_size, seed=args.seed, output_len=args.output_len
    )
    train = all_items[: args.train_size]
    test = all_items[args.train_size :]

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()

    result = search(
        train_items=train,
        test_items=test,
        tokenizer=tokenizer,
        draft_model=draft,
        target_model=target,
        dataset=args.dataset,
        domain=domain,
        num_layers=int(target.config.num_hidden_layers),
        device="cuda",
        k=k,
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
