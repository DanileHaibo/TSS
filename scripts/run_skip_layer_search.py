#!/usr/bin/env python3
"""Greedy skip-layer search on TARGET verify path (draft stays full Qwen3-0.6B).

Setup:
  - Draft: full Qwen3-0.6B (never skip layers)
  - Verify: Qwen3-8B target with selectively skipped decoder layers
  - Score: output vs full-target greedy baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spec_exp.io import ensure_dir, write_json
from spec_exp.self_spec_decode import DecodeItem, build_baseline_texts, evaluate_self_spec
from spec_exp.specbench import load_specbench


DOMAIN_CATEGORIES: dict[str, list[str]] = {
    "summarization": ["summarization"],
    "translation": ["translation"],
    "qa": ["qa"],
    "math_reasoning": ["math_reasoning"],
    "rag": ["rag"],
    "multi_turn": ["writing", "roleplay"],
}

OPTIMAL_K: dict[str, int] = {
    "summarization": 2,
    "translation": 2,
    "qa": 1,
    "math_reasoning": 2,
    "rag": 2,
    "multi_turn": 2,
}

# Qwen3-0.6B draft accept rates from prior domain_k_sweep (for comparison table)
VLLM_DRAFT_REF: dict[str, dict[str, float]] = {
    "summarization": {"accept_rate_pct": 82.5, "mean_accepted_per_step": 2.65, "optimal_k": 2},
    "translation": {"accept_rate_pct": 77.1, "mean_accepted_per_step": 2.54, "optimal_k": 2},
    "qa": {"accept_rate_pct": 61.8, "mean_accepted_per_step": 1.62, "optimal_k": 1},
    "math_reasoning": {"accept_rate_pct": 75.2, "mean_accepted_per_step": 2.50, "optimal_k": 2},
    "rag": {"accept_rate_pct": 66.0, "mean_accepted_per_step": 2.32, "optimal_k": 2},
    "multi_turn": {"accept_rate_pct": 56.2, "mean_accepted_per_step": 2.12, "optimal_k": 2},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target-model", default="/root/autodl-tmp/models/Qwen3-8B")
    p.add_argument("--draft-model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    p.add_argument("--dataset-path", default="/root/autodl-tmp/specdecode-system-exp/data/spec_bench/question.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--domains", nargs="+", default=["summarization", "translation", "qa"])
    p.add_argument("--train-size", type=int, default=10)
    p.add_argument("--test-size", type=int, default=6)
    p.add_argument("--output-len", type=int, default=128, help="Output tokens for test set")
    p.add_argument("--train-output-len", type=int, default=64, help="Shorter output for faster train search")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--accept-drop-tol", type=float, default=0.05, help="Max allowed accept_rate drop per layer removal")
    p.add_argument("--score-drop-tol", type=float, default=0.05, help="Max allowed task_score drop vs init (skip=0)")
    p.add_argument("--max-rounds", type=int, default=12)
    p.add_argument("--max-skip-layers", type=int, default=12)
    p.add_argument(
        "--candidate-layer-step",
        type=int,
        default=2,
        help="Only try every N-th layer index during search (full 0..L-1 still allowed once skipped)",
    )
    return p.parse_args()


def ensure_env() -> None:
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf-cache")


def load_domain_split(
    dataset_path: str,
    categories: list[str],
    *,
    train_size: int,
    test_size: int,
    output_len: int,
    seed: int,
) -> tuple[list[DecodeItem], list[DecodeItem]]:
    all_items = []
    for cat in categories:
        rows = load_specbench(
            dataset_path,
            output_len=output_len,
            category=cat,
            num_requests=None,
            seed=seed,
            shuffle=True,
        )
        all_items.extend(rows)
    if len(all_items) < train_size + test_size:
        raise ValueError(f"Not enough prompts: have {len(all_items)}, need {train_size + test_size}")

    train_rows = all_items[:train_size]
    test_rows = all_items[train_size : train_size + test_size]
    train = [
        DecodeItem(r.request_id, r.prompt, r.max_tokens, category=r.category, reference=r.reference)
        for r in train_rows
    ]
    test = [
        DecodeItem(r.request_id, r.prompt, r.max_tokens, category=r.category, reference=r.reference)
        for r in test_rows
    ]
    return train, test


def greedy_skip_search(
    *,
    train_items: list[DecodeItem],
    test_items: list[DecodeItem],
    tokenizer: Any,
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    domain: str,
    k: int,
    device: str,
    accept_drop_tol: float,
    score_drop_tol: float,
    max_rounds: int,
    max_skip_layers: int,
    num_layers: int,
    vllm_ref_accept: float,
    candidate_layer_step: int,
) -> dict[str, Any]:
    print("[INFO] Building skip=0 baseline texts (for QA w/o gold ref)...", flush=True)
    train_baseline_texts = build_baseline_texts(
        items=train_items,
        tokenizer=tokenizer,
        draft_model=draft_model,
        target_model=target_model,
        k=k,
        device=device,
    )
    test_baseline_texts = build_baseline_texts(
        items=test_items,
        tokenizer=tokenizer,
        draft_model=draft_model,
        target_model=target_model,
        k=k,
        device=device,
    )

    skip_layers: set[int] = set()
    history: list[dict[str, Any]] = []

    current_train = evaluate_self_spec(
        items=train_items,
        tokenizer=tokenizer,
        draft_model=draft_model,
        target_model=target_model,
        skip_layers=skip_layers,
        k=k,
        device=device,
        domain=domain,
        baseline_texts=train_baseline_texts,
    )
    current_test = evaluate_self_spec(
        items=test_items,
        tokenizer=tokenizer,
        draft_model=draft_model,
        target_model=target_model,
        skip_layers=skip_layers,
        k=k,
        device=device,
        domain=domain,
        baseline_texts=test_baseline_texts,
    )
    init_train_task_score = current_train["task_score"]
    init_test_task_score = current_test["task_score"]
    history.append(
        {
            "round": 0,
            "action": "init",
            "layer": None,
            "skip_layers": sorted(skip_layers),
            "train_accept_rate": current_train["accept_rate"],
            "train_task_score": current_train["task_score"],
            "test_accept_rate": current_test["accept_rate"],
            "test_task_score": current_test["task_score"],
        }
    )
    print(
        f"  [init] skip=0 train_accept={current_train['accept_rate']:.3f} "
        f"train_score={current_train['task_score']:.3f} "
        f"test_accept={current_test['accept_rate']:.3f} test_score={current_test['task_score']:.3f}",
        flush=True,
    )

    for round_id in range(1, max_rounds + 1):
        if len(skip_layers) >= max_skip_layers:
            print(f"  [round {round_id}] reached max_skip_layers={max_skip_layers}, stop", flush=True)
            break

        best_layer = None
        best_train = None
        best_accept = -1.0

        for layer in range(num_layers):
            if layer in skip_layers:
                continue
            if layer % candidate_layer_step != 0:
                continue
            trial_skip = set(skip_layers)
            trial_skip.add(layer)
            trial_train = evaluate_self_spec(
                items=train_items,
                tokenizer=tokenizer,
                draft_model=draft_model,
                target_model=target_model,
                skip_layers=trial_skip,
                k=k,
                device=device,
                domain=domain,
                baseline_texts=train_baseline_texts,
            )

            accept_ok = trial_train["accept_rate"] >= current_train["accept_rate"] - accept_drop_tol
            score_ok = trial_train["task_score"] >= init_train_task_score - score_drop_tol
            if accept_ok and score_ok and trial_train["accept_rate"] >= best_accept:
                best_accept = trial_train["accept_rate"]
                best_layer = layer
                best_train = trial_train

        if best_layer is None:
            print(f"  [round {round_id}] no layer can be skipped within tolerance, stop", flush=True)
            break

        skip_layers.add(best_layer)
        current_train = best_train
        current_test = evaluate_self_spec(
            items=test_items,
            tokenizer=tokenizer,
            draft_model=draft_model,
            target_model=target_model,
            skip_layers=skip_layers,
            k=k,
            device=device,
            domain=domain,
            baseline_texts=test_baseline_texts,
        )
        history.append(
            {
                "round": round_id,
                "action": "skip_layer",
                "layer": best_layer,
                "skip_layers": sorted(skip_layers),
                "train_accept_rate": current_train["accept_rate"],
                "train_task_score": current_train["task_score"],
                "test_accept_rate": current_test["accept_rate"],
                "test_task_score": current_test["task_score"],
                "accept_delta": current_train["accept_rate"] - history[-1]["train_accept_rate"],
                "task_score_delta": current_test["task_score"] - init_test_task_score,
            }
        )
        beats_vllm = "yes" if current_test["accept_rate"] >= vllm_ref_accept else "no"
        print(
            f"  [round {round_id}] skip layer {best_layer} -> "
            f"train_accept={current_train['accept_rate']:.3f} "
            f"test_accept={current_test['accept_rate']:.3f} (vllm_ref={vllm_ref_accept:.3f}, beat={beats_vllm}) "
            f"test_score={current_test['task_score']:.3f} (init={init_test_task_score:.3f})",
            flush=True,
        )

    return {
        "skip_layers": sorted(skip_layers),
        "num_skip_layers": len(skip_layers),
        "train_metrics": current_train,
        "test_metrics": current_test,
        "history": history,
    }


def build_report(domain_results: list[dict[str, Any]], out_dir: Path) -> str:
    lines = [
        "# Skip-Layer on TARGET Verify (Draft = full Qwen3-0.6B)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "- **Draft**: Qwen3-0.6B full (no layer skip)",
        "- **Verify**: Qwen3-8B target with skipped decoder layers",
        "- Greedy: skip TARGET verify layers; score = domain task metric vs SpecBench reference.",
        "- summarization/rag: ROUGE-L; translation: BLEU-4; math_reasoning: GSM8K EM; qa: ROUGE-L vs skip=0 baseline.",
        "",
        "## Summary",
        "",
        "| Domain | K | Target Skip | Test Accept % | Mean Acc/Step | Task Score | Init Score |",
        "|--------|---|-------------|---------------|---------------|------------|------------|",
    ]
    for r in domain_results:
        tm = r["test_metrics"]
        init_score = r["history"][0]["test_task_score"]
        lines.append(
            f"| {r['domain']} | {r['k']} | `{r['skip_layers']}` | "
            f"{tm['accept_rate']*100:.1f}% | {tm['mean_accepted_per_step']:.2f} | "
            f"{tm['task_score']:.3f} | {init_score:.3f} |"
        )

    lines.extend(["", "## Per-Round Search History", ""])
    for r in domain_results:
        lines.append(f"### {r['domain']}")
        lines.append("")
        lines.append("| Round | Layer | Skip Set | Train Accept | Test Accept | Test Task Score |")
        lines.append("|-------|-------|----------|--------------|-------------|-----------------|")
        for h in r["history"]:
            lines.append(
                f"| {h['round']} | {h.get('layer', '-')} | {len(h['skip_layers'])} layers | "
                f"{h['train_accept_rate']*100:.1f}% | {h['test_accept_rate']*100:.1f}% | "
                f"{h['test_task_score']:.3f} |"
            )
        lines.append("")

    report = "\n".join(lines) + "\n"
    (out_dir / "SKIP_LAYER_REPORT.md").write_text(report, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    ensure_env()
    out_dir = ensure_dir(args.output_dir)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=dtype, trust_remote_code=True
    ).to(args.device).eval()
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model, torch_dtype=dtype, trust_remote_code=True
    ).to(args.device).eval()
    num_layers = int(target_model.config.num_hidden_layers)

    config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "skip_on": "target_verify_only",
        "num_hidden_layers": num_layers,
        "domains": args.domains,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "output_len": args.output_len,
        "algorithm": "greedy_one_layer_per_round_if_accept_increases",
        "reference": "vLLM-style spec decode; layer skip on target verify only",
    }
    write_json(config, out_dir / "run_config.json")

    domain_results: list[dict[str, Any]] = []

    for domain in args.domains:
        cats = DOMAIN_CATEGORIES[domain]
        k = OPTIMAL_K[domain]
        print(f"\n[DOMAIN] {domain} (K={k}, layers={num_layers})", flush=True)
        train_items, test_items = load_domain_split(
            args.dataset_path,
            cats,
            train_size=args.train_size,
            test_size=args.test_size,
            output_len=args.train_output_len,
            seed=args.seed,
        )
        # Rebuild test with full output_len for fair comparison with prior K sweep.
        test_items_full = load_domain_split(
            args.dataset_path,
            cats,
            train_size=args.train_size,
            test_size=args.test_size,
            output_len=args.output_len,
            seed=args.seed,
        )[1]

        vllm_ref = VLLM_DRAFT_REF.get(domain, {}).get("accept_rate_pct", 0.0) / 100.0
        result = greedy_skip_search(
            train_items=train_items,
            test_items=test_items_full,
            tokenizer=tokenizer,
            draft_model=draft_model,
            target_model=target_model,
            domain=domain,
            k=k,
            device=args.device,
            accept_drop_tol=args.accept_drop_tol,
            score_drop_tol=args.score_drop_tol,
            max_rounds=args.max_rounds,
            max_skip_layers=args.max_skip_layers,
            num_layers=num_layers,
            vllm_ref_accept=vllm_ref,
            candidate_layer_step=args.candidate_layer_step,
        )
        row = {
            "domain": domain,
            "k": k,
            "categories": cats,
            **result,
        }
        domain_results.append(row)
        write_json(row, out_dir / f"{domain}_result.json")
        pd.DataFrame(result["history"]).to_csv(out_dir / f"{domain}_history.csv", index=False)

    summary_rows = []
    for r in domain_results:
        tm = r["test_metrics"]
        ref = VLLM_DRAFT_REF.get(r["domain"], {})
        summary_rows.append(
            {
                "domain": r["domain"],
                "k": r["k"],
                "skip_layers": str(r["skip_layers"]),
                "num_skip_layers": r["num_skip_layers"],
                "test_accept_rate_pct": tm["accept_rate"] * 100,
                "test_mean_accepted_per_step": tm["mean_accepted_per_step"],
                "test_task_score": tm["task_score"],
                "init_test_task_score": r["history"][0]["test_task_score"],
                "vllm_draft_accept_rate_pct": ref.get("accept_rate_pct"),
                "vllm_draft_mean_accepted_per_step": ref.get("mean_accepted_per_step"),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "skip_layer_summary.csv", index=False)
    report = build_report(domain_results, out_dir)
    print("\n" + report, flush=True)
    print(f"[DONE] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
