#!/usr/bin/env python3
"""Greedy target-layer skip for linear draft_model pairs (HF verify path)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spec_exp.benchmark_config import OFFICIAL_K, SCORE_CATEGORY
from spec_exp.benchmark_datasets import load_dataset_items
from spec_exp.dynamic_k_decode import build_vanilla_texts
from spec_exp.io import ensure_dir, write_json
from spec_exp.self_spec_decode import DecodeItem, evaluate_self_spec
from spec_exp.sleb_skip_search import SlebSearchConfig, sleb_layer_search

PRESETS: dict[str, dict[str, Any]] = {
    "qwen3": {
        "target": "/root/autodl-tmp/models/Qwen3-8B",
        "draft": "/root/autodl-tmp/models/Qwen3-0.6B",
        "chat_template": "qwen",
        "vllm_target": "/root/autodl-tmp/models/Qwen3-8B",
        "vllm_draft": "/root/autodl-tmp/models/Qwen3-0.6B",
    },
    "llama31": {
        "target": "/root/autodl-tmp/models/Llama-3.1-8B-Instruct",
        "draft": "/root/autodl-tmp/models/Llama-3.2-1B-Instruct",
        "chat_template": "llama3",
        "vllm_target": "/root/autodl-tmp/models/Llama-3.1-8B-Instruct",
        "vllm_draft": "/root/autodl-tmp/models/Llama-3.2-1B-Instruct",
    },
    "qwen25": {
        "target": "/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
        "draft": "/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct",
        "chat_template": "qwen",
        "vllm_target": "/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
        "vllm_draft": "/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct-vllm-draft",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", required=True, choices=sorted(PRESETS))
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-requests", type=int, default=8)
    p.add_argument("--output-len", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rounds", type=int, default=12)
    p.add_argument("--max-skip-layers", type=int, default=12)
    p.add_argument("--layer-step", type=int, default=2)
    p.add_argument("--accept-drop-tol", type=float, default=0.05)
    p.add_argument("--score-drop-tol", type=float, default=0.10)
    p.add_argument(
        "--score-tol-mode",
        choices=["absolute", "relative"],
        default="relative",
    )
    p.add_argument(
        "--pick-metric",
        choices=["accept", "tok_per_s", "mean_accepted"],
        default="tok_per_s",
    )
    p.add_argument(
        "--accept-mode",
        choices=["improve", "baseline"],
        default="baseline",
        help="legacy only: improve vs baseline accept constraint",
    )
    p.add_argument("--search-mode", choices=["sleb", "legacy"], default="sleb")
    p.add_argument("--early-barrier", type=int, default=1)
    p.add_argument("--latter-barrier", type=int, default=1)
    p.add_argument(
        "--accept-metric",
        choices=["mean_accepted_per_step", "accept_rate"],
        default="mean_accepted_per_step",
    )
    p.add_argument("--accept-weight", type=float, default=1.0)
    p.add_argument("--score-weight", type=float, default=1.0)
    p.add_argument("--with-vllm-baseline", action="store_true")
    return p.parse_args()


def ensure_env() -> None:
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def wrap_llama3_prompt(user_text: str, tokenizer: Any) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _extract_user_text(prompt: str) -> str:
    if "<|im_start|>user" in prompt:
        return prompt.split("<|im_start|>user\n", 1)[1].split("\n<|im_start|>assistant")[0]
    return prompt


def load_items(
    preset: str,
    dataset: str,
    *,
    num_requests: int,
    seed: int,
    output_len: int,
    tokenizer: Any | None = None,
) -> list[DecodeItem]:
    chat = PRESETS[preset]["chat_template"]
    if chat == "qwen":
        if dataset in ("humaneval", "mt_bench"):
            return load_dataset_items(
                dataset, num_requests=num_requests, seed=seed, output_len=output_len, prompt_style="plain"
            )
        return load_dataset_items(
            dataset, num_requests=num_requests, seed=seed, output_len=output_len, prompt_style="qwen"
        )
    if dataset in ("humaneval", "mt_bench"):
        return load_dataset_items(
            dataset, num_requests=num_requests, seed=seed, output_len=output_len, prompt_style="plain"
        )
    base = load_dataset_items(
        dataset, num_requests=num_requests, seed=seed, output_len=output_len, prompt_style="qwen"
    )
    return [
        DecodeItem(
            request_id=item.request_id,
            prompt=wrap_llama3_prompt(_extract_user_text(item.prompt), tokenizer),
            max_tokens=item.max_tokens,
            category=item.category,
            reference=item.reference,
        )
        for item in base
    ]


def eval_with_quality(
    *,
    items: list[DecodeItem],
    tokenizer: Any,
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    skip_layers: set[int],
    k: int,
    domain: str,
    quality_baseline: dict[str, str],
) -> dict[str, Any]:
    metrics = evaluate_self_spec(
        items=items,
        tokenizer=tokenizer,
        draft_model=draft_model,
        target_model=target_model,
        skip_layers=skip_layers,
        k=k,
        device="cuda",
        domain=domain,
        baseline_texts=quality_baseline,
    )
    metrics["task_score_vs_vanilla"] = metrics["task_score"]
    return metrics


def run_vllm_baseline(*, preset: str, items: list[DecodeItem], k: int) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    cfg = PRESETS[preset]
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    llm = LLM(
        model=cfg["vllm_target"],
        dtype="auto",
        gpu_memory_utilization=0.88,
        max_model_len=8192,
        trust_remote_code=True,
        speculative_config={"model": cfg["vllm_draft"], "num_speculative_tokens": k},
    )
    total_out = 0
    import time

    t0 = time.perf_counter()
    for item in items:
        out = llm.generate([item.prompt], SamplingParams(temperature=0.0, max_tokens=item.max_tokens), use_tqdm=False)
        total_out += len(out[0].outputs[0].token_ids)
    wall = time.perf_counter() - t0
    metrics: dict[str, Any] = {"tok_per_s": total_out / wall if wall else 0, "wall_s": wall}
    try:
        drafts = accepted = None
        for m in llm.get_metrics():
            if m.name == "vllm:spec_decode_num_drafts":
                drafts = float(m.value)
            elif m.name == "vllm:spec_decode_num_draft_tokens":
                pass
            elif m.name == "vllm:spec_decode_num_accepted_tokens":
                accepted = float(m.value)
        if drafts and accepted is not None and drafts > 0:
            metrics["mean_accepted_per_step"] = 1.0 + accepted / drafts
    except Exception:
        pass
    del llm
    torch.cuda.empty_cache()
    return metrics


def _score_ok(score: float, init_score: float, tol: float, mode: str) -> bool:
    import math

    if math.isnan(score):
        return False
    if math.isnan(init_score):
        return True
    if mode == "relative":
        if abs(init_score) < 1e-9:
            return score >= init_score - tol
        return score >= init_score * (1.0 - tol)
    return score >= init_score - tol


def _pick_value(metrics: dict[str, Any], pick_metric: str) -> float:
    if pick_metric == "tok_per_s":
        return float(metrics.get("tok_per_s") or -1.0)
    if pick_metric == "mean_accepted":
        return float(metrics.get("mean_accepted_per_step") or -1.0)
    return float(metrics.get("accept_rate") or -1.0)


def _accept_ok(
    metrics: dict[str, Any],
    *,
    init_accept: float,
    current_accept: float,
    mode: str,
    accept_drop_tol: float,
) -> bool:
    accept = float(metrics.get("accept_rate") or float("nan"))
    if accept != accept:
        return False
    if mode == "baseline":
        return accept >= init_accept * (1.0 - accept_drop_tol)
    return accept > current_accept


def greedy_search(
    *,
    items: list[DecodeItem],
    tokenizer: Any,
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    domain: str,
    k: int,
    num_layers: int,
    layer_step: int,
    accept_drop_tol: float,
    accept_mode: str,
    score_drop_tol: float,
    score_tol_mode: str,
    pick_metric: str,
    max_rounds: int,
    max_skip_layers: int,
    search_mode: str = "sleb",
    early_barrier: int = 1,
    latter_barrier: int = 1,
    accept_metric: str = "mean_accepted_per_step",
    accept_weight: float = 1.0,
    score_weight: float = 1.0,
) -> dict[str, Any]:
    quality_baseline = build_vanilla_texts(
        items=items, tokenizer=tokenizer, target_model=target_model, device="cuda"
    )
    skip_layers: set[int] = set()
    current = eval_with_quality(
        items=items,
        tokenizer=tokenizer,
        draft_model=draft_model,
        target_model=target_model,
        skip_layers=skip_layers,
        k=k,
        domain=domain,
        quality_baseline=quality_baseline,
    )
    init_accept = current["accept_rate"]
    init_score = current["task_score_vs_vanilla"]
    baseline_row = dict(current)

    if search_mode == "sleb":
        sleb_cfg = SlebSearchConfig(
            num_layers=num_layers,
            max_skip_layers=max_skip_layers,
            early_barrier=early_barrier,
            latter_barrier=latter_barrier,
            accept_drop_tol=accept_drop_tol,
            score_drop_tol=score_drop_tol,
            score_tol_mode=score_tol_mode,
            accept_metric=accept_metric,
            score_key="task_score_vs_vanilla",
            accept_weight=accept_weight,
            score_weight=score_weight,
            layer_step=layer_step,
        )

        def _eval_skip(trial: set[int]) -> dict[str, Any]:
            return eval_with_quality(
                items=items,
                tokenizer=tokenizer,
                draft_model=draft_model,
                target_model=target_model,
                skip_layers=trial,
                k=k,
                domain=domain,
                quality_baseline=quality_baseline,
            )

        def _on_err(layer: int, exc: Exception) -> None:
            print(f"  [sleb] skip+{layer} failed: {exc}", flush=True)

        skip_layers, history, current = sleb_layer_search(
            eval_fn=_eval_skip,
            baseline=baseline_row,
            config=sleb_cfg,
            on_trial_error=_on_err,
        )
        return {
            "mode": "sleb",
            "search_mode": search_mode,
            "k": k,
            "score_tol_mode": score_tol_mode,
            "accept_mode": accept_mode,
            "accept_metric": accept_metric,
            "accept_weight": accept_weight,
            "score_weight": score_weight,
            "early_barrier": early_barrier,
            "latter_barrier": latter_barrier,
            "pick_metric": pick_metric,
            "accept_drop_tol": accept_drop_tol,
            "score_drop_tol": score_drop_tol,
            "baseline": baseline_row,
            "best": current,
            "history": history,
            "skip_layers": sorted(skip_layers),
        }
    history = [
        {
            "round": 0,
            "skip_layers": [],
            "accept_rate": init_accept,
            "task_score": init_score,
            "tok_per_s": current["tok_per_s"],
            "mean_accepted_per_step": current["mean_accepted_per_step"],
        }
    ]
    print(
        f"[baseline] accept={init_accept:.4f} mean_acc={current['mean_accepted_per_step']:.2f} "
        f"score={init_score:.3f} tok/s={current['tok_per_s']:.1f}",
        flush=True,
    )

    for rnd in range(1, max_rounds + 1):
        if len(skip_layers) >= max_skip_layers:
            break
        best_layer = None
        best_metrics = None
        best_pick = -1.0
        for layer in range(0, num_layers, layer_step):
            if layer in skip_layers:
                continue
            trial = set(skip_layers)
            trial.add(layer)
            try:
                m = eval_with_quality(
                    items=items,
                    tokenizer=tokenizer,
                    draft_model=draft_model,
                    target_model=target_model,
                    skip_layers=trial,
                    k=k,
                    domain=domain,
                    quality_baseline=quality_baseline,
                )
            except torch.cuda.OutOfMemoryError as exc:
                print(f"  [round {rnd}] skip+{layer} OOM: {exc}", flush=True)
                torch.cuda.empty_cache()
                continue
            torch.cuda.empty_cache()
            accept_ok = _accept_ok(
                m,
                init_accept=init_accept,
                current_accept=current["accept_rate"],
                mode=accept_mode,
                accept_drop_tol=accept_drop_tol,
            )
            score_ok = _score_ok(m["task_score_vs_vanilla"], init_score, score_drop_tol, score_tol_mode)
            if not (accept_ok and score_ok):
                continue
            pv = _pick_value(m, pick_metric)
            if best_metrics is None or pv > best_pick + 1e-12:
                best_pick = pv
                best_layer = layer
                best_metrics = m
        if best_layer is None:
            print(f"  [round {rnd}] stop", flush=True)
            break
        skip_layers.add(best_layer)
        current = best_metrics
        history.append(
            {
                "round": rnd,
                "layer": best_layer,
                "skip_layers": sorted(skip_layers),
                "accept_rate": current["accept_rate"],
                "task_score": current["task_score_vs_vanilla"],
                "tok_per_s": current["tok_per_s"],
                "mean_accepted_per_step": current["mean_accepted_per_step"],
                "delta_accept": current["accept_rate"] - init_accept,
            }
        )
        print(
            f"  [round {rnd}] +L{best_layer} skip={sorted(skip_layers)} "
            f"accept={current['accept_rate']:.4f} mean_acc={current['mean_accepted_per_step']:.2f} "
            f"tok/s={current['tok_per_s']:.1f}",
            flush=True,
        )

    return {
        "mode": "greedy",
        "search_mode": search_mode,
        "k": k,
        "score_tol_mode": score_tol_mode,
        "accept_mode": accept_mode,
        "pick_metric": pick_metric,
        "accept_drop_tol": accept_drop_tol,
        "score_drop_tol": score_drop_tol,
        "baseline": history[0],
        "best": current,
        "history": history,
        "skip_layers": sorted(skip_layers),
    }


def main() -> None:
    args = parse_args()
    ensure_env()
    cfg = PRESETS[args.preset]
    out_dir = ensure_dir(args.output_dir)
    domain = SCORE_CATEGORY[args.dataset]
    k = OFFICIAL_K[args.dataset]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.preset}: {cfg['target']} + {cfg['draft']}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["target"], trust_remote_code=True)
    items = load_items(
        args.preset,
        args.dataset,
        num_requests=args.num_requests,
        seed=args.seed,
        output_len=args.output_len,
        tokenizer=tokenizer,
    )
    target = AutoModelForCausalLM.from_pretrained(
        cfg["target"], torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()
    draft = AutoModelForCausalLM.from_pretrained(
        cfg["draft"], torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()
    num_layers = int(target.config.num_hidden_layers)

    result = greedy_search(
        items=items,
        tokenizer=tokenizer,
        draft_model=draft,
        target_model=target,
        domain=domain,
        k=k,
        num_layers=num_layers,
        layer_step=args.layer_step,
        accept_drop_tol=args.accept_drop_tol,
        accept_mode=args.accept_mode,
        score_drop_tol=args.score_drop_tol,
        score_tol_mode=args.score_tol_mode,
        pick_metric=args.pick_metric,
        max_rounds=args.max_rounds,
        max_skip_layers=args.max_skip_layers,
        search_mode=args.search_mode,
        early_barrier=args.early_barrier,
        latter_barrier=args.latter_barrier,
        accept_metric=args.accept_metric,
        accept_weight=args.accept_weight,
        score_weight=args.score_weight,
    )
    result.update(
        {
            "preset": args.preset,
            "dataset": args.dataset,
            "target_model": cfg["target"],
            "draft_model": cfg["draft"],
            "num_layers": num_layers,
            "num_requests": args.num_requests,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    if args.with_vllm_baseline:
        print("Running vLLM draft_model baseline...", flush=True)
        del target, draft
        torch.cuda.empty_cache()
        result["vllm_baseline"] = run_vllm_baseline(preset=args.preset, items=items, k=k)

    tag = f"{args.dataset}_{args.preset}_draft_skip_greedy"
    write_json(result, out_dir / f"{tag}.json")
    print(json.dumps({"preset": args.preset, "dataset": args.dataset, "best": result["best"]}, indent=2))

    del target, draft
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
