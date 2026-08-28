"""Dynamic-K speculative decode with optional skip layers and fallback."""
from __future__ import annotations

import math
import time
from typing import Any

import torch

from spec_exp.self_spec_decode import (
    DecodeItem,
    build_baseline_texts,
    decode_one_step,
    greedy_generate_full,
    make_verify_model,
    next_token,
)
from spec_exp.task_score import mean_task_score_detailed


def _adapt_k(current_k: int, accept_rate: float, *, min_k: int = 1, max_k: int = 8) -> int:
    if accept_rate >= 0.80 and current_k < max_k:
        return current_k + 1
    if accept_rate <= 0.50 and current_k > min_k:
        return current_k - 1
    return current_k


def greedy_generate_dynamic_k(
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    init_k: int,
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    skip_layers: set[int],
    device: str,
    min_k: int = 1,
    max_k: int = 8,
) -> tuple[list[int], list[dict[str, Any]], list[int]]:
    verify_model = make_verify_model(target_model, skip_layers)
    ids = list(prompt_ids)
    traces: list[dict[str, Any]] = []
    k_history: list[int] = []
    generated = 0
    current_k = init_k
    recent_accepts: list[float] = []

    while generated < max_new_tokens:
        k_history.append(current_k)
        new_ids, trace = decode_one_step(
            ids=ids,
            max_new_tokens=max_new_tokens - generated,
            k=current_k,
            draft_model=draft_model,
            verify_model=verify_model,
            device=device,
        )
        drafted = trace["drafted_len"]
        accepted = trace["accepted_len"]
        step_accept = accepted / drafted if drafted else 0.0
        recent_accepts.append(step_accept)
        if len(recent_accepts) > 5:
            recent_accepts.pop(0)
        avg_accept = sum(recent_accepts) / len(recent_accepts)
        current_k = _adapt_k(current_k, avg_accept, min_k=min_k, max_k=max_k)

        emitted = len(new_ids) - len(ids)
        ids = new_ids
        generated += emitted
        traces.append(trace)
        if emitted == 0:
            break
    return ids, traces, k_history


def evaluate_dynamic_k(
    *,
    items: list[DecodeItem],
    tokenizer: Any,
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    skip_layers: set[int],
    init_k: int,
    device: str,
    domain: str,
    baseline_texts: dict[str, str] | None = None,
    quality_floor: float | None = None,
    vanilla_texts: dict[str, str] | None = None,
) -> dict[str, float]:
    all_traces: list[dict[str, Any]] = []
    hypotheses: dict[str, str] = {}
    references: dict[str, str | None] = {}
    questions: dict[str, str] = {}
    latencies: list[float] = []
    fallback_count = 0
    t0 = time.perf_counter()

    for item in items:
        prompt_ids = tokenizer(item.prompt, add_special_tokens=True).input_ids
        req_t0 = time.perf_counter()
        out_ids, traces, _ = greedy_generate_dynamic_k(
            prompt_ids=prompt_ids,
            max_new_tokens=item.max_tokens,
            init_k=init_k,
            draft_model=draft_model,
            target_model=target_model,
            skip_layers=skip_layers,
            device=device,
        )
        gen_ids = out_ids[len(prompt_ids) :]
        hyp = tokenizer.decode(gen_ids, skip_special_tokens=True)
        references[item.request_id] = item.reference

        if quality_floor is not None and vanilla_texts:
            from spec_exp.task_score import score_generation

            score = score_generation(
                category=domain,
                hypothesis=hyp,
                reference=item.reference,
                request_id=item.request_id,
                question=item.prompt,
            )
            if score < quality_floor:
                fallback_count += 1
                full_ids = greedy_generate_full(
                    prompt_ids=prompt_ids,
                    max_new_tokens=item.max_tokens,
                    target_model=target_model,
                    device=device,
                )
                hyp = tokenizer.decode(full_ids[len(prompt_ids) :], skip_special_tokens=True)

        hypotheses[item.request_id] = hyp
        questions[item.request_id] = item.prompt
        latencies.append(time.perf_counter() - req_t0)
        all_traces.extend(traces)

    wall = time.perf_counter() - t0
    drafted = sum(int(t["drafted_len"]) for t in all_traces)
    accepted = sum(int(t["accepted_len"]) for t in all_traces)
    drafts = len(all_traces)
    total_out = sum(len(tokenizer(h, add_special_tokens=False).input_ids) for h in hypotheses.values())
    score_detail = mean_task_score_detailed(
        category=domain,
        hypotheses=hypotheses,
        references=references,
        baseline_hypotheses=baseline_texts,
        questions=questions,
    )
    task_score = float(score_detail["mean_score"])
    lat_sorted = sorted(latencies)
    p99_idx = max(0, int(math.ceil(0.99 * len(lat_sorted))) - 1)

    return {
        "accept_rate": accepted / drafted if drafted else math.nan,
        "mean_accepted_per_step": (1.0 + accepted / drafts) if drafts else 1.0,
        "num_skip_layers": len(skip_layers),
        "wall_s": wall,
        "task_score": task_score,
        "task_score_metric": score_detail.get("metric"),
        "num_draft_steps": drafts,
        "total_output_tokens": total_out,
        "tok_per_s": total_out / wall if wall > 0 else math.nan,
        "tpot_ms": (wall / total_out * 1000) if total_out > 0 else math.nan,
        "p99_lat_ms": lat_sorted[p99_idx] * 1000 if lat_sorted else math.nan,
        "fallback_pct": 100.0 * fallback_count / len(items) if items else 0.0,
        "skipped_blocks_pct": 100.0 * len(skip_layers) / 28.0,
    }


def build_vanilla_texts(
    *,
    items: list[DecodeItem],
    tokenizer: Any,
    target_model: torch.nn.Module,
    device: str,
) -> dict[str, str]:
    texts: dict[str, str] = {}
    for item in items:
        prompt_ids = tokenizer(item.prompt, add_special_tokens=True).input_ids
        out_ids = greedy_generate_full(
            prompt_ids=prompt_ids,
            max_new_tokens=item.max_tokens,
            target_model=target_model,
            device=device,
        )
        texts[item.request_id] = tokenizer.decode(out_ids[len(prompt_ids) :], skip_special_tokens=True)
    return texts
