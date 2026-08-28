"""Speculative decoding: full draft model + skip-layer target verify."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from spec_exp.skip_layer_model import SkipLayerTargetModel
from spec_exp.task_score import mean_task_score_detailed


@dataclass(frozen=True)
class DecodeItem:
    request_id: str
    prompt: str
    max_tokens: int
    category: str = "unknown"
    reference: str | None = None


def next_token(model: torch.nn.Module, input_ids: torch.Tensor) -> int:
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[:, -1, :]
    return int(torch.argmax(logits, dim=-1).item())


def target_predictions(model: torch.nn.Module, input_ids: torch.Tensor, start_pos: int, count: int) -> list[int]:
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0]
    return [int(torch.argmax(logits[start_pos - 1 + offset], dim=-1).item()) for offset in range(count)]


def make_verify_model(target_model: torch.nn.Module, skip_layers: set[int]) -> torch.nn.Module:
    if not skip_layers:
        return target_model
    return SkipLayerTargetModel(target_model, skip_layers)


def decode_one_step(
    *,
    ids: list[int],
    max_new_tokens: int,
    k: int,
    draft_model: torch.nn.Module,
    verify_model: torch.nn.Module,
    device: str,
) -> tuple[list[int], dict[str, Any]]:
    remaining = max_new_tokens
    drafted_len = min(k, remaining)

    draft_tokens: list[int] = []
    work = list(ids)
    for _ in range(drafted_len):
        inp = torch.tensor([work], dtype=torch.long, device=device)
        tok = next_token(draft_model, inp)
        draft_tokens.append(tok)
        work.append(tok)

    verify_inp = torch.tensor([ids + draft_tokens], dtype=torch.long, device=device)
    target_next = target_predictions(verify_model, verify_inp, len(ids), drafted_len + 1)

    accepted = 0
    for draft_tok, target_tok in zip(draft_tokens, target_next[:drafted_len], strict=True):
        if draft_tok == target_tok:
            accepted += 1
        else:
            break

    if accepted == drafted_len:
        emitted = list(draft_tokens)
        if len(emitted) < remaining:
            emitted.append(target_next[drafted_len])
    else:
        emitted = draft_tokens[:accepted] + [target_next[accepted]]

    trace = {
        "drafted_len": drafted_len,
        "accepted_len": accepted,
        "emitted_len": len(emitted),
        "wasted_drafted_tokens": drafted_len - accepted,
    }
    return ids + emitted, trace


def greedy_generate(
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    k: int,
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    skip_layers: set[int],
    device: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    verify_model = make_verify_model(target_model, skip_layers)
    ids = list(prompt_ids)
    traces: list[dict[str, Any]] = []
    generated = 0
    while generated < max_new_tokens:
        new_ids, trace = decode_one_step(
            ids=ids,
            max_new_tokens=max_new_tokens - generated,
            k=k,
            draft_model=draft_model,
            verify_model=verify_model,
            device=device,
        )
        emitted = len(new_ids) - len(ids)
        ids = new_ids
        generated += emitted
        traces.append(trace)
        if emitted == 0:
            break
    return ids, traces


def greedy_generate_full(
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    target_model: torch.nn.Module,
    device: str,
) -> list[int]:
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        tok = next_token(target_model, inp)
        ids.append(tok)
    return ids


def evaluate_spec_decode(
    *,
    items: list[DecodeItem],
    tokenizer: Any,
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    skip_layers: set[int],
    k: int,
    device: str,
    domain: str,
    baseline_texts: dict[str, str] | None = None,
) -> dict[str, float]:
    all_traces: list[dict[str, Any]] = []
    hypotheses: dict[str, str] = {}
    references: dict[str, str | None] = {}
    questions: dict[str, str] = {}
    latencies: list[float] = []
    total_output_tokens = 0
    t0 = time.perf_counter()

    for item in items:
        req_t0 = time.perf_counter()
        prompt_ids = tokenizer(item.prompt, add_special_tokens=True).input_ids
        out_ids, traces = greedy_generate(
            prompt_ids=prompt_ids,
            max_new_tokens=item.max_tokens,
            k=k,
            draft_model=draft_model,
            target_model=target_model,
            skip_layers=skip_layers,
            device=device,
        )
        all_traces.extend(traces)
        gen_ids = out_ids[len(prompt_ids) :]
        total_output_tokens += len(gen_ids)
        hypotheses[item.request_id] = tokenizer.decode(gen_ids, skip_special_tokens=True)
        references[item.request_id] = item.reference
        questions[item.request_id] = item.prompt
        latencies.append(time.perf_counter() - req_t0)

    wall = time.perf_counter() - t0
    lat_sorted = sorted(latencies)
    p99_idx = max(0, int(math.ceil(0.99 * len(lat_sorted))) - 1) if lat_sorted else 0
    drafted = sum(int(t["drafted_len"]) for t in all_traces)
    accepted = sum(int(t["accepted_len"]) for t in all_traces)
    drafts = len(all_traces)
    score_detail = mean_task_score_detailed(
        category=domain,
        hypotheses=hypotheses,
        references=references,
        baseline_hypotheses=baseline_texts,
        questions=questions,
    )
    task_score = float(score_detail["mean_score"])

    return {
        "accept_rate": accepted / drafted if drafted else math.nan,
        "mean_accepted_per_step": (1.0 + accepted / drafts) if drafts else 1.0,
        "avg_accepted_draft_tokens": accepted / drafts if drafts else 0.0,
        "num_skip_layers": len(skip_layers),
        "wall_s": wall,
        "task_score": task_score,
        "task_score_metric": score_detail.get("metric"),
        "num_draft_steps": drafts,
        "total_drafted_tokens": drafted,
        "total_accepted_tokens": accepted,
        "total_output_tokens": total_output_tokens,
        "tok_per_s": total_output_tokens / wall if wall > 0 else math.nan,
        "tpot_ms": wall / total_output_tokens * 1000 if total_output_tokens else math.nan,
        "p99_lat_ms": lat_sorted[p99_idx] * 1000 if lat_sorted else math.nan,
        "skipped_blocks_pct": 100.0 * len(skip_layers) / max(
            int(getattr(getattr(target_model, "config", None), "num_hidden_layers", 28)), 1
        ),
    }


# Backward-compatible alias
evaluate_self_spec = evaluate_spec_decode


def build_baseline_texts(
    *,
    items: list[DecodeItem],
    tokenizer: Any,
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    k: int,
    device: str,
) -> dict[str, str]:
    """Full-target verify baseline (skip=0) texts for QA domains without gold refs."""
    texts: dict[str, str] = {}
    for item in items:
        prompt_ids = tokenizer(item.prompt, add_special_tokens=True).input_ids
        out_ids, _ = greedy_generate(
            prompt_ids=prompt_ids,
            max_new_tokens=item.max_tokens,
            k=k,
            draft_model=draft_model,
            target_model=target_model,
            skip_layers=set(),
            device=device,
        )
        texts[item.request_id] = tokenizer.decode(out_ids[len(prompt_ids) :], skip_special_tokens=True)
    return texts
