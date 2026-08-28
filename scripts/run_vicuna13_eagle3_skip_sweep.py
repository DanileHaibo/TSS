#!/usr/bin/env python3
"""EAGLE3 skip-layer sweep (greedy / preset sweep): raise accept_rate while keeping task_score."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import torch

REPO = Path(__file__).resolve().parents[1]
EAGLE_ROOT = Path("/root/autodl-tmp/eagle3-system-exp/repos/EAGLE")
EAGLE_SCRIPTS = Path("/root/autodl-tmp/eagle3-system-exp/scripts")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(EAGLE_SCRIPTS))
sys.path.insert(0, str(EAGLE_ROOT))

from spec_exp.benchmark_config import LLAMA_BASE_MODEL, LLAMA_EAGLE3_DRAFT, SCORE_CATEGORY
from spec_exp.benchmark_datasets import load_dataset_items
from spec_exp.io import ensure_dir, write_json, write_table
from spec_exp.pareto_bridge_search import (
    ParetoBridgeOptions,
    pareto_bridge_v2_search,
)
from spec_exp.self_spec_decode import DecodeItem
from spec_exp.sleb_skip_search import (
    SlebSearchConfig,
    max_accept_preserve_score_search,
    max_metrics_balanced_search,
    max_skip_latter_search,
    max_toks_preserve_quality_search,
)
from spec_exp.task_score import mean_task_score
from spec_exp.tri_objective_search import (
    TriObjectiveOptions,
    tri_objective_v3_search,
)

VICUNA13_BASE = (
    "/root/autodl-tmp/hf-cache/hub/models--lmsys--vicuna-13b-v1.3/snapshots/"
    "6566e9cb1787585d1147dcf4f9bc48f29e1328d2"
)
VICUNA13_EAGLE3 = (
    "/root/autodl-tmp/hf-cache/hub/models--yuhuili--EAGLE3-Vicuna1.3-13B/snapshots/"
    "651195736adad4c05282d140e94bfff058b1fc8b"
)

VICUNA7_BASE = "/root/autodl-tmp/models/vicuna-7b-v1.3"
VICUNA7_EAGLE = "/root/autodl-tmp/models/EAGLE-Vicuna-7B-v1.3"


def _resolve_vicuna7_base() -> str:
    cache = Path("/root/autodl-tmp/hf-cache/hub/models--lmsys--vicuna-7b-v1.3/snapshots")
    if cache.exists():
        snaps = sorted(cache.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for s in snaps:
            if (s / "config.json").exists():
                return str(s)
    if Path(VICUNA7_BASE, "config.json").exists():
        return VICUNA7_BASE
    return VICUNA7_BASE


LLAMA2_13B_BASE = "/root/autodl-tmp/models/Llama-2-13b-chat-hf"
LLAMA2_13B_EAGLE = "/root/autodl-tmp/models/EAGLE-llama2-chat-13B"

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "vicuna7": {
        "base_model": _resolve_vicuna7_base(),
        "ea_model": VICUNA7_EAGLE,
        "chat_template": "vicuna",
        "use_eagle3": False,
        "num_layers": 32,
        "preset_configs": {
            "baseline": [],
            "default_8_14_20_24": [8, 14, 20, 24],
        },
    },
    "llama2_13b": {
        "base_model": LLAMA2_13B_BASE,
        "ea_model": LLAMA2_13B_EAGLE,
        "chat_template": "llama2",
        "use_eagle3": False,
        "num_layers": 40,
        "preset_configs": {
            "baseline": [],
            "default_8_14_20_24": [8, 14, 20, 24],
            "qwen_style_step3": [3, 6, 9, 12, 15, 18, 21, 24, 27],
            "middle_12_28_step4": [12, 16, 20, 24, 28],
            "sparse_every10": [10, 20, 30],
            "early_4_16": [4, 8, 12, 16],
            "late_24_36": [24, 28, 32, 36],
            "aggressive_8layers": [4, 8, 12, 16, 20, 24, 28, 32],
        },
    },
    "llama31": {
        "base_model": LLAMA_BASE_MODEL,
        "ea_model": LLAMA_EAGLE3_DRAFT,
        "chat_template": "llama3",
        "use_eagle3": True,
        "num_layers": 32,
        "preset_configs": {
            "baseline": [],
            "default_8_14_20_24": [8, 14, 20, 24],
            "middle_8_28_step4": [8, 12, 16, 20, 24, 28],
            "sparse_every8": [8, 16, 24],
            "early_4_12": [4, 8, 12],
            "late_20_28": [20, 24, 28],
            "aggressive_7layers": [4, 8, 12, 16, 20, 24, 28],
        },
    },
    "vicuna13": {
        "base_model": VICUNA13_BASE,
        "ea_model": VICUNA13_EAGLE3,
        "chat_template": "vicuna",
        "use_eagle3": True,
        "num_layers": 40,
        "preset_configs": {
            "baseline": [],
            "default_8_16_24_32": [8, 16, 24, 32],
            "llama_scaled_8_14_20_24": [8, 14, 20, 24],
            "middle_12_28_step4": [12, 16, 20, 24, 28],
            "sparse_every10": [10, 20, 30],
            "early_4_16": [4, 8, 12, 16],
            "late_24_36": [24, 28, 32, 36],
            "aggressive_8layers": [4, 8, 12, 16, 20, 24, 28, 32],
        },
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="vicuna13")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--mode",
        choices=[
            "sweep",
            "greedy",
            "max_accept",
            "max_toks",
            "max_skip_latter",
            "max_metrics_balanced",
            "pareto_bridge_v2",
            "tri_objective_v3",
            "single",
        ],
        default="max_metrics_balanced",
    )
    p.add_argument("--skip-layers", default=None, help="For --mode single, comma-separated ids.")
    p.add_argument("--num-requests", type=int, default=16)
    p.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="If set, use load_dataset_split train (first N after shuffle) instead of num-requests.",
    )
    p.add_argument("--output-len", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-token", type=int, default=60)
    p.add_argument("--score-drop-tol", type=float, default=0.05)
    p.add_argument("--score-tol-mode", choices=["absolute", "relative"], default="absolute")
    p.add_argument("--accept-drop-tol", type=float, default=0.08)
    p.add_argument("--early-barrier", type=int, default=2)
    p.add_argument("--latter-barrier", type=int, default=0)
    p.add_argument(
        "--accept-metric",
        choices=["mean_accepted_per_step", "accept_rate"],
        default="mean_accepted_per_step",
    )
    p.add_argument("--layer-step", type=int, default=1, help="Try every N-th layer (1 = all layers).")
    p.add_argument("--max-skip-layers", type=int, default=24)
    p.add_argument(
        "--exhaustive-singles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Phase-1 sweep: evaluate every single-layer skip before greedy multi-layer",
    )
    p.add_argument("--max-rounds", type=int, default=12)
    p.add_argument("--bridge-score-drop-tol", type=float, default=0.05)
    p.add_argument(
        "--bridge-score-tol-mode",
        choices=["absolute", "relative"],
        default="relative",
    )
    p.add_argument("--pareto-beam-width", type=int, default=2)
    p.add_argument("--refine-top-k", type=int, default=4)
    p.add_argument("--refine-max-evals", type=int, default=80)
    return p.parse_args()


def wrap_chat_prompt(user_text: str, chat_template: str) -> str:
    if chat_template == "llama3":
        return (
            "<|im_start|>system\nYou are a helpful assistant.\n"
            f"<|im_start|>user\n{user_text}\n"
            "<|im_start|>assistant\n"
        )
    if chat_template == "llama2":
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("llama-2-chat")
        sys_p = (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
            "while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, "
            "dangerous, or illegal content."
        )
        conv.system_message = sys_p
        conv.append_message(conv.roles[0], user_text)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt() + " "
    from fastchat.model import get_conversation_template

    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], user_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _extract_user_text(prompt: str) -> str:
    if "<|im_start|>user" in prompt:
        return prompt.split("<|im_start|>user\n", 1)[1].split("\n<|im_start|>assistant")[0]
    return prompt


def load_items_for_preset(
    dataset: str,
    *,
    chat_template: str,
    num_requests: int,
    seed: int,
    output_len: int,
) -> list[DecodeItem]:
    if dataset in ("humaneval", "mt_bench"):
        return load_dataset_items(
            dataset, num_requests=num_requests, seed=seed, output_len=output_len, prompt_style="plain"
        )
    if chat_template == "llama3":
        return load_dataset_items(
            dataset, num_requests=num_requests, seed=seed, output_len=output_len, prompt_style="qwen"
        )
    base_items = load_dataset_items(
        dataset, num_requests=num_requests, seed=seed, output_len=output_len, prompt_style="qwen"
    )
    return [
        DecodeItem(
            request_id=item.request_id,
            prompt=wrap_chat_prompt(_extract_user_text(item.prompt), chat_template),
            max_tokens=item.max_tokens,
            category=item.category,
            reference=item.reference,
        )
        for item in base_items
    ]


@contextmanager
def eagle_skip_layers(model: Any, skip_layers: set[int]):
    if not skip_layers:
        yield
        return
    llama = model.base_model.model
    saved: list[tuple[int, Callable]] = []
    for idx, layer in enumerate(llama.layers):
        if idx not in skip_layers:
            continue
        orig = layer.forward

        def _make_passthrough(original_forward, layer_idx: int, skip_set: set[int]):
            def passthrough(self, hidden_states, *args, **kwargs):
                if layer_idx in skip_set:
                    return hidden_states, None, None
                return original_forward(hidden_states, *args, **kwargs)

            return passthrough

        layer.forward = MethodType(_make_passthrough(orig, idx, skip_layers), layer)
        saved.append((idx, orig))
    try:
        yield
    finally:
        for idx, orig in saved:
            llama.layers[idx].forward = orig


def traced_eagenerate_skip(
    model: Any,
    input_ids: torch.Tensor,
    *,
    request_id: str,
    skip_layers: set[int],
    temperature: float,
    max_new_tokens: int,
    max_length: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import (
        evaluate_posterior,
        initialize_tree,
        prepare_logits_processor,
        reset_tree_mode,
        tree_decoding,
    )

    logits_processor = prepare_logits_processor(temperature=temperature) if temperature > 1e-5 else None
    padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
    input_ids = input_ids.clone()
    model.ea_layer.reset_kv()

    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        current_length_data.zero_()
    else:
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
            model.base_model, max_length=max_length
        )
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    reset_tree_mode(model)
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = initialize_tree(
        input_ids, model, past_key_values, logits_processor
    )

    trace: list[dict[str, Any]] = []
    max_decode_len = max_length - model.ea_layer.total_tokens - 10
    eos = model.tokenizer.eos_token_id
    new_token = 0

    for step_id in range(max_decode_len):
        model.base_model.model.tree_mask = tree_mask
        draft_tokens = draft_tokens.to(input_ids.device)
        drafted_len = int(draft_tokens.shape[1])

        with torch.inference_mode():
            with eagle_skip_layers(model, skip_layers):
                target_logits, hidden_state_new, _ = tree_decoding(
                    model,
                    draft_tokens,
                    past_key_values,
                    tree_position_ids,
                    input_ids,
                    retrieve_indices,
                )

        padded_draft_tokens = torch.cat((draft_tokens, padding), dim=1)
        candidates = padded_draft_tokens[0, retrieve_indices]
        best_candidate, accept_length, sample_p = evaluate_posterior(target_logits, candidates, logits_processor)

        accepted_len = int(accept_length) + 1
        prev_input_len = int(input_ids.shape[1])
        select_indices = retrieve_indices[best_candidate, :accepted_len] + prev_input_len

        for past_key_values_data_item in past_key_values_data:
            tgt = past_key_values_data_item[..., select_indices.to(past_key_values_data_item.device), :]
            dst = past_key_values_data_item[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
            dst.copy_(tgt, non_blocking=True)
        current_length_data.fill_(prev_input_len + accepted_len)
        input_ids = torch.cat(
            [input_ids, candidates[None, best_candidate, :accepted_len].to(input_ids.device)], dim=-1
        )

        retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
        accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, :accepted_len]
        if logits_processor is not None:
            token = torch.multinomial(sample_p, 1)[None]
        else:
            token = torch.argmax(sample_p)[None, None]
        draft_input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
            accept_hidden_state_new,
            input_ids=draft_input_ids,
            head=model.base_model.lm_head,
            logits_processor=logits_processor,
        )

        new_token += accepted_len
        trace.append(
            {
                "request_id": request_id,
                "step_id": step_id,
                "drafted_len": drafted_len,
                "accepted_len": int(accept_length),
                "accepted_token_ids": [
                    int(token_id)
                    for token_id in candidates[
                        best_candidate, :accepted_len
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                    if int(token_id) >= 0
                ],
            }
        )
        if new_token >= max_new_tokens or (eos is not None and token.item() == eos):
            break

    return input_ids, trace


def load_model(*, base_model: str, ea_model: str, total_token: int, use_eagle3: bool) -> Any:
    from eagle.model.ea_model import EaModel

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": "cuda:0",
    }
    # Incomplete safetensors shards → fall back to pytorch .bin weights.
    idx = Path(base_model) / "model.safetensors.index.json"
    if idx.exists():
        import json as _json

        weight_map = _json.loads(idx.read_text()).get("weight_map", {})
        shards = {Path(base_model) / name for name in set(weight_map.values())}
        if not all(p.exists() for p in shards):
            kwargs["use_safetensors"] = False

    model = EaModel.from_pretrained(
        base_model_path=base_model,
        ea_model_path=ea_model,
        use_eagle3=use_eagle3,
        total_token=total_token,
        **kwargs,
    )
    model.eval()
    if hasattr(model, "ea_layer"):
        model.ea_layer.eval()
        model.ea_layer.gradient_checkpointing = False
    return model


def eval_skip_config(
    model: Any,
    items: list[DecodeItem],
    skip_layers: set[int],
    *,
    baseline_hypotheses: dict[str, str] | None,
    domain: str,
) -> dict[str, Any]:
    from eagle.model.utils import reset_tree_mode
    from eagle3_resource_profile_pod_v2 import clear_kv

    tokenizer = model.get_tokenizer()
    hypotheses: dict[str, str] = {}
    references: dict[str, str | None] = {}
    total_drafted = total_accepted = total_verify = total_out = 0
    t0 = time.perf_counter()

    for item in items:
        ids = tokenizer(item.prompt, return_tensors="pt").input_ids.to("cuda")
        plen = int(ids.shape[1])
        max_length = plen + item.max_tokens + 256
        clear_kv(model)
        reset_tree_mode(model)
        model.ea_layer.reset_kv()
        out_ids, trace = traced_eagenerate_skip(
            model,
            ids,
            request_id=item.request_id,
            skip_layers=skip_layers,
            temperature=0.0,
            max_new_tokens=item.max_tokens,
            max_length=max_length,
        )
        gen_ids = out_ids[0, plen:].tolist()
        hypotheses[item.request_id] = tokenizer.decode(gen_ids, skip_special_tokens=True) if gen_ids else ""
        references[item.request_id] = item.reference
        total_out += len(gen_ids)
        total_verify += len(trace)
        total_drafted += sum(int(r["drafted_len"]) for r in trace)
        total_accepted += sum(int(r["accepted_len"]) for r in trace)

    wall = time.perf_counter() - t0
    has_ref = any(r for r in references.values())
    quality = mean_task_score(
        category=domain,
        hypotheses=hypotheses,
        references=references,
        baseline_hypotheses=None if has_ref else baseline_hypotheses,
    )
    accept_rate = total_accepted / total_drafted if total_drafted else math.nan
    return {
        "skip_layers": sorted(skip_layers),
        "num_skip_layers": len(skip_layers),
        "accept_rate": accept_rate,
        "mean_accepted_per_step": 1.0 + total_accepted / max(total_verify, 1),
        "task_score": quality,
        "wall_s": wall,
        "total_output_tokens": total_out,
        "tok_per_s": total_out / wall if wall > 0 else math.nan,
        "num_verify_steps": total_verify,
    }


def max_accept_search(
    model: Any,
    items: list[DecodeItem],
    *,
    domain: str,
    num_layers: int,
    score_drop_tol: float,
    score_tol_mode: str,
    accept_metric: str,
    layer_step: int,
    max_skip_layers: int,
    exhaustive_singles: bool = True,
) -> dict[str, Any]:
    print("[max_accept] baseline (no skip)...", flush=True)
    baseline = eval_skip_config(model, items, set(), baseline_hypotheses=None, domain=domain)
    baseline_hyp = _collect_hypotheses(model, items, set())

    macfg = SlebSearchConfig(
        num_layers=num_layers,
        max_skip_layers=max_skip_layers,
        score_drop_tol=score_drop_tol,
        score_tol_mode=score_tol_mode,
        accept_metric=accept_metric,
        score_key="task_score",
        layer_step=layer_step,
        exhaustive_singles=exhaustive_singles,
    )

    def _eval_skip(skip_layers: set[int]) -> dict[str, Any]:
        try:
            return eval_skip_config(
                model, items, skip_layers, baseline_hypotheses=baseline_hyp, domain=domain
            )
        finally:
            torch.cuda.empty_cache()

    def _on_err(layer: int, exc: Exception) -> None:
        print(f"  [max_accept] skip+{layer} failed: {exc}", flush=True)
        torch.cuda.empty_cache()

    skip_set, history, current = max_accept_preserve_score_search(
        eval_fn=_eval_skip,
        baseline=baseline,
        config=macfg,
        on_trial_error=_on_err,
    )
    return {
        "mode": "max_accept",
        "selection_criterion": f"max {accept_metric} subject to task_score >= baseline",
        "accept_metric": accept_metric,
        "score_drop_tol": score_drop_tol,
        "score_tol_mode": score_tol_mode,
        "baseline": baseline,
        "best": current,
        "skip_layers": sorted(skip_set),
        "history": history,
    }


def max_toks_search(
    model: Any,
    items: list[DecodeItem],
    *,
    domain: str,
    num_layers: int,
    accept_metric: str,
    layer_step: int,
    max_skip_layers: int,
    exhaustive_singles: bool = True,
) -> dict[str, Any]:
    print("[max_toks] baseline (no skip)...", flush=True)
    baseline = eval_skip_config(model, items, set(), baseline_hypotheses=None, domain=domain)
    baseline_hyp = _collect_hypotheses(model, items, set())

    cfg = SlebSearchConfig(
        num_layers=num_layers,
        max_skip_layers=max_skip_layers,
        score_drop_tol=0.0,
        score_tol_mode="absolute",
        accept_metric=accept_metric,
        score_key="task_score",
        layer_step=layer_step,
        exhaustive_singles=exhaustive_singles,
    )

    def _eval_skip(skip_layers: set[int]) -> dict[str, Any]:
        try:
            return eval_skip_config(
                model, items, skip_layers, baseline_hypotheses=baseline_hyp, domain=domain
            )
        finally:
            torch.cuda.empty_cache()

    def _on_err(layer: int, exc: Exception) -> None:
        print(f"  [max_toks] skip+{layer} failed: {exc}", flush=True)
        torch.cuda.empty_cache()

    skip_set, history, current = max_toks_preserve_quality_search(
        eval_fn=_eval_skip,
        baseline=baseline,
        config=cfg,
        on_trial_error=_on_err,
    )
    return {
        "mode": "max_toks",
        "selection_criterion": (
            "max tok_per_s subject to accept>=baseline AND task_score>=baseline"
        ),
        "accept_metric": accept_metric,
        "baseline": baseline,
        "best": current,
        "skip_layers": sorted(skip_set),
        "history": history,
    }


def max_metrics_balanced_mode(
    model: Any,
    items: list[DecodeItem],
    *,
    domain: str,
    num_layers: int,
    accept_metric: str,
    max_skip_layers: int,
    early_barrier: int,
    latter_barrier: int,
) -> dict[str, Any]:
    print("[max_metrics_balanced] baseline (no skip)...", flush=True)
    baseline = eval_skip_config(model, items, set(), baseline_hypotheses=None, domain=domain)
    baseline_hyp = _collect_hypotheses(model, items, set())

    cfg = SlebSearchConfig(
        num_layers=num_layers,
        max_skip_layers=max_skip_layers,
        early_barrier=early_barrier,
        latter_barrier=latter_barrier,
        accept_drop_tol=0.0,
        score_drop_tol=0.0,
        score_tol_mode="absolute",
        accept_metric=accept_metric,
        score_key="task_score",
        exhaustive_singles=False,
    )

    def _eval_skip(skip_layers: set[int]) -> dict[str, Any]:
        try:
            return eval_skip_config(
                model, items, skip_layers, baseline_hypotheses=baseline_hyp, domain=domain
            )
        finally:
            torch.cuda.empty_cache()

    def _on_err(layer: int, exc: Exception) -> None:
        print(f"  [max_metrics_balanced] skip+{layer} failed: {exc}", flush=True)
        torch.cuda.empty_cache()

    skip_set, history, current, candidates = max_metrics_balanced_search(
        eval_fn=_eval_skip,
        baseline=baseline,
        config=cfg,
        on_trial_error=_on_err,
        beam_width=3,
    )
    return {
        "mode": "max_metrics_balanced",
        "selection_criterion": (
            "Metrics-centric explore: expand only if task_score>=baseline; "
            "rank by score then |S| then accept (latter layers first); "
            "retain max_metrics / max_skip / max_accept; select most balanced"
        ),
        "accept_metric": accept_metric,
        "score_drop_tol": 0.0,
        "score_tol_mode": "absolute",
        "early_barrier": early_barrier,
        "latter_barrier": latter_barrier,
        "baseline": baseline,
        "best": current,
        "candidates": candidates,
        "skip_layers": sorted(skip_set),
        "history": history,
    }


def pareto_bridge_v2_mode(
    model: Any,
    items: list[DecodeItem],
    *,
    domain: str,
    num_layers: int,
    accept_metric: str,
    max_skip_layers: int,
    early_barrier: int,
    latter_barrier: int,
    bridge_score_drop_tol: float,
    bridge_score_tol_mode: str,
    pareto_beam_width: int,
    refine_top_k: int,
    refine_max_evals: int,
) -> dict[str, Any]:
    print("[pareto_bridge_v2] baseline (no skip)...", flush=True)
    baseline = eval_skip_config(
        model, items, set(), baseline_hypotheses=None, domain=domain
    )
    baseline_hyp = _collect_hypotheses(model, items, set())
    cfg = SlebSearchConfig(
        num_layers=num_layers,
        max_skip_layers=max_skip_layers,
        early_barrier=early_barrier,
        latter_barrier=latter_barrier,
        accept_drop_tol=0.0,
        score_drop_tol=0.0,
        score_tol_mode="absolute",
        accept_metric=accept_metric,
        score_key="task_score",
        exhaustive_singles=True,
    )
    options = ParetoBridgeOptions(
        bridge_score_drop_tol=bridge_score_drop_tol,
        bridge_score_tol_mode=bridge_score_tol_mode,
        beam_width_per_objective=pareto_beam_width,
        refine_top_k=refine_top_k,
        refine_max_evals=refine_max_evals,
    )

    def _eval_skip(skip_layers: set[int]) -> dict[str, Any]:
        try:
            return eval_skip_config(
                model,
                items,
                skip_layers,
                baseline_hypotheses=baseline_hyp,
                domain=domain,
            )
        finally:
            torch.cuda.empty_cache()

    def _on_err(layer: int, exc: Exception) -> None:
        print(f"  [pareto_bridge_v2] skip+{layer} failed: {exc}", flush=True)
        torch.cuda.empty_cache()

    skip_set, history, current, candidates, metadata = pareto_bridge_v2_search(
        eval_fn=_eval_skip,
        baseline=baseline,
        config=cfg,
        options=options,
        on_trial_error=_on_err,
    )
    return {
        **metadata,
        "mode": "pareto_bridge_v2",
        "selection_criterion": (
            "Full-layer task-aware bridge exploration with metrics/accept/"
            "continuation beams; strict score-preserving Pareto selection "
            "followed by add/delete/swap refinement"
        ),
        "accept_metric": accept_metric,
        "early_barrier": early_barrier,
        "latter_barrier": latter_barrier,
        "baseline": baseline,
        "best": current,
        "candidates": candidates,
        "skip_layers": sorted(skip_set),
        "history": history,
    }


def tri_objective_v3_mode(
    model: Any,
    items: list[DecodeItem],
    *,
    domain: str,
    num_layers: int,
    accept_metric: str,
    max_skip_layers: int,
    early_barrier: int,
    latter_barrier: int,
    bridge_score_drop_tol: float,
    bridge_score_tol_mode: str,
    pareto_beam_width: int,
    refine_top_k: int,
    refine_max_evals: int,
    seed_sets: tuple[tuple[int, ...], ...] = (),
    enable_accept_track: bool = True,
    enable_refine: bool = True,
) -> dict[str, Any]:
    print("[tri_objective_v3] baseline (no skip)...", flush=True)
    baseline = eval_skip_config(
        model, items, set(), baseline_hypotheses=None, domain=domain
    )
    baseline_hyp = _collect_hypotheses(model, items, set())
    cfg = SlebSearchConfig(
        num_layers=num_layers,
        max_skip_layers=max_skip_layers,
        early_barrier=early_barrier,
        latter_barrier=latter_barrier,
        accept_drop_tol=0.0,
        score_drop_tol=0.0,
        score_tol_mode="absolute",
        accept_metric=accept_metric,
        score_key="task_score",
        exhaustive_singles=True,
    )
    options = TriObjectiveOptions(
        score_bridge_drop_tol=bridge_score_drop_tol,
        score_bridge_tol_mode=bridge_score_tol_mode,
        metrics_beam_width=max(4, pareto_beam_width),
        accept_beam_width=max(6, pareto_beam_width)
        if enable_accept_track
        else 0,
        speed_beam_width=max(3, pareto_beam_width // 2),
        refine_top_k=max(8, refine_top_k),
        refine_max_evals=max(180, refine_max_evals) if enable_refine else 0,
        seed_sets=seed_sets,
    )

    def _eval_skip(skip_layers: set[int]) -> dict[str, Any]:
        try:
            return eval_skip_config(
                model,
                items,
                skip_layers,
                baseline_hypotheses=baseline_hyp,
                domain=domain,
            )
        finally:
            torch.cuda.empty_cache()

    def _on_err(layer: int, exc: Exception) -> None:
        print(f"  [tri_objective_v3] skip+{layer} failed: {exc}", flush=True)
        torch.cuda.empty_cache()

    skip_set, history, current, candidates, metadata = tri_objective_v3_search(
        eval_fn=_eval_skip,
        baseline=baseline,
        config=cfg,
        options=options,
        on_trial_error=_on_err,
    )
    return {
        **metadata,
        "mode": "tri_objective_v3",
        "selection_criterion": (
            "Score-bridge metrics beam plus score-unconstrained accept beam; "
            "final score/accept/tok_s hard gates and tri-geometric ranking"
        ),
        "ablation": {
            "enable_accept_track": enable_accept_track,
            "enable_refine": enable_refine,
        },
        "accept_metric": accept_metric,
        "early_barrier": early_barrier,
        "latter_barrier": latter_barrier,
        "baseline": baseline,
        "best": current,
        "candidates": candidates,
        "skip_layers": sorted(skip_set),
        "history": history,
    }


def max_skip_latter_mode(
    model: Any,
    items: list[DecodeItem],
    *,
    domain: str,
    num_layers: int,
    accept_metric: str,
    max_skip_layers: int,
    accept_drop_tol: float,
    score_drop_tol: float,
    score_tol_mode: str,
    early_barrier: int,
    latter_barrier: int,
) -> dict[str, Any]:
    print("[max_skip_latter] baseline (no skip)...", flush=True)
    baseline = eval_skip_config(model, items, set(), baseline_hypotheses=None, domain=domain)
    baseline_hyp = _collect_hypotheses(model, items, set())

    cfg = SlebSearchConfig(
        num_layers=num_layers,
        max_skip_layers=max_skip_layers,
        early_barrier=early_barrier,
        latter_barrier=latter_barrier,
        accept_drop_tol=accept_drop_tol,
        score_drop_tol=score_drop_tol,
        score_tol_mode=score_tol_mode,
        accept_metric=accept_metric,
        score_key="task_score",
        exhaustive_singles=False,
    )

    def _eval_skip(skip_layers: set[int]) -> dict[str, Any]:
        try:
            return eval_skip_config(
                model, items, skip_layers, baseline_hypotheses=baseline_hyp, domain=domain
            )
        finally:
            torch.cuda.empty_cache()

    def _on_err(layer: int, exc: Exception) -> None:
        print(f"  [max_skip_latter] skip+{layer} failed: {exc}", flush=True)
        torch.cuda.empty_cache()

    skip_set, history, current = max_skip_latter_search(
        eval_fn=_eval_skip,
        baseline=baseline,
        config=cfg,
        on_trial_error=_on_err,
        beam_width=3,
        min_skips=min(7, max_skip_layers),
    )
    return {
        "mode": "max_skip_latter",
        "selection_criterion": (
            "Explore to target depth even if intermediate score dips; "
            "expand only if accept>=baseline; "
            "final: accept hard + score within soft tol (e.g. 5%); "
            "prefer accept, then |S|>=min_skips, then tok/s; latter layers first"
        ),
        "accept_metric": accept_metric,
        "accept_drop_tol": accept_drop_tol,
        "score_drop_tol": score_drop_tol,
        "score_tol_mode": score_tol_mode,
        "early_barrier": early_barrier,
        "latter_barrier": latter_barrier,
        "baseline": baseline,
        "best": current,
        "skip_layers": sorted(skip_set),
        "history": history,
    }


def greedy_search(
    model: Any,
    items: list[DecodeItem],
    *,
    domain: str,
    num_layers: int,
    score_drop_tol: float,
    score_tol_mode: str,
    accept_metric: str,
    layer_step: int,
    max_skip_layers: int,
    max_rounds: int,
) -> dict[str, Any]:
    return max_accept_search(
        model,
        items,
        domain=domain,
        num_layers=num_layers,
        score_drop_tol=score_drop_tol,
        score_tol_mode=score_tol_mode,
        accept_metric=accept_metric,
        layer_step=layer_step,
        max_skip_layers=min(max_skip_layers, max_rounds),
    )


def _collect_hypotheses(model: Any, items: list[DecodeItem], skip_layers: set[int]) -> dict[str, str]:
    from eagle.model.utils import reset_tree_mode
    from eagle3_resource_profile_pod_v2 import clear_kv

    tokenizer = model.get_tokenizer()
    out: dict[str, str] = {}
    for item in items:
        ids = tokenizer(item.prompt, return_tensors="pt").input_ids.to("cuda")
        plen = int(ids.shape[1])
        clear_kv(model)
        reset_tree_mode(model)
        model.ea_layer.reset_kv()
        gen_ids, _ = traced_eagenerate_skip(
            model,
            ids,
            request_id=item.request_id,
            skip_layers=skip_layers,
            temperature=0.0,
            max_new_tokens=item.max_tokens,
            max_length=plen + item.max_tokens + 256,
        )
        out[item.request_id] = tokenizer.decode(gen_ids[0, plen:].tolist(), skip_special_tokens=True)
    return out


def run_sweep(
    model: Any,
    items: list[DecodeItem],
    *,
    domain: str,
    num_layers: int,
    preset_configs: dict[str, list[int]],
    score_drop_tol: float,
) -> dict[str, Any]:
    print("[sweep] baseline...", flush=True)
    baseline = eval_skip_config(model, items, set(), baseline_hypotheses=None, domain=domain)
    baseline_hyp = _collect_hypotheses(model, items, set())
    init_score = baseline["task_score"]
    init_accept = baseline["accept_rate"]

    rows: list[dict[str, Any]] = []
    for name, layers in preset_configs.items():
        skip = set(layers)
        print(f"  [sweep] {name} skip={sorted(skip)}...", flush=True)
        m = eval_skip_config(model, items, skip, baseline_hypotheses=baseline_hyp, domain=domain)
        d_accept = m["accept_rate"] - init_accept
        d_score = m["task_score"] - init_score
        score_ok = m["task_score"] >= init_score - score_drop_tol
        accept_up = m["accept_rate"] > init_accept
        rows.append(
            {
                "config": name,
                "skip_layers": json.dumps(sorted(skip)),
                "num_skip_layers": len(skip),
                "accept_rate": m["accept_rate"],
                "delta_accept": d_accept,
                "task_score": m["task_score"],
                "delta_score": d_score,
                "score_ok": score_ok,
                "accept_up": accept_up,
                "pass_both": score_ok and accept_up,
                "mean_accepted_per_step": m["mean_accepted_per_step"],
                "wall_s": m["wall_s"],
            }
        )

    # Per-layer singles (every 4 layers) for ablation hints.
    for layer in range(0, num_layers, 4):
        if layer == 0:
            continue
        name = f"single_L{layer}"
        print(f"  [sweep] {name}...", flush=True)
        m = eval_skip_config(model, items, {layer}, baseline_hypotheses=baseline_hyp, domain=domain)
        d_accept = m["accept_rate"] - init_accept
        d_score = m["task_score"] - init_score
        score_ok = m["task_score"] >= init_score - score_drop_tol
        accept_up = m["accept_rate"] > init_accept
        rows.append(
            {
                "config": name,
                "skip_layers": json.dumps([layer]),
                "num_skip_layers": 1,
                "accept_rate": m["accept_rate"],
                "delta_accept": d_accept,
                "task_score": m["task_score"],
                "delta_score": d_score,
                "score_ok": score_ok,
                "accept_up": accept_up,
                "pass_both": score_ok and accept_up,
                "mean_accepted_per_step": m["mean_accepted_per_step"],
                "wall_s": m["wall_s"],
            }
        )

    passing = [r for r in rows if r["pass_both"]]
    passing.sort(key=lambda r: (-r["delta_accept"], -r["task_score"]))
    return {
        "mode": "sweep",
        "baseline": baseline,
        "rows": rows,
        "best_passing": passing[0] if passing else None,
        "all_passing": passing,
    }


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    cfg = MODEL_PRESETS[args.preset]
    domain = SCORE_CATEGORY[args.dataset]
    if args.train_size is not None:
        from spec_exp.benchmark_datasets import load_dataset_split

        raw = load_dataset_split(
            args.dataset,
            split="train",
            train_size=args.train_size,
            seed=args.seed,
            output_len=args.output_len,
        )
        items = [
            DecodeItem(
                request_id=item.request_id,
                prompt=wrap_chat_prompt(_extract_user_text(item.prompt), cfg["chat_template"]),
                max_tokens=item.max_tokens,
                category=item.category,
                reference=item.reference,
            )
            for item in raw
        ]
    else:
        items = load_items_for_preset(
            args.dataset,
            chat_template=cfg["chat_template"],
            num_requests=args.num_requests,
            seed=args.seed,
            output_len=args.output_len,
        )

    print(
        f"[INFO] preset={args.preset} EAGLE3 skip dataset={args.dataset} mode={args.mode} n={len(items)}",
        flush=True,
    )
    model = load_model(
        base_model=cfg["base_model"],
        ea_model=cfg["ea_model"],
        total_token=args.total_token,
        use_eagle3=cfg.get("use_eagle3", True),
    )

    if args.mode == "single":
        skip = set(int(x) for x in args.skip_layers.split(",") if x.strip()) if args.skip_layers else set()
        baseline = eval_skip_config(model, items, set(), baseline_hypotheses=None, domain=domain)
        baseline_hyp = _collect_hypotheses(model, items, set())
        result = eval_skip_config(model, items, skip, baseline_hypotheses=baseline_hyp, domain=domain)
        payload = {"mode": "single", "baseline": baseline, "result": result}
    elif args.mode == "pareto_bridge_v2":
        payload = pareto_bridge_v2_mode(
            model,
            items,
            domain=domain,
            num_layers=cfg["num_layers"],
            accept_metric=args.accept_metric,
            max_skip_layers=args.max_skip_layers,
            early_barrier=args.early_barrier,
            latter_barrier=args.latter_barrier,
            bridge_score_drop_tol=args.bridge_score_drop_tol,
            bridge_score_tol_mode=args.bridge_score_tol_mode,
            pareto_beam_width=args.pareto_beam_width,
            refine_top_k=args.refine_top_k,
            refine_max_evals=args.refine_max_evals,
        )
    elif args.mode == "tri_objective_v3":
        payload = tri_objective_v3_mode(
            model,
            items,
            domain=domain,
            num_layers=cfg["num_layers"],
            accept_metric=args.accept_metric,
            max_skip_layers=args.max_skip_layers,
            early_barrier=args.early_barrier,
            latter_barrier=args.latter_barrier,
            bridge_score_drop_tol=args.bridge_score_drop_tol,
            bridge_score_tol_mode=args.bridge_score_tol_mode,
            pareto_beam_width=args.pareto_beam_width,
            refine_top_k=args.refine_top_k,
            refine_max_evals=args.refine_max_evals,
        )
    elif args.mode == "max_metrics_balanced":
        payload = max_metrics_balanced_mode(
            model,
            items,
            domain=domain,
            num_layers=cfg["num_layers"],
            accept_metric=args.accept_metric,
            max_skip_layers=args.max_skip_layers,
            early_barrier=args.early_barrier,
            latter_barrier=args.latter_barrier,
        )
    elif args.mode == "max_skip_latter":
        payload = max_skip_latter_mode(
            model,
            items,
            domain=domain,
            num_layers=cfg["num_layers"],
            accept_metric=args.accept_metric,
            max_skip_layers=args.max_skip_layers,
            accept_drop_tol=args.accept_drop_tol,
            score_drop_tol=args.score_drop_tol,
            score_tol_mode=args.score_tol_mode,
            early_barrier=args.early_barrier,
            latter_barrier=args.latter_barrier,
        )
    elif args.mode == "max_toks":
        payload = max_toks_search(
            model,
            items,
            domain=domain,
            num_layers=cfg["num_layers"],
            accept_metric=args.accept_metric,
            layer_step=args.layer_step,
            max_skip_layers=args.max_skip_layers,
            exhaustive_singles=args.exhaustive_singles,
        )
    elif args.mode in ("greedy", "max_accept"):
        payload = max_accept_search(
            model,
            items,
            domain=domain,
            num_layers=cfg["num_layers"],
            score_drop_tol=args.score_drop_tol,
            score_tol_mode=args.score_tol_mode,
            accept_metric=args.accept_metric,
            layer_step=args.layer_step,
            max_skip_layers=args.max_skip_layers,
            exhaustive_singles=args.exhaustive_singles,
        )
    else:
        payload = run_sweep(
            model,
            items,
            domain=domain,
            num_layers=cfg["num_layers"],
            preset_configs=cfg["preset_configs"],
            score_drop_tol=args.score_drop_tol,
        )

    payload["preset"] = args.preset
    payload["use_eagle3"] = cfg.get("use_eagle3", True)
    payload["base_model"] = cfg["base_model"]
    payload["ea_model"] = cfg["ea_model"]
    payload["dataset"] = args.dataset
    payload["domain"] = domain
    payload["num_requests"] = len(items)
    payload["train_size"] = args.train_size
    payload["score_drop_tol"] = args.score_drop_tol
    payload["score_tol_mode"] = args.score_tol_mode
    payload["accept_metric"] = args.accept_metric

    tag = f"{args.preset}_skip_{args.mode}_{args.dataset}"
    write_json(payload, out_dir / f"{tag}.json")
    if "history" in payload:
        write_table(payload["history"], out_dir / f"{tag}_history.csv")
    if "rows" in payload:
        write_table(payload["rows"], out_dir / f"{tag}.csv")
    print(json.dumps(payload.get("best_passing") or payload.get("best") or payload.get("result"), indent=2), flush=True)


if __name__ == "__main__":
    main()
