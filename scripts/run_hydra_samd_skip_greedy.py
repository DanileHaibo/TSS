#!/usr/bin/env python3
"""Greedy multi-layer skip search for Hydra and SAMD[EAGLE2] on Vicuna-7B-v1.3."""
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
SPEC_BENCH = REPO / "data" / "Spec-Bench-repo"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SPEC_BENCH))

from spec_exp.benchmark_config import SCORE_CATEGORY
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
    sleb_layer_search,
)
from spec_exp.task_score import mean_task_score_detailed

VICUNA7_BASE = "/root/autodl-tmp/models/vicuna-7b-v1.3"
VICUNA33_BASE = "/root/autodl-tmp/models/vicuna-33b-v1.3"
HYDRA7_PATH = "/root/autodl-tmp/models/hydra-vicuna-7b-v1.3"
HYDRA13_PATH = "/root/autodl-tmp/models/hydra-vicuna-13b-v1.3"
HYDRA33_PATH = "/root/autodl-tmp/models/hydra-vicuna-33b-v1.3"
EAGLE_PATHS: dict[str, str] = {
    "7b": "/root/autodl-tmp/models/EAGLE-Vicuna-7B-v1.3",
    "33b": "/root/autodl-tmp/models/EAGLE-Vicuna-33B-v1.3",
}

SIZE_PRESETS: dict[str, dict[str, Any]] = {
    "7b": {
        "num_layers": 32,
        "layer_step": 1,
        "resolve_base": "resolve_vicuna7",
        "hydra_path": HYDRA7_PATH,
        "base_label": "vicuna-7b-v1.3",
    },
    "13b": {
        "num_layers": 40,
        "layer_step": 1,
        "resolve_base": "resolve_vicuna13",
        "hydra_path": HYDRA13_PATH,
        "base_label": "vicuna-13b-v1.3",
    },
    "33b": {
        "num_layers": 60,
        "layer_step": 1,
        "resolve_base": "resolve_vicuna33",
        "hydra_path": HYDRA33_PATH,
        "base_label": "vicuna-33b-v1.3",
        "eagle_path": EAGLE_PATHS["33b"],
    },
}


def resolve_vicuna7() -> str:
    cache = Path("/root/autodl-tmp/hf-cache/hub/models--lmsys--vicuna-7b-v1.3/snapshots")
    if cache.exists():
        snaps = sorted(cache.iterdir())
        for s in snaps:
            if (s / "config.json").exists():
                return str(s)
    if Path(VICUNA7_BASE, "config.json").exists():
        return VICUNA7_BASE
    raise FileNotFoundError("Vicuna-7B not found")


def resolve_vicuna13() -> str:
    cache = Path("/root/autodl-tmp/hf-cache/hub/models--lmsys--vicuna-13b-v1.3/snapshots")
    if cache.exists():
        for s in sorted(cache.iterdir()):
            if (s / "config.json").exists():
                return str(s)
    raise FileNotFoundError("Vicuna-13B not found")


def resolve_vicuna33() -> str:
    if Path(VICUNA33_BASE, "config.json").exists():
        return VICUNA33_BASE
    cache = Path("/root/autodl-tmp/hf-cache/hub/models--lmsys--vicuna-33b-v1.3/snapshots")
    if cache.exists():
        for s in sorted(cache.iterdir()):
            if (s / "config.json").exists():
                return str(s)
    raise FileNotFoundError("Vicuna-33B not found")


def resolve_vicuna(size: str) -> str:
    if size == "33b":
        return resolve_vicuna33()
    if size == "13b":
        return resolve_vicuna13()
    return resolve_vicuna7()


def resolve_hydra_path(size: str) -> str:
    cfg = SIZE_PRESETS[size]
    path = Path(cfg["hydra_path"])
    if (path / "config.json").exists():
        return str(path)
    cache = Path(f"/root/autodl-tmp/hf-cache/hub/models--ankner--hydra-vicuna-{size}-v1.3/snapshots")
    if cache.exists():
        for s in sorted(cache.iterdir()):
            if (s / "config.json").exists():
                return str(s)
    raise FileNotFoundError(f"Hydra Vicuna-{size} not found at {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["hydra", "samd"])
    p.add_argument("--size", choices=["7b", "13b", "33b"], default="7b")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-requests", type=int, default=16)
    p.add_argument("--output-len", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score-drop-tol", type=float, default=0.05)
    p.add_argument("--score-tol-mode", choices=["absolute", "relative"], default="absolute")
    p.add_argument("--accept-mode", choices=["improve", "baseline"], default="baseline")
    p.add_argument("--accept-drop-tol", type=float, default=0.08)
    p.add_argument("--pick-metric", choices=["accept", "tok_per_s", "mean_accepted"], default="tok_per_s")
    p.add_argument("--max-skip-layers", type=int, default=12)
    p.add_argument(
        "--min-skips",
        type=int,
        default=3,
        help="Minimum preferred final skip count for max_skip_latter.",
    )
    p.add_argument("--max-rounds", type=int, default=12)
    p.add_argument("--layer-step", type=int, default=1, help="Try every N-th layer (1 = all layers)")
    p.add_argument(
        "--exhaustive-singles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Phase-1 sweep: evaluate every single-layer skip before greedy multi-layer",
    )
    p.add_argument(
        "--search-mode",
        choices=[
            "sleb",
            "legacy",
            "max_accept",
            "max_toks",
            "max_skip_latter",
            "score_first",
            "max_metrics_balanced",
            "pareto_bridge_v2",
        ],
        default="max_metrics_balanced",
        help="max_metrics_balanced: preserve/max metrics, keep max_metrics/max_skip/max_accept, pick balanced; "
        "max_skip_latter: prefer later layers, soft tol, maximize #skips; "
        "score_first: force target skip depth and optimize task score only; "
        "max_toks: maximize tok/s with hard accept/score; "
        "max_accept: preserve task_score, maximize mean_accepted_per_step; "
        "sleb: SLEB min-harm; legacy: forward greedy pick_metric",
    )
    p.add_argument("--train-size", type=int, default=None, help="Use first N shuffled prompts for search")
    p.add_argument("--early-barrier", type=int, default=2)
    p.add_argument("--latter-barrier", type=int, default=0)
    p.add_argument(
        "--accept-metric",
        choices=["mean_accepted_per_step", "accept_rate"],
        default="mean_accepted_per_step",
    )
    p.add_argument("--accept-weight", type=float, default=1.0)
    p.add_argument("--score-weight", type=float, default=1.0)
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


def wrap_vicuna_prompt(user_text: str) -> str:
    from fastchat.model import get_conversation_template

    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], user_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _extract_user_text(prompt: str) -> str:
    if "<|im_start|>user" in prompt:
        return prompt.split("<|im_start|>user\n", 1)[1].split("\n<|im_start|>assistant")[0]
    return prompt


def load_items(dataset: str, *, num_requests: int, seed: int, output_len: int) -> list[DecodeItem]:
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
            prompt=wrap_vicuna_prompt(_extract_user_text(item.prompt)),
            max_tokens=item.max_tokens,
            category=item.category,
            reference=item.reference,
        )
        for item in base
    ]


@contextmanager
def skip_llama_layers(module: Any, skip_layers_set: set[int], *, legacy_tuple: bool = False):
    if not skip_layers_set:
        yield
        return
    layers = module.layers
    saved: list[tuple[int, Callable]] = []
    for idx, layer in enumerate(layers):
        if idx not in skip_layers_set:
            continue
        orig = layer.forward

        def _make_passthrough(original_forward, layer_idx: int, skip_set: set[int], use_legacy_tuple: bool):
            def passthrough(self, hidden_states, *args, **kwargs):
                if layer_idx in skip_set:
                    if use_legacy_tuple:
                        return hidden_states, None, None
                    return hidden_states
                return original_forward(hidden_states, *args, **kwargs)

            return passthrough

        layer.forward = MethodType(
            _make_passthrough(orig, idx, skip_layers_set, legacy_tuple),
            layer,
        )
        saved.append((idx, orig))
    try:
        yield
    finally:
        for idx, orig in saved:
            layers[idx].forward = orig


@contextmanager
def hydra_skip_ctx(model: Any, skip_layers_set: set[int]):
    with skip_llama_layers(model.base_model.model, skip_layers_set, legacy_tuple=True):
        yield


def clear_hydra_state(model: Any) -> None:
    from model.hydra.utils import reset_hydra_mode

    reset_hydra_mode(model)
    for attr in (
        "past_key_values",
        "past_key_values_data",
        "current_length_data",
        "hydra_buffers",
        "hydra_choices",
    ):
        if hasattr(model, attr):
            delattr(model, attr)


def traced_hydra_skip(
    *,
    inputs: Any,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    hydra_choices: list,
    skip_layers_set: set[int],
) -> tuple[torch.Tensor, int, list[int], list[int]]:
    from model.hydra.kv_cache import initialize_past_key_values
    from model.hydra.utils import (
        evaluate_posterior,
        generate_hydra_buffers,
        initialize_hydra,
        reset_hydra_mode,
        tree_decoding,
        update_inference_inputs,
    )

    input_ids = inputs.input_ids.clone()
    accept_lengths: list[int] = []
    drafted_lengths: list[int] = []

    if hasattr(model, "hydra_choices") and model.hydra_choices == hydra_choices:
        hydra_buffers = model.hydra_buffers
    else:
        hydra_buffers = generate_hydra_buffers(hydra_choices, device=model.base_model.device)
    model.hydra_buffers = hydra_buffers
    model.hydra_choices = hydra_choices

    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        current_length_data.zero_()
    else:
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
            model.base_model, model.hydra_head_arch
        )
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    cur_length = input_len
    reset_hydra_mode(model)
    hidden_states, logits = initialize_hydra(
        input_ids,
        model,
        hydra_buffers["hydra_attn_mask"],
        past_key_values,
        hydra_buffers["proposal_cross_attn_masks"],
    )
    new_token = 0

    for _ in range(max_new_tokens):
        to_pass_input_ids = input_ids if _ == 0 else None
        candidates, tree_candidates = model.hydra_head.proposal(
            logits, hidden_states, hydra_buffers, past_key_values, to_pass_input_ids
        )
        drafted_lengths.append(int(tree_candidates.shape[1]))
        with hydra_skip_ctx(model, skip_layers_set):
            hidden_states, logits = tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                hydra_buffers["hydra_position_ids"],
                input_ids,
                hydra_buffers["retrieve_indices"],
            )
        best_candidate, accept_length = evaluate_posterior(
            logits,
            candidates,
            temperature=0.0,
            posterior_threshold=0.09,
            posterior_alpha=0.3,
            max_accepts=hydra_buffers["max_accepts"],
        )
        accept_lengths.append(int(accept_length))
        input_ids, logits, hidden_states, new_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            hydra_buffers["retrieve_indices"],
            logits,
            hidden_states,
            new_token,
            past_key_values_data,
            current_length_data,
            model.hydra_head_arch,
        )
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_new_tokens:
            break
    return input_ids, new_token, accept_lengths, drafted_lengths


def traced_samd_skip(
    *,
    inputs: Any,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    skip_layers_set: set[int],
) -> tuple[torch.Tensor, int, list[int], list[int]]:
    from model.samd.samd_model import SamdGenerationConfig
    from model.samd.utils import CandidateType

    input_ids = inputs.input_ids
    # Long Spec-Bench prompts can exceed 1K tokens. 13B has enough headroom for
    # a 2048-token static cache; retain the tighter limit only for 33B.
    size = getattr(model, "_samd_size", "")
    cache_type = getattr(model, "_samd_cache_type", "static")
    max_cache_len = getattr(model, "_samd_max_cache_len", None) or (
        512 if cache_type == "static" and size == "33b" else 2048
    )
    gen_cfg = SamdGenerationConfig(max_new_tokens=max_new_tokens, greedy=True, max_cache_len=max_cache_len)
    model.gen_config = gen_cfg
    # Fresh static cache per request — reuse leaks KV tensors across prompts.
    model.cache = None
    model.set_cache(gen_cfg)
    model.draft.reset()

    accept_lengths: list[int] = []
    drafted_lengths: list[int] = []
    input_ids_list = input_ids.squeeze(0).tolist()
    sample_p = model.prefill(input_ids, None)
    input_length = input_ids.shape[-1]
    decode_tokens = 0

    orig_decode = model.decode

    def decode_with_skip(sample_p_in, length: int):
        from model.samd.utils import eval_posterior, gen_candidates
        from model.samd.samd_model import OptionalTensor

        candidates = gen_candidates(
            sample_p_in,
            model.base_tree_retrieve_indices,
            model.draft,
            model.samd_config,
            model.gen_config,
            model.device,
        )
        model.update_buffers(candidates.buffers_kwargs)
        from model.samd.samd_config import ForwardType

        if candidates.type == CandidateType.sequence:
            model.forward_state.forward_type = ForwardType.seq_decode
            position_ids = model.seq_position_ids + length
            is_tree = False
        else:
            model.forward_state.forward_type = ForwardType.tree_decode
            position_ids = model.tree_position_ids + length
            is_tree = True
        input_ids_local = candidates.tokens
        if is_tree and skip_layers_set:
            with skip_llama_layers(model.lm.model, skip_layers_set):
                outputs = model.lm(
                    input_ids=input_ids_local,
                    position_ids=position_ids,
                    past_key_values=model.cache,
                )
        else:
            outputs = model.lm(
                input_ids=input_ids_local,
                position_ids=position_ids,
                past_key_values=model.cache,
            )
        tree_logits = outputs.logits
        if model.samd_config.use_last_hidden_states:
            tree_last_hidden_states = OptionalTensor(outputs.last_hidden_states)
        else:
            tree_last_hidden_states = OptionalTensor(None)
        if candidates.type == CandidateType.sequence:
            candidate_logits = tree_logits
            candidate_last_hidden_states = tree_last_hidden_states
            candidate_indices = OptionalTensor(None)
            drafted_lengths.append(0)
        else:
            candidate_logits = tree_logits.squeeze(0)[model.tree_retrieve_indices]
            candidate_last_hidden_states = tree_last_hidden_states.apply(
                lambda x: x.squeeze(0)[model.tree_retrieve_indices]
            )
            candidate_indices = OptionalTensor(model.tree_retrieve_indices)
            drafted_lengths.append(int(candidates.tokens.shape[-1]))
        best_candidate, accept_length, sample_p_out = eval_posterior(
            candidate_logits, candidates.candidate_tokens, model.gen_config
        )
        accept_lengths.append(int(accept_length))
        new_tokens = model.update_state(
            input_ids_local.squeeze(0),
            tree_logits.squeeze(0),
            best_candidate,
            accept_length,
            candidates.candidate_tokens,
            candidate_indices,
            candidate_last_hidden_states,
        )
        return sample_p_out, new_tokens

    model.decode = decode_with_skip
    try:
        for _ in range(max_new_tokens):
            if input_length + decode_tokens + model.samd_config.max_predicts >= gen_cfg.max_cache_len:
                break
            sample_p, new_ids = model.decode(sample_p, input_length + decode_tokens)
            if model.eos_token in new_ids:
                new_ids = new_ids[: new_ids.index(model.eos_token) + 1]
            input_ids_list.extend(new_ids)
            decode_tokens += len(new_ids)
            if model.eos_token in new_ids:
                break
            if decode_tokens >= max_new_tokens:
                break
    finally:
        model.decode = orig_decode

    out = torch.tensor([input_ids_list[: input_length + max_new_tokens]], device=input_ids.device)
    return out, decode_tokens, accept_lengths, drafted_lengths


def load_hydra_model(vicuna_base: str, *, hydra_path: str) -> Any:
    from fastchat.utils import str_to_torch_dtype
    from model.hydra.hydra_model import HydraModel

    return HydraModel.from_pretrained(
        hydra_path,
        vicuna_base,
        torch_dtype=str_to_torch_dtype("float16"),
        low_cpu_mem_usage=True,
        device_map="cuda:0",
    ).eval()


def load_samd_model(
    vicuna_base: str,
    *,
    eagle_path: str | None = None,
    size: str = "7b",
    tree_method: str = "eagle2",
    use_safetensors: bool | None = None,
    token_tree_path: str | None = None,
    cache_type: str = "static",
    max_cache_len: int | None = None,
) -> Any:
    from fastchat.utils import str_to_torch_dtype
    from model.samd import SamdConfig, SamdModel, DraftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tree_path = eagle_path or EAGLE_PATHS["7b"]
    load_kwargs = {
        "torch_dtype": str_to_torch_dtype("float16"),
        "low_cpu_mem_usage": True,
        "device_map": "cuda:0",
        "attn_implementation": "sdpa",
    }
    if use_safetensors is not None:
        load_kwargs["use_safetensors"] = use_safetensors
    lm = AutoModelForCausalLM.from_pretrained(vicuna_base, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(vicuna_base)
    device = next(lm.lm_head.parameters()).device
    n_predicts = 10 if size in {"13b", "33b"} else 40
    samd_config = SamdConfig(
        n_predicts=n_predicts,
        tree_method=tree_method,
        tree_model_path=tree_path if tree_method in {"eagle", "eagle2"} else None,
        tree_path=token_tree_path,
        len_threshold=5,
        len_bias=5,
        cache_type=cache_type,
    )
    draft = DraftModel(samd_config, sam_static=None, lm=lm, dtype=torch.float16, device=device)
    model = SamdModel(
        samd_config,
        lm,
        draft,
        tokenizer.eos_token_id,
        torch.float16,
        device,
    ).eval()
    model._samd_size = size
    model._samd_cache_type = cache_type
    model._samd_max_cache_len = max_cache_len
    return model


def eval_hydra(
    model: Any,
    items: list[DecodeItem],
    skip_layers_set: set[int],
    *,
    baseline_hypotheses: dict[str, str] | None,
    domain: str,
) -> dict[str, Any]:
    from model.hydra.hydra_choices import mc_sim_7b_63

    tokenizer = model.get_tokenizer()
    hypotheses: dict[str, str] = {}
    references: dict[str, str | None] = {}
    questions: dict[str, str] = {}
    total_drafted = total_accepted = total_verify = total_out = 0
    t0 = time.perf_counter()

    for item in items:
        clear_hydra_state(model)
        inputs = tokenizer(item.prompt, return_tensors="pt").to(model.base_model.device)
        plen = int(inputs.input_ids.shape[1])
        out_ids, _, accept_list, draft_list = traced_hydra_skip(
            inputs=inputs,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=item.max_tokens,
            hydra_choices=mc_sim_7b_63,
            skip_layers_set=skip_layers_set,
        )
        gen_ids = out_ids[0, plen:].tolist()
        hypotheses[item.request_id] = tokenizer.decode(gen_ids, skip_special_tokens=True) if gen_ids else ""
        references[item.request_id] = item.reference
        questions[item.request_id] = item.prompt
        total_out += len(gen_ids)
        total_verify += len(accept_list)
        total_drafted += sum(draft_list)
        total_accepted += sum(accept_list)

    wall = time.perf_counter() - t0
    has_ref = any(r for r in references.values())
    score_detail = mean_task_score_detailed(
        category=domain,
        hypotheses=hypotheses,
        references=references,
        baseline_hypotheses=None if has_ref else baseline_hypotheses,
        questions=questions,
    )
    accept_rate = total_accepted / total_drafted if total_drafted else math.nan
    return {
        "skip_layers": sorted(skip_layers_set),
        "num_skip_layers": len(skip_layers_set),
        "accept_rate": accept_rate,
        "mean_accepted_per_step": 1.0 + total_accepted / max(total_verify, 1),
        "task_score": score_detail["mean_score"],
        "task_score_metric": score_detail.get("metric"),
        "tok_per_s": total_out / wall if wall > 0 else math.nan,
        "wall_s": wall,
        "total_output_tokens": total_out,
        "num_verify_steps": total_verify,
        "hypotheses": hypotheses,
    }


def eval_samd(
    model: Any,
    items: list[DecodeItem],
    skip_layers_set: set[int],
    *,
    baseline_hypotheses: dict[str, str] | None,
    domain: str,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model.lm.config._name_or_path)
    hypotheses: dict[str, str] = {}
    references: dict[str, str | None] = {}
    questions: dict[str, str] = {}
    total_drafted = total_accepted = total_verify = total_out = 0
    t0 = time.perf_counter()

    torch.cuda.empty_cache()
    for item in items:
        model.cache = None
        model.draft.reset()
        inputs = tokenizer(item.prompt, return_tensors="pt").to(model.device)
        plen = int(inputs.input_ids.shape[1])
        out_ids, _, accept_list, draft_list = traced_samd_skip(
            inputs=inputs,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=item.max_tokens,
            skip_layers_set=skip_layers_set,
        )
        gen_ids = out_ids[0, plen:].tolist()
        hypotheses[item.request_id] = tokenizer.decode(gen_ids, skip_special_tokens=True) if gen_ids else ""
        references[item.request_id] = item.reference
        questions[item.request_id] = item.prompt
        total_out += len(gen_ids)
        total_verify += len(accept_list)
        total_drafted += sum(draft_list)
        total_accepted += sum(accept_list)
        torch.cuda.empty_cache()

    wall = time.perf_counter() - t0
    has_ref = any(r for r in references.values())
    score_detail = mean_task_score_detailed(
        category=domain,
        hypotheses=hypotheses,
        references=references,
        baseline_hypotheses=None if has_ref else baseline_hypotheses,
        questions=questions,
    )
    accept_rate = total_accepted / total_drafted if total_drafted else math.nan
    return {
        "skip_layers": sorted(skip_layers_set),
        "num_skip_layers": len(skip_layers_set),
        "accept_rate": accept_rate,
        "mean_accepted_per_step": 1.0 + total_accepted / max(total_verify, 1),
        "task_score": score_detail["mean_score"],
        "task_score_metric": score_detail.get("metric"),
        "tok_per_s": total_out / wall if wall > 0 else math.nan,
        "wall_s": wall,
        "total_output_tokens": total_out,
        "num_verify_steps": total_verify,
        "hypotheses": hypotheses,
    }


def collect_baseline_hypotheses(method: str, model: Any, items: list[DecodeItem]) -> dict[str, str]:
    from model.hydra.hydra_choices import mc_sim_7b_63
    from transformers import AutoTokenizer

    out: dict[str, str] = {}
    if method == "hydra":
        tokenizer = model.get_tokenizer()
        for item in items:
            clear_hydra_state(model)
            inputs = tokenizer(item.prompt, return_tensors="pt").to(model.base_model.device)
            plen = int(inputs.input_ids.shape[1])
            gen_ids, _, _, _ = traced_hydra_skip(
                inputs=inputs,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=item.max_tokens,
                hydra_choices=mc_sim_7b_63,
                skip_layers_set=set(),
            )
            out[item.request_id] = tokenizer.decode(gen_ids[0, plen:].tolist(), skip_special_tokens=True)
        return out
    vicuna_base = resolve_vicuna7()
    tokenizer = AutoTokenizer.from_pretrained(vicuna_base)
    for item in items:
        inputs = tokenizer(item.prompt, return_tensors="pt").to(model.device)
        plen = int(inputs.input_ids.shape[1])
        gen_ids, _, _, _ = traced_samd_skip(
            inputs=inputs,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=item.max_tokens,
            skip_layers_set=set(),
        )
        out[item.request_id] = tokenizer.decode(gen_ids[0, plen:].tolist(), skip_special_tokens=True)
    return out


def _score_ok(score: float, init_score: float, tol: float, mode: str) -> bool:
    if math.isnan(score):
        return False
    if math.isnan(init_score):
        return True
    if mode == "relative":
        if abs(init_score) < 1e-9:
            return score >= init_score - tol
        return score >= init_score * (1.0 - tol)
    return score >= init_score - tol


def _accept_ok(
    accept: float,
    *,
    init_accept: float,
    current_accept: float,
    mode: str,
    accept_drop_tol: float,
) -> bool:
    if math.isnan(accept):
        return False
    if mode == "baseline":
        return accept >= init_accept * (1.0 - accept_drop_tol)
    return accept > current_accept


def _pick_value(metrics: dict[str, Any], pick_metric: str) -> float:
    if pick_metric == "tok_per_s":
        return float(metrics.get("tok_per_s") or -1.0)
    if pick_metric == "mean_accepted":
        return float(metrics.get("mean_accepted_per_step") or -1.0)
    return float(metrics.get("accept_rate") or -1.0)


def greedy_search(
    *,
    method: str,
    model: Any,
    items: list[DecodeItem],
    domain: str,
    num_layers: int,
    layer_step: int,
    score_drop_tol: float,
    score_tol_mode: str,
    accept_mode: str,
    accept_drop_tol: float,
    pick_metric: str,
    max_skip_layers: int,
    min_skips: int = 3,
    max_rounds: int,
    search_mode: str = "sleb",
    early_barrier: int = 1,
    latter_barrier: int = 1,
    accept_metric: str = "mean_accepted_per_step",
    accept_weight: float = 1.0,
    score_weight: float = 1.0,
    layer_step_override: int | None = None,
    exhaustive_singles: bool = True,
    bridge_score_drop_tol: float = 0.05,
    bridge_score_tol_mode: str = "relative",
    pareto_beam_width: int = 2,
    refine_top_k: int = 4,
    refine_max_evals: int = 80,
) -> dict[str, Any]:
    eval_fn = eval_hydra if method == "hydra" else eval_samd

    print(f"[{search_mode}] baseline (no skip)...", flush=True)
    baseline = eval_fn(model, items, set(), baseline_hypotheses=None, domain=domain)
    baseline_hyp = baseline.pop("hypotheses", None) or collect_baseline_hypotheses(method, model, items)
    init_score = baseline["task_score"]
    init_accept = baseline["accept_rate"]
    init_mean_acc = baseline.get("mean_accepted_per_step", init_accept)

    if search_mode == "pareto_bridge_v2":
        cfg = SlebSearchConfig(
            num_layers=num_layers,
            max_skip_layers=max_skip_layers,
            early_barrier=early_barrier,
            latter_barrier=latter_barrier,
            accept_drop_tol=accept_drop_tol,
            score_drop_tol=0.0,
            score_tol_mode="absolute",
            accept_metric=accept_metric,
            score_key="task_score",
            layer_step=layer_step_override or layer_step,
            exhaustive_singles=True,
        )
        options = ParetoBridgeOptions(
            bridge_score_drop_tol=bridge_score_drop_tol,
            bridge_score_tol_mode=bridge_score_tol_mode,
            beam_width_per_objective=pareto_beam_width,
            refine_top_k=refine_top_k,
            refine_max_evals=refine_max_evals,
        )

        def _eval_skip_pareto(skip_layers: set[int]) -> dict[str, Any]:
            if method == "samd":
                model.cache = None
                if hasattr(model, "draft"):
                    model.draft.reset()
            try:
                return eval_fn(
                    model,
                    items,
                    skip_layers,
                    baseline_hypotheses=baseline_hyp,
                    domain=domain,
                )
            finally:
                torch.cuda.empty_cache()

        def _on_err_pareto(layer: int, exc: Exception) -> None:
            print(f"  [pareto_bridge_v2] skip+{layer} failed: {exc}", flush=True)
            if method == "samd":
                model.cache = None
                if hasattr(model, "draft"):
                    model.draft.reset()
            torch.cuda.empty_cache()

        skip_set, history, current, candidates, metadata = pareto_bridge_v2_search(
            eval_fn=_eval_skip_pareto,
            baseline=baseline,
            config=cfg,
            options=options,
            on_trial_error=_on_err_pareto,
        )
        return {
            **metadata,
            "method": method,
            "mode": "pareto_bridge_v2",
            "search_mode": search_mode,
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

    if search_mode == "max_metrics_balanced":
        macfg = SlebSearchConfig(
            num_layers=num_layers,
            max_skip_layers=max_skip_layers,
            early_barrier=early_barrier,
            latter_barrier=latter_barrier,
            accept_drop_tol=accept_drop_tol,
            score_drop_tol=0.0,
            score_tol_mode="absolute",
            accept_metric=accept_metric,
            score_key="task_score",
            layer_step=layer_step_override or layer_step,
            exhaustive_singles=False,
        )

        def _eval_skip_metrics(skip_layers: set[int]) -> dict[str, Any]:
            if method == "samd":
                model.cache = None
                if hasattr(model, "draft"):
                    model.draft.reset()
            try:
                return eval_fn(
                    model, items, skip_layers, baseline_hypotheses=baseline_hyp, domain=domain
                )
            finally:
                torch.cuda.empty_cache()

        def _on_err_metrics(layer: int, exc: Exception) -> None:
            print(f"  [max_metrics_balanced] skip+{layer} failed: {exc}", flush=True)
            if method == "samd":
                model.cache = None
                if hasattr(model, "draft"):
                    model.draft.reset()
            torch.cuda.empty_cache()

        skip_set, history, current, candidates = max_metrics_balanced_search(
            eval_fn=_eval_skip_metrics,
            baseline=baseline,
            config=macfg,
            on_trial_error=_on_err_metrics,
            beam_width=3,
        )
        return {
            "method": method,
            "mode": "max_metrics_balanced",
            "search_mode": search_mode,
            "selection_criterion": (
                "Metrics-centric explore: expand only if task_score>=baseline; "
                "rank by score then |S| then accept (latter layers first); "
                "retain max_metrics / max_skip / max_accept; select most balanced"
            ),
            "accept_drop_tol": accept_drop_tol,
            "score_drop_tol": 0.0,
            "score_tol_mode": "absolute",
            "early_barrier": early_barrier,
            "latter_barrier": latter_barrier,
            "accept_metric": accept_metric,
            "baseline": baseline,
            "best": current,
            "candidates": candidates,
            "skip_layers": sorted(skip_set),
            "history": history,
        }

    if search_mode in {"max_skip_latter", "score_first"}:
        macfg = SlebSearchConfig(
            num_layers=num_layers,
            max_skip_layers=max_skip_layers,
            early_barrier=early_barrier,
            latter_barrier=latter_barrier,
            accept_drop_tol=accept_drop_tol,
            score_drop_tol=score_drop_tol,
            score_tol_mode=score_tol_mode,
            accept_metric=accept_metric,
            score_key="task_score",
            layer_step=layer_step_override or layer_step,
            exhaustive_singles=False,
        )

        def _eval_skip_latter(skip_layers: set[int]) -> dict[str, Any]:
            if method == "samd":
                model.cache = None
                if hasattr(model, "draft"):
                    model.draft.reset()
            try:
                return eval_fn(
                    model, items, skip_layers, baseline_hypotheses=baseline_hyp, domain=domain
                )
            finally:
                torch.cuda.empty_cache()

        def _on_err_latter(layer: int, exc: Exception) -> None:
            print(f"  [max_skip_latter] skip+{layer} failed: {exc}", flush=True)
            if method == "samd":
                model.cache = None
                if hasattr(model, "draft"):
                    model.draft.reset()
            torch.cuda.empty_cache()

        skip_set, history, current = max_skip_latter_search(
            eval_fn=_eval_skip_latter,
            baseline=baseline,
            config=macfg,
            on_trial_error=_on_err_latter,
            beam_width=3,
            min_skips=min(max(0, min_skips), max_skip_layers),
            score_first=search_mode == "score_first",
        )
        return {
            "method": method,
            "mode": search_mode,
            "search_mode": search_mode,
            "selection_criterion": (
                "Require |S|>=min_skips; rank and select by task score; "
                "accept and tok/s are tie-breakers only"
                if search_mode == "score_first"
                else (
                    "Explore to target depth even if intermediate score dips; "
                    "expand only if accept>=baseline; "
                    "final: accept hard + score within soft tol (e.g. 5%); "
                    "prefer accept, then |S|>=min_skips, then tok/s; latter layers first"
                )
            ),
            "accept_drop_tol": accept_drop_tol,
            "score_drop_tol": score_drop_tol,
            "score_tol_mode": score_tol_mode,
            "early_barrier": early_barrier,
            "latter_barrier": latter_barrier,
            "accept_metric": accept_metric,
            "baseline": baseline,
            "best": current,
            "skip_layers": sorted(skip_set),
            "history": history,
        }

    if search_mode == "max_toks":
        macfg = SlebSearchConfig(
            num_layers=num_layers,
            max_skip_layers=max_skip_layers,
            score_drop_tol=0.0,
            score_tol_mode="absolute",
            accept_metric=accept_metric,
            score_key="task_score",
            layer_step=layer_step_override or layer_step,
            exhaustive_singles=exhaustive_singles,
        )

        def _eval_skip_toks(skip_layers: set[int]) -> dict[str, Any]:
            if method == "samd":
                model.cache = None
                if hasattr(model, "draft"):
                    model.draft.reset()
            try:
                return eval_fn(
                    model, items, skip_layers, baseline_hypotheses=baseline_hyp, domain=domain
                )
            finally:
                torch.cuda.empty_cache()

        def _on_err_toks(layer: int, exc: Exception) -> None:
            print(f"  [max_toks] skip+{layer} failed: {exc}", flush=True)
            if method == "samd":
                model.cache = None
                if hasattr(model, "draft"):
                    model.draft.reset()
            torch.cuda.empty_cache()

        skip_set, history, current = max_toks_preserve_quality_search(
            eval_fn=_eval_skip_toks,
            baseline=baseline,
            config=macfg,
            on_trial_error=_on_err_toks,
        )
        return {
            "method": method,
            "mode": "max_toks",
            "search_mode": search_mode,
            "selection_criterion": (
                "max tok_per_s subject to accept>=baseline AND task_score>=baseline"
            ),
            "accept_metric": accept_metric,
            "baseline": baseline,
            "best": current,
            "skip_layers": sorted(skip_set),
            "history": history,
        }

    if search_mode == "max_accept":
        macfg = SlebSearchConfig(
            num_layers=num_layers,
            max_skip_layers=max_skip_layers,
            score_drop_tol=score_drop_tol,
            score_tol_mode=score_tol_mode,
            accept_metric=accept_metric,
            score_key="task_score",
            layer_step=layer_step_override or layer_step,
            exhaustive_singles=exhaustive_singles,
        )

        def _eval_skip(skip_layers: set[int]) -> dict[str, Any]:
            if method == "samd":
                model.cache = None
                if hasattr(model, "draft"):
                    model.draft.reset()
            try:
                return eval_fn(
                    model, items, skip_layers, baseline_hypotheses=baseline_hyp, domain=domain
                )
            finally:
                torch.cuda.empty_cache()

        def _on_err(layer: int, exc: Exception) -> None:
            print(f"  [max_accept] skip+{layer} failed: {exc}", flush=True)
            if method == "samd":
                model.cache = None
                if hasattr(model, "draft"):
                    model.draft.reset()
            torch.cuda.empty_cache()

        skip_set, history, current = max_accept_preserve_score_search(
            eval_fn=_eval_skip,
            baseline=baseline,
            config=macfg,
            on_trial_error=_on_err,
        )
        return {
            "method": method,
            "mode": "max_accept",
            "search_mode": search_mode,
            "selection_criterion": "max mean_accepted_per_step subject to task_score >= baseline",
            "score_tol_mode": score_tol_mode,
            "accept_metric": accept_metric,
            "score_drop_tol": score_drop_tol,
            "baseline": baseline,
            "best": current,
            "skip_layers": sorted(skip_set),
            "history": history,
        }

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
            score_key="task_score",
            accept_weight=accept_weight,
            score_weight=score_weight,
            layer_step=layer_step,
        )

        def _eval_skip(skip_layers: set[int]) -> dict[str, Any]:
            return eval_fn(model, items, skip_layers, baseline_hypotheses=baseline_hyp, domain=domain)

        def _on_err(layer: int, exc: Exception) -> None:
            print(f"  [sleb] skip+{layer} failed: {exc}", flush=True)

        skip_set, history, current = sleb_layer_search(
            eval_fn=_eval_skip,
            baseline=baseline,
            config=sleb_cfg,
            on_trial_error=_on_err,
        )
        return {
            "method": method,
            "mode": "sleb",
            "search_mode": search_mode,
            "score_tol_mode": score_tol_mode,
            "accept_mode": accept_mode,
            "accept_drop_tol": accept_drop_tol,
            "accept_metric": accept_metric,
            "accept_weight": accept_weight,
            "score_weight": score_weight,
            "early_barrier": early_barrier,
            "latter_barrier": latter_barrier,
            "pick_metric": pick_metric,
            "score_drop_tol": score_drop_tol,
            "baseline": baseline,
            "best": current,
            "history": history,
        }

    skip_set: set[int] = set()
    current = baseline
    history = [
        {
            "round": 0,
            "layer": None,
            "skip_layers": [],
            "accept_rate": init_accept,
            "task_score": init_score,
            "tok_per_s": baseline["tok_per_s"],
        }
    ]

    for rnd in range(1, max_rounds + 1):
        if len(skip_set) >= max_skip_layers:
            break
        best_layer = None
        best_metrics = None
        best_pick = -1.0
        for layer in range(0, num_layers, layer_step):
            if layer in skip_set:
                continue
            trial = set(skip_set)
            trial.add(layer)
            try:
                m = eval_fn(model, items, trial, baseline_hypotheses=baseline_hyp, domain=domain)
            except Exception as exc:
                print(f"  [round {rnd}] skip+{layer} failed: {exc}", flush=True)
                continue
            score_ok = _score_ok(m["task_score"], init_score, score_drop_tol, score_tol_mode)
            accept_key = "mean_accepted_per_step" if pick_metric == "mean_accepted" else "accept_rate"
            cur_accept_val = current.get(accept_key, current["accept_rate"])
            init_accept_val = init_mean_acc if pick_metric == "mean_accepted" else init_accept
            accept_ok = _accept_ok(
                m.get(accept_key, m["accept_rate"]),
                init_accept=init_accept_val,
                current_accept=cur_accept_val,
                mode=accept_mode,
                accept_drop_tol=accept_drop_tol,
            )
            if not (score_ok and accept_ok):
                continue
            pv = _pick_value(m, pick_metric)
            if best_metrics is None or pv > best_pick + 1e-12:
                best_pick = pv
                best_layer = layer
                best_metrics = m
        if best_layer is None:
            print(f"  [round {rnd}] no improving layer, stop", flush=True)
            break
        skip_set.add(best_layer)
        current = best_metrics
        history.append(
            {
                "round": rnd,
                "layer": best_layer,
                "skip_layers": sorted(skip_set),
                "accept_rate": current["accept_rate"],
                "task_score": current["task_score"],
                "tok_per_s": current["tok_per_s"],
                "delta_accept": current["accept_rate"] - init_accept,
                "delta_score": current["task_score"] - init_score,
            }
        )
        print(
            f"  [round {rnd}] +layer {best_layer} -> accept={current['accept_rate']:.4f} "
            f"score={current['task_score']:.4f} tok/s={current['tok_per_s']:.1f}",
            flush=True,
        )

    return {
        "method": method,
        "mode": "greedy",
        "search_mode": search_mode,
        "score_tol_mode": score_tol_mode,
        "accept_mode": accept_mode,
        "accept_drop_tol": accept_drop_tol,
        "pick_metric": pick_metric,
        "score_drop_tol": score_drop_tol,
        "baseline": baseline,
        "best": current,
        "history": history,
    }


def main() -> None:
    args = parse_args()
    if args.method == "samd" and args.size == "13b":
        raise SystemExit("SAMD/EAGLE-2 skip search is only supported for Vicuna-7B/33B")
    out_dir = ensure_dir(args.output_dir)
    size_cfg = SIZE_PRESETS[args.size]
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
                prompt=wrap_vicuna_prompt(_extract_user_text(item.prompt)),
                max_tokens=item.max_tokens,
                category=item.category,
                reference=item.reference,
            )
            for item in raw
        ]
    else:
        items = load_items(
            args.dataset, num_requests=args.num_requests, seed=args.seed, output_len=args.output_len
        )
    vicuna_base = resolve_vicuna(args.size)

    print(
        f"Loading {args.method} model (Vicuna-{args.size} base={vicuna_base}) "
        f"search={args.search_mode} n={len(items)} "
        f"score_tol={args.score_drop_tol}({args.score_tol_mode})...",
        flush=True,
    )
    if args.method == "hydra":
        hydra_path = resolve_hydra_path(args.size)
        model = load_hydra_model(vicuna_base, hydra_path=hydra_path)
    else:
        eagle_path = size_cfg.get("eagle_path", EAGLE_PATHS["7b"])
        model = load_samd_model(vicuna_base, eagle_path=eagle_path, size=args.size)

    result = greedy_search(
        method=args.method,
        model=model,
        items=items,
        domain=domain,
        num_layers=size_cfg["num_layers"],
        layer_step=size_cfg["layer_step"],
        layer_step_override=args.layer_step,
        exhaustive_singles=args.exhaustive_singles,
        score_drop_tol=args.score_drop_tol,
        score_tol_mode=args.score_tol_mode,
        accept_mode=args.accept_mode,
        accept_drop_tol=args.accept_drop_tol,
        pick_metric=args.pick_metric,
        max_skip_layers=args.max_skip_layers,
        min_skips=args.min_skips,
        max_rounds=args.max_rounds,
        search_mode=args.search_mode,
        early_barrier=args.early_barrier,
        latter_barrier=args.latter_barrier,
        accept_metric=args.accept_metric,
        accept_weight=args.accept_weight,
        score_weight=args.score_weight,
        bridge_score_drop_tol=args.bridge_score_drop_tol,
        bridge_score_tol_mode=args.bridge_score_tol_mode,
        pareto_beam_width=args.pareto_beam_width,
        refine_top_k=args.refine_top_k,
        refine_max_evals=args.refine_max_evals,
    )
    result["dataset"] = args.dataset
    result["num_requests"] = args.num_requests
    result["output_len"] = args.output_len
    result["base_model"] = size_cfg["base_label"]
    result["size"] = args.size

    tag = f"{args.dataset}_{args.size}_{args.method}_skip_greedy"
    write_json(result, out_dir / f"{tag}.json")
    write_table(result["history"], out_dir / f"{tag}_history.csv")
    print(json.dumps({"dataset": args.dataset, "method": args.method, "best": result["best"]}, indent=2))

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
