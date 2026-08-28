#!/usr/bin/env python3
"""TSS metrics-balanced pipeline: train=16 search, test=64 heldout.

Hard constraint: task_score >= baseline (metrics-preserving).
Explore by maximizing metrics; retain max_metrics / max_skip / max_accept;
final pick = most balanced of the three.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data" / "Spec-Bench-repo"))

from spec_exp.benchmark_config import SCORE_CATEGORY
from spec_exp.benchmark_datasets import load_dataset_split
from spec_exp.io import ensure_dir, write_json
from spec_exp.self_spec_decode import DecodeItem
from spec_exp.transformers_compat import install_transformers_compat

DOMAINS = ("translation", "summarization", "rag", "humaneval", "qa", "mmlu")
N_LAYERS = {"7b": 32, "13b": 40}
V3_EAGLE7_SEEDS: dict[str, tuple[tuple[int, ...], ...]] = {
    "translation": ((3, 11, 15, 25, 30), (8, 13, 25, 30), (14, 27, 28, 29, 30)),
    "summarization": ((3, 12), (5, 11, 25), (5, 15, 23)),
    "rag": ((3, 6, 14, 25, 30), (3, 15, 27, 28, 30), (2,), (2, 12, 23, 29)),
    "humaneval": ((15, 20), (15, 27)),
    "qa": ((4, 9, 19, 28), (5, 10, 21, 31)),
    "mmlu": ((6, 7, 8, 14, 23),),
}


@dataclass(frozen=True)
class PipelineSearchOptions:
    mode: str
    bridge_score_drop_tol: float
    bridge_score_tol_mode: str
    pareto_beam_width: int
    refine_top_k: int
    refine_max_evals: int
    early_barrier: int
    latter_barrier: int
    enable_accept_track: bool
    enable_refine: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--jobs",
        default="7b_samd,13b_eagle3",
        help="Comma list: 7b_samd,7b_hydra,7b_eagle,13b_hydra,13b_eagle3",
    )
    p.add_argument("--datasets", default=",".join(DOMAINS))
    p.add_argument("--train-size", type=int, default=16)
    p.add_argument("--output-len", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-skip-layers", type=int, default=12)
    p.add_argument(
        "--output-dir",
        default=str(REPO / "results" / "tss_metrics_balanced_train16_20260712"),
    )
    p.add_argument("--accept-drop-tol", type=float, default=0.0)
    p.add_argument("--score-drop-tol", type=float, default=0.05)
    p.add_argument("--early-barrier", type=int, default=2)
    p.add_argument("--latter-barrier", type=int, default=0)
    p.add_argument(
        "--search-mode",
        choices=["max_metrics_balanced", "pareto_bridge_v2", "tri_objective_v3"],
        default="max_metrics_balanced",
    )
    p.add_argument("--bridge-score-drop-tol", type=float, default=0.05)
    p.add_argument(
        "--bridge-score-tol-mode",
        choices=["absolute", "relative"],
        default="relative",
    )
    p.add_argument("--pareto-beam-width", type=int, default=2)
    p.add_argument("--refine-top-k", type=int, default=4)
    p.add_argument("--refine-max-evals", type=int, default=80)
    p.add_argument("--disable-accept-track", action="store_true")
    p.add_argument("--disable-refine", action="store_true")
    p.add_argument("--skip-search", action="store_true")
    p.add_argument("--skip-heldout", action="store_true")
    return p.parse_args()


def search_options_for_dataset(
    args: argparse.Namespace, dataset: str
) -> PipelineSearchOptions:
    if args.search_mode in {"pareto_bridge_v2", "tri_objective_v3"} and dataset in {"humaneval", "mmlu"}:
        bridge_tol = 1.0 / max(1, args.train_size)
        bridge_mode = "absolute"
    else:
        bridge_tol = args.bridge_score_drop_tol
        bridge_mode = args.bridge_score_tol_mode
    return PipelineSearchOptions(
        mode=args.search_mode,
        bridge_score_drop_tol=bridge_tol,
        bridge_score_tol_mode=bridge_mode,
        pareto_beam_width=args.pareto_beam_width,
        refine_top_k=args.refine_top_k,
        refine_max_evals=args.refine_max_evals,
        early_barrier=args.early_barrier,
        latter_barrier=args.latter_barrier,
        enable_accept_track=not args.disable_accept_track,
        enable_refine=not args.disable_refine,
    )


def search_config_payload(
    *,
    args: argparse.Namespace,
    dataset: str,
    job: str,
    options: PipelineSearchOptions,
) -> dict[str, Any]:
    payload = {
        "job": job,
        "dataset": dataset,
        "seed": args.seed,
        "train_size": args.train_size,
        "output_len": args.output_len,
        "max_skip_layers": args.max_skip_layers,
        **asdict(options),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["config_hash"] = hashlib.sha256(encoded).hexdigest()[:16]
    return payload


def wrap_vicuna(user: str) -> str:
    from fastchat.model import get_conversation_template

    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], user)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _extract_user(prompt: str) -> str:
    if "<|im_start|>user" in prompt:
        return prompt.split("<|im_start|>user\n", 1)[1].split("\n<|im_start|>assistant")[0]
    return prompt


def load_split(dataset: str, split: str, *, train_size: int, seed: int, output_len: int) -> list[DecodeItem]:
    # Keep every domain on the same 80-example protocol. Some loaders (for
    # example Natural Questions) expose thousands of examples, whereas the
    # original Spec-Bench domains contain exactly 80.
    all_items = load_dataset_split(
        dataset, split="all", train_size=train_size, seed=seed, output_len=output_len
    )[:80]
    if split == "train":
        raw = all_items[:train_size]
    elif split == "test":
        raw = all_items[train_size:]
    else:
        raw = all_items
    return [
        DecodeItem(
            request_id=it.request_id,
            prompt=wrap_vicuna(_extract_user(it.prompt)),
            max_tokens=it.max_tokens,
            category=it.category,
            reference=it.reference,
        )
        for it in raw
    ]


def strip_hyp(m: dict[str, Any]) -> dict[str, Any]:
    out = dict(m)
    out.pop("hypotheses", None)
    return out


def cuda_cleanup(*objs: Any) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    time.sleep(3)


def run_7b_samd_search(
    dataset: str,
    items: list[DecodeItem],
    *,
    max_skip: int,
    out_json: Path,
    search_options: PipelineSearchOptions,
) -> dict:
    from scripts.run_hydra_samd_skip_greedy import (
        EAGLE_PATHS,
        SIZE_PRESETS,
        greedy_search,
        load_samd_model,
        resolve_vicuna,
    )

    domain = SCORE_CATEGORY[dataset]
    size_cfg = SIZE_PRESETS["7b"]
    model = load_samd_model(
        resolve_vicuna("7b"), eagle_path=EAGLE_PATHS["7b"], size="7b"
    )
    result = greedy_search(
        method="samd",
        model=model,
        items=items,
        domain=domain,
        num_layers=size_cfg["num_layers"],
        layer_step=1,
        layer_step_override=1,
        exhaustive_singles=False,
        score_drop_tol=0.05,
        score_tol_mode="relative",
        accept_mode="baseline",
        accept_drop_tol=0.0,
        pick_metric="tok_per_s",
        max_skip_layers=max_skip,
        max_rounds=max_skip,
        search_mode=search_options.mode,
        early_barrier=search_options.early_barrier,
        latter_barrier=search_options.latter_barrier,
        accept_metric="mean_accepted_per_step",
        bridge_score_drop_tol=search_options.bridge_score_drop_tol,
        bridge_score_tol_mode=search_options.bridge_score_tol_mode,
        pareto_beam_width=search_options.pareto_beam_width,
        refine_top_k=search_options.refine_top_k,
        refine_max_evals=search_options.refine_max_evals,
    )
    result.update(
        {
            "dataset": dataset,
            "domain": domain,
            "size": "7b",
            "method": "samd",
            "train_size": len(items),
            "split": "train",
        }
    )
    write_json(result, out_json)
    cuda_cleanup(model)
    return result


def run_7b_samd_heldout(
    dataset: str,
    items: list[DecodeItem],
    skip_layers: list[int],
    *,
    out_json: Path,
    train_result: dict,
) -> dict:
    from scripts.run_hydra_samd_skip_greedy import (
        EAGLE_PATHS,
        SIZE_PRESETS,
        eval_samd,
        load_samd_model,
        resolve_vicuna,
    )

    domain = SCORE_CATEGORY[dataset]
    size_cfg = SIZE_PRESETS["7b"]
    model = None
    try:
        model = load_samd_model(
            resolve_vicuna("7b"), eagle_path=EAGLE_PATHS["7b"], size="7b"
        )
        baseline = eval_samd(model, items, set(), baseline_hypotheses=None, domain=domain)
        baseline_hyp = baseline.pop("hypotheses", None)
        best = eval_samd(
            model, items, set(skip_layers), baseline_hypotheses=baseline_hyp, domain=domain
        )
        best.pop("hypotheses", None)
        payload = {
            "dataset": dataset,
            "domain": domain,
            "size": "7b",
            "method": "samd",
            "split": "test",
            "test_size": len(items),
            "skip_layers": list(skip_layers),
            "train_source": str(train_result.get("_path", "")),
            "test_eval": {
                "baseline": strip_hyp(baseline),
                "best": strip_hyp(best),
                "delta": {
                    "accept": best["mean_accepted_per_step"] - baseline["mean_accepted_per_step"],
                    "task_score": best["task_score"] - baseline["task_score"],
                    "tok_per_s": best["tok_per_s"] - baseline["tok_per_s"],
                },
            },
            "train_eval": {
                "baseline": strip_hyp(train_result.get("baseline", {})),
                "best": strip_hyp(train_result.get("best", {})),
            },
        }
        write_json(payload, out_json)
        return payload
    finally:
        cuda_cleanup(model)


def run_7b_hydra_search(
    dataset: str,
    items: list[DecodeItem],
    *,
    max_skip: int,
    out_json: Path,
    search_options: PipelineSearchOptions,
) -> dict:
    from scripts.run_hydra_samd_skip_greedy import (
        SIZE_PRESETS,
        greedy_search,
        load_hydra_model,
        resolve_hydra_path,
        resolve_vicuna,
    )

    domain = SCORE_CATEGORY[dataset]
    size_cfg = SIZE_PRESETS["7b"]
    model = load_hydra_model(
        resolve_vicuna("7b"), hydra_path=resolve_hydra_path("7b")
    )
    result = greedy_search(
        method="hydra",
        model=model,
        items=items,
        domain=domain,
        num_layers=size_cfg["num_layers"],
        layer_step=1,
        layer_step_override=1,
        exhaustive_singles=False,
        score_drop_tol=0.05,
        score_tol_mode="relative",
        accept_mode="baseline",
        accept_drop_tol=0.0,
        pick_metric="tok_per_s",
        max_skip_layers=max_skip,
        max_rounds=max_skip,
        search_mode=search_options.mode,
        early_barrier=search_options.early_barrier,
        latter_barrier=search_options.latter_barrier,
        accept_metric="mean_accepted_per_step",
        bridge_score_drop_tol=search_options.bridge_score_drop_tol,
        bridge_score_tol_mode=search_options.bridge_score_tol_mode,
        pareto_beam_width=search_options.pareto_beam_width,
        refine_top_k=search_options.refine_top_k,
        refine_max_evals=search_options.refine_max_evals,
    )
    result.update(
        {
            "dataset": dataset,
            "domain": domain,
            "size": "7b",
            "method": "hydra",
            "train_size": len(items),
            "split": "train",
        }
    )
    write_json(result, out_json)
    cuda_cleanup(model)
    return result


def run_7b_hydra_heldout(
    dataset: str,
    items: list[DecodeItem],
    skip_layers: list[int],
    *,
    out_json: Path,
    train_result: dict,
) -> dict:
    from scripts.run_hydra_samd_skip_greedy import (
        eval_hydra,
        load_hydra_model,
        resolve_hydra_path,
        resolve_vicuna,
    )

    domain = SCORE_CATEGORY[dataset]
    model = None
    try:
        model = load_hydra_model(
            resolve_vicuna("7b"), hydra_path=resolve_hydra_path("7b")
        )
        baseline = eval_hydra(model, items, set(), baseline_hypotheses=None, domain=domain)
        baseline_hyp = baseline.pop("hypotheses", None)
        best = eval_hydra(
            model, items, set(skip_layers), baseline_hypotheses=baseline_hyp, domain=domain
        )
        best.pop("hypotheses", None)
        payload = {
            "dataset": dataset,
            "domain": domain,
            "size": "7b",
            "method": "hydra",
            "split": "test",
            "test_size": len(items),
            "skip_layers": list(skip_layers),
            "train_source": str(train_result.get("_path", "")),
            "test_eval": {
                "baseline": strip_hyp(baseline),
                "best": strip_hyp(best),
                "delta": {
                    "accept": best["mean_accepted_per_step"] - baseline["mean_accepted_per_step"],
                    "task_score": best["task_score"] - baseline["task_score"],
                    "tok_per_s": best["tok_per_s"] - baseline["tok_per_s"],
                },
            },
            "train_eval": {
                "baseline": strip_hyp(train_result.get("baseline", {})),
                "best": strip_hyp(train_result.get("best", {})),
            },
        }
        write_json(payload, out_json)
        return payload
    finally:
        cuda_cleanup(model)


def run_13b_hydra_search(
    dataset: str,
    items: list[DecodeItem],
    *,
    max_skip: int,
    out_json: Path,
    search_options: PipelineSearchOptions,
) -> dict:
    from scripts.run_hydra_samd_skip_greedy import (
        SIZE_PRESETS,
        greedy_search,
        load_hydra_model,
        resolve_hydra_path,
        resolve_vicuna,
    )

    domain = SCORE_CATEGORY[dataset]
    size_cfg = SIZE_PRESETS["13b"]
    model = load_hydra_model(
        resolve_vicuna("13b"), hydra_path=resolve_hydra_path("13b")
    )
    result = greedy_search(
        method="hydra",
        model=model,
        items=items,
        domain=domain,
        num_layers=size_cfg["num_layers"],
        layer_step=1,
        layer_step_override=1,
        exhaustive_singles=False,
        score_drop_tol=0.05,
        score_tol_mode="relative",
        accept_mode="baseline",
        accept_drop_tol=0.0,
        pick_metric="tok_per_s",
        max_skip_layers=max_skip,
        max_rounds=max_skip,
        search_mode=search_options.mode,
        early_barrier=search_options.early_barrier,
        latter_barrier=search_options.latter_barrier,
        accept_metric="mean_accepted_per_step",
        bridge_score_drop_tol=search_options.bridge_score_drop_tol,
        bridge_score_tol_mode=search_options.bridge_score_tol_mode,
        pareto_beam_width=search_options.pareto_beam_width,
        refine_top_k=search_options.refine_top_k,
        refine_max_evals=search_options.refine_max_evals,
    )
    result.update(
        {
            "dataset": dataset,
            "domain": domain,
            "size": "13b",
            "method": "hydra",
            "train_size": len(items),
            "split": "train",
        }
    )
    write_json(result, out_json)
    cuda_cleanup(model)
    return result


def run_13b_hydra_heldout(
    dataset: str,
    items: list[DecodeItem],
    skip_layers: list[int],
    *,
    out_json: Path,
    train_result: dict,
) -> dict:
    from scripts.run_hydra_samd_skip_greedy import (
        eval_hydra,
        load_hydra_model,
        resolve_hydra_path,
        resolve_vicuna,
    )

    domain = SCORE_CATEGORY[dataset]
    model = None
    try:
        model = load_hydra_model(
            resolve_vicuna("13b"), hydra_path=resolve_hydra_path("13b")
        )
        baseline = eval_hydra(model, items, set(), baseline_hypotheses=None, domain=domain)
        baseline_hyp = baseline.pop("hypotheses", None)
        best = eval_hydra(
            model, items, set(skip_layers), baseline_hypotheses=baseline_hyp, domain=domain
        )
        best.pop("hypotheses", None)
        payload = {
            "dataset": dataset,
            "domain": domain,
            "size": "13b",
            "method": "hydra",
            "split": "test",
            "test_size": len(items),
            "skip_layers": list(skip_layers),
            "train_source": str(train_result.get("_path", "")),
            "test_eval": {
                "baseline": strip_hyp(baseline),
                "best": strip_hyp(best),
                "delta": {
                    "accept": best["mean_accepted_per_step"] - baseline["mean_accepted_per_step"],
                    "task_score": best["task_score"] - baseline["task_score"],
                    "tok_per_s": best["tok_per_s"] - baseline["tok_per_s"],
                },
            },
            "train_eval": {
                "baseline": strip_hyp(train_result.get("baseline", {})),
                "best": strip_hyp(train_result.get("best", {})),
            },
        }
        write_json(payload, out_json)
        return payload
    finally:
        cuda_cleanup(model)


def run_7b_eagle_search(
    dataset: str,
    items: list[DecodeItem],
    *,
    max_skip: int,
    out_json: Path,
    search_options: PipelineSearchOptions,
) -> dict:
    """Classic EAGLE (not EAGLE3) on Vicuna-7B."""
    from scripts.run_vicuna13_eagle3_skip_sweep import (
        MODEL_PRESETS,
        load_model,
        max_metrics_balanced_mode,
        pareto_bridge_v2_mode,
        tri_objective_v3_mode,
    )

    domain = SCORE_CATEGORY[dataset]
    cfg = MODEL_PRESETS["vicuna7"]
    # refresh base path in case HF cache moved
    from scripts.run_vicuna13_eagle3_skip_sweep import _resolve_vicuna7_base

    cfg = dict(cfg)
    cfg["base_model"] = _resolve_vicuna7_base()
    model = load_model(
        base_model=cfg["base_model"],
        ea_model=cfg["ea_model"],
        total_token=60,
        use_eagle3=False,
    )
    common = {
        "domain": domain,
        "num_layers": cfg["num_layers"],
        "accept_metric": "mean_accepted_per_step",
        "max_skip_layers": max_skip,
        "early_barrier": search_options.early_barrier,
        "latter_barrier": search_options.latter_barrier,
    }
    if search_options.mode == "pareto_bridge_v2":
        result = pareto_bridge_v2_mode(
            model,
            items,
            **common,
            bridge_score_drop_tol=search_options.bridge_score_drop_tol,
            bridge_score_tol_mode=search_options.bridge_score_tol_mode,
            pareto_beam_width=search_options.pareto_beam_width,
            refine_top_k=search_options.refine_top_k,
            refine_max_evals=search_options.refine_max_evals,
        )
    elif search_options.mode == "tri_objective_v3":
        result = tri_objective_v3_mode(
            model,
            items,
            **common,
            bridge_score_drop_tol=search_options.bridge_score_drop_tol,
            bridge_score_tol_mode=search_options.bridge_score_tol_mode,
            pareto_beam_width=search_options.pareto_beam_width,
            refine_top_k=search_options.refine_top_k,
            refine_max_evals=search_options.refine_max_evals,
            seed_sets=V3_EAGLE7_SEEDS.get(dataset, ()),
            enable_accept_track=search_options.enable_accept_track,
            enable_refine=search_options.enable_refine,
        )
    else:
        result = max_metrics_balanced_mode(model, items, **common)
    result.update(
        {
            "dataset": dataset,
            "domain": domain,
            "size": "7b",
            "method": "eagle",
            "preset": "vicuna7",
            "train_size": len(items),
            "split": "train",
        }
    )
    write_json(result, out_json)
    cuda_cleanup(model)
    return result


def run_7b_eagle_heldout(
    dataset: str,
    items: list[DecodeItem],
    skip_layers: list[int],
    *,
    out_json: Path,
    train_result: dict,
) -> dict:
    from scripts.run_vicuna13_eagle3_skip_sweep import (
        MODEL_PRESETS,
        _collect_hypotheses,
        _resolve_vicuna7_base,
        eval_skip_config,
        load_model,
    )

    domain = SCORE_CATEGORY[dataset]
    cfg = dict(MODEL_PRESETS["vicuna7"])
    cfg["base_model"] = _resolve_vicuna7_base()
    model = None
    try:
        model = load_model(
            base_model=cfg["base_model"],
            ea_model=cfg["ea_model"],
            total_token=60,
            use_eagle3=False,
        )
        baseline = eval_skip_config(
            model, items, set(), baseline_hypotheses=None, domain=domain
        )
        baseline_hyp = _collect_hypotheses(model, items, set())
        best = eval_skip_config(
            model, items, set(skip_layers), baseline_hypotheses=baseline_hyp, domain=domain
        )
        payload = {
            "dataset": dataset,
            "domain": domain,
            "size": "7b",
            "method": "eagle",
            "preset": "vicuna7",
            "split": "test",
            "test_size": len(items),
            "skip_layers": list(skip_layers),
            "train_source": str(train_result.get("_path", "")),
            "test_eval": {
                "baseline": strip_hyp(baseline),
                "best": strip_hyp(best),
                "delta": {
                    "accept": best["mean_accepted_per_step"] - baseline["mean_accepted_per_step"],
                    "task_score": best["task_score"] - baseline["task_score"],
                    "tok_per_s": (best.get("tok_per_s") or 0) - (baseline.get("tok_per_s") or 0),
                },
            },
            "train_eval": {
                "baseline": strip_hyp(train_result.get("baseline", {})),
                "best": strip_hyp(train_result.get("best", {})),
            },
        }
        write_json(payload, out_json)
        return payload
    finally:
        cuda_cleanup(model)


def run_13b_eagle3_search(
    dataset: str,
    items: list[DecodeItem],
    *,
    max_skip: int,
    out_json: Path,
    search_options: PipelineSearchOptions,
) -> dict:
    from scripts.run_vicuna13_eagle3_skip_sweep import (
        MODEL_PRESETS,
        load_model,
        max_metrics_balanced_mode,
        pareto_bridge_v2_mode,
    )

    domain = SCORE_CATEGORY[dataset]
    cfg = MODEL_PRESETS["vicuna13"]
    model = load_model(
        base_model=cfg["base_model"],
        ea_model=cfg["ea_model"],
        total_token=60,
        use_eagle3=True,
    )
    common = {
        "domain": domain,
        "num_layers": cfg["num_layers"],
        "accept_metric": "mean_accepted_per_step",
        "max_skip_layers": max_skip,
        "early_barrier": search_options.early_barrier,
        "latter_barrier": search_options.latter_barrier,
    }
    if search_options.mode == "pareto_bridge_v2":
        result = pareto_bridge_v2_mode(
            model,
            items,
            **common,
            bridge_score_drop_tol=search_options.bridge_score_drop_tol,
            bridge_score_tol_mode=search_options.bridge_score_tol_mode,
            pareto_beam_width=search_options.pareto_beam_width,
            refine_top_k=search_options.refine_top_k,
            refine_max_evals=search_options.refine_max_evals,
        )
    else:
        result = max_metrics_balanced_mode(model, items, **common)
    result.update(
        {
            "dataset": dataset,
            "domain": domain,
            "size": "13b",
            "method": "eagle3",
            "preset": "vicuna13",
            "train_size": len(items),
            "split": "train",
        }
    )
    write_json(result, out_json)
    cuda_cleanup(model)
    return result


def run_13b_eagle3_heldout(
    dataset: str,
    items: list[DecodeItem],
    skip_layers: list[int],
    *,
    out_json: Path,
    train_result: dict,
) -> dict:
    from scripts.run_vicuna13_eagle3_skip_sweep import (
        MODEL_PRESETS,
        _collect_hypotheses,
        eval_skip_config,
        load_model,
    )

    domain = SCORE_CATEGORY[dataset]
    cfg = MODEL_PRESETS["vicuna13"]
    model = None
    try:
        model = load_model(
            base_model=cfg["base_model"],
            ea_model=cfg["ea_model"],
            total_token=60,
            use_eagle3=True,
        )
        baseline = eval_skip_config(model, items, set(), baseline_hypotheses=None, domain=domain)
        baseline_hyp = _collect_hypotheses(model, items, set())
        best = eval_skip_config(
            model, items, set(skip_layers), baseline_hypotheses=baseline_hyp, domain=domain
        )
        payload = {
            "dataset": dataset,
            "domain": domain,
            "size": "13b",
            "method": "eagle3",
            "split": "test",
            "test_size": len(items),
            "skip_layers": list(skip_layers),
            "test_eval": {
                "baseline": strip_hyp(baseline),
                "best": strip_hyp(best),
                "delta": {
                    "accept": best["mean_accepted_per_step"] - baseline["mean_accepted_per_step"],
                    "task_score": best["task_score"] - baseline["task_score"],
                    "tok_per_s": (best.get("tok_per_s") or 0) - (baseline.get("tok_per_s") or 0),
                },
            },
            "train_eval": {
                "baseline": strip_hyp(train_result.get("baseline", {})),
                "best": strip_hyp(train_result.get("best", {})),
            },
        }
        write_json(payload, out_json)
        return payload
    finally:
        cuda_cleanup(model)


def main() -> None:
    install_transformers_compat()
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf-cache")
    args = parse_args()
    out = ensure_dir(args.output_dir)
    search_dir = ensure_dir(out / "search")
    held_dir = ensure_dir(out / "heldout")
    jobs = [j.strip() for j in args.jobs.split(",") if j.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    summary: list[dict] = []
    t_all = time.perf_counter()

    for job in jobs:
        for ds in datasets:
            print(f"\n======== JOB {job} DATASET {ds} ========", flush=True)
            try:
                search_options = search_options_for_dataset(args, ds)
                expected_config = search_config_payload(
                    args=args,
                    dataset=ds,
                    job=job,
                    options=search_options,
                )
                expected_hash = expected_config["config_hash"]
                train_items = load_split(
                    ds, "train", train_size=args.train_size, seed=args.seed, output_len=args.output_len
                )
                test_items = load_split(
                    ds, "test", train_size=args.train_size, seed=args.seed, output_len=args.output_len
                )
                assert len(train_items) == args.train_size, (len(train_items), args.train_size)
                assert len(test_items) == 80 - args.train_size, (len(test_items), 80 - args.train_size)

                search_json = search_dir / f"{ds}_{job}_{args.search_mode}.json"
                held_json = held_dir / f"{ds}_{job}_heldout.json"

                _reuse = False
                if held_json.exists() and not args.skip_heldout:
                    held = json.loads(held_json.read_text())
                    if held.get("pipeline_search_config_hash") == expected_hash:
                        _reuse = True
                    else:
                        print(
                            f"  ignore incompatible heldout {held_json} "
                            f"(expected config {expected_hash})",
                            flush=True,
                        )
                if _reuse:
                    print(f"  reuse heldout {held_json} (skip search)", flush=True)
                    train_result = {}
                    if search_json.exists():
                        train_result = json.loads(search_json.read_text())
                    skip = list(
                        held.get("skip_layers")
                        or train_result.get("skip_layers")
                        or train_result.get("best", {}).get("skip_layers")
                        or []
                    )
                else:
                    if not args.skip_search:
                        search_compatible = False
                        if search_json.exists():
                            cached_search = json.loads(search_json.read_text())
                            search_compatible = (
                                cached_search.get("pipeline_search_config_hash")
                                == expected_hash
                            )
                        if search_compatible:
                            print(f"  reuse search {search_json}", flush=True)
                            train_result = cached_search
                        elif job == "7b_samd":
                            train_result = run_7b_samd_search(
                                ds,
                                train_items,
                                max_skip=args.max_skip_layers,
                                out_json=search_json,
                                search_options=search_options,
                            )
                        elif job == "7b_hydra":
                            train_result = run_7b_hydra_search(
                                ds,
                                train_items,
                                max_skip=args.max_skip_layers,
                                out_json=search_json,
                                search_options=search_options,
                            )
                        elif job == "13b_hydra":
                            train_result = run_13b_hydra_search(
                                ds,
                                train_items,
                                max_skip=args.max_skip_layers,
                                out_json=search_json,
                                search_options=search_options,
                            )
                        elif job == "7b_eagle":
                            train_result = run_7b_eagle_search(
                                ds,
                                train_items,
                                max_skip=args.max_skip_layers,
                                out_json=search_json,
                                search_options=search_options,
                            )
                        elif job == "13b_eagle3":
                            train_result = run_13b_eagle3_search(
                                ds,
                                train_items,
                                max_skip=args.max_skip_layers,
                                out_json=search_json,
                                search_options=search_options,
                            )
                        else:
                            raise ValueError(job)
                    else:
                        train_result = json.loads(search_json.read_text())

                    train_result["pipeline_search_config"] = expected_config
                    train_result["pipeline_search_config_hash"] = expected_hash
                    write_json(train_result, search_json)
                    train_result["_path"] = str(search_json)
                    skip = list(
                        train_result.get("skip_layers")
                        or train_result.get("best", {}).get("skip_layers")
                        or []
                    )
                    print(
                        f"  selected skip={skip} "
                        f"train tok/s {train_result.get('baseline', {}).get('tok_per_s')} -> "
                        f"{train_result.get('best', {}).get('tok_per_s')}",
                        flush=True,
                    )

                    if not args.skip_heldout:
                        if job == "7b_samd":
                            held = run_7b_samd_heldout(
                                ds, test_items, skip, out_json=held_json, train_result=train_result
                            )
                        elif job == "7b_hydra":
                            held = run_7b_hydra_heldout(
                                ds, test_items, skip, out_json=held_json, train_result=train_result
                            )
                        elif job == "13b_hydra":
                            held = run_13b_hydra_heldout(
                                ds, test_items, skip, out_json=held_json, train_result=train_result
                            )
                        elif job == "7b_eagle":
                            held = run_7b_eagle_heldout(
                                ds, test_items, skip, out_json=held_json, train_result=train_result
                            )
                        elif job == "13b_eagle3":
                            held = run_13b_eagle3_heldout(
                                ds, test_items, skip, out_json=held_json, train_result=train_result
                            )
                        else:
                            raise ValueError(job)
                    else:
                        held = json.loads(held_json.read_text())

                    held["schema_version"] = "heldout_eval_v2"
                    held["search_algorithm"] = {
                        "name": args.search_mode,
                        "version": train_result.get("algorithm", {}).get(
                            "version", "1"
                        ),
                    }
                    held["pipeline_search_config_hash"] = expected_hash
                    held["pipeline_search_config"] = expected_config
                    write_json(held, held_json)

                b, s = held["test_eval"]["baseline"], held["test_eval"]["best"]
                row = {
                    "job": job,
                    "dataset": ds,
                    "search_mode": args.search_mode,
                    "pipeline_search_config_hash": expected_hash,
                    "skip_layers": skip,
                    "sparsity": 100.0 * len(skip) / N_LAYERS[job.split("_")[0]],
                    "test_native_accept": b["mean_accepted_per_step"],
                    "test_tss_accept": s["mean_accepted_per_step"],
                    "test_native_score": b["task_score"],
                    "test_tss_score": s["task_score"],
                    "test_native_tok_per_s": b.get("tok_per_s"),
                    "test_tss_tok_per_s": s.get("tok_per_s"),
                    "coverage": train_result.get("coverage", {}),
                }
                summary.append(row)
                print(
                    f"  HELD-OUT native tok/s={b.get('tok_per_s'):.1f} accept={b['mean_accepted_per_step']:.2f} "
                    f"score={b['task_score']:.3f} | TSS tok/s={s.get('tok_per_s'):.1f} "
                    f"accept={s['mean_accepted_per_step']:.2f} score={s['task_score']:.3f} skip={skip}",
                    flush=True,
                )
            except Exception as exc:
                print(f"  [ERROR] {job}/{ds} failed: {type(exc).__name__}: {exc}", flush=True)
                err_path = out / "errors.jsonl"
                with err_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "job": job,
                                "dataset": ds,
                                "error": f"{type(exc).__name__}: {exc}",
                                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )
                        + "\n"
                    )
            finally:
                cuda_cleanup()
                # heartbeat for external watchdog
                (out / "heartbeat.json").write_text(
                    json.dumps(
                        {
                            "job": job,
                            "dataset": ds,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "pid": os.getpid(),
                            "done_pairs": [f"{r['job']}/{r['dataset']}" for r in summary],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    write_json({"summary": summary, "wall_s": time.perf_counter() - t_all}, out / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"DONE in {(time.perf_counter() - t_all)/3600:.2f}h -> {out}", flush=True)


if __name__ == "__main__":
    main()
