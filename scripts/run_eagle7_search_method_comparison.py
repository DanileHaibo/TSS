#!/usr/bin/env python3
"""Compare skip-layer search methods on Vicuna-7B + EAGLE (translation / MMLU).

Methods:
  SLEB, Random Search, Greedy Search, Accept-only, Metric-only,
  TSS Breadth Search (reuse existing held-out results; not re-run).

Fast protocol (≈2h): train=4 search, held-out test=64, output_len=96.
TSS Breadth Search metrics/evals are reused from the full train=16 runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from spec_exp.benchmark_config import SCORE_CATEGORY
from spec_exp.benchmark_datasets import load_dataset_split
from spec_exp.io import ensure_dir, write_json
from spec_exp.self_spec_decode import DecodeItem
from spec_exp.sleb_skip_search import (
    SlebSearchConfig,
    sleb_layer_search,
    _metric,
    _toks,
)
from spec_exp.tri_objective_search import three_win_feasible, tri_geomean_ratio

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

METHODS = (
    "sleb",
    "random",
    "greedy",
    "accept_only",
    "metric_only",
)

METHOD_DISPLAY = {
    "sleb": "SLEB",
    "random": "Random Search",
    "greedy": "Greedy Search",
    "accept_only": "Accept-only",
    "metric_only": "Metric-only",
    "tss_breadth": "TSS Breadth Search",
}

# Existing TSS Breadth winners (held-out already measured).
TSS_EXISTING = {
    "translation": {
        "skip_layers": [3, 25, 30],
        "heldout_path": (
            REPO
            / "results/tss_tri_objective_v3_train16_20260714/final_validation"
            / "translation_7b_eagle_heldout.json"
        ),
        # Source search that first surfaced the winning set in its beam history.
        "search_evaluations": 334,
        "search_source": (
            "results/tss_eagle7_explore_depth_train16_20260711/search/"
            "translation_7b_eagle_max_skip_latter.json"
        ),
    },
    "mmlu": {
        "skip_layers": [3, 7, 9, 14, 20],
        "heldout_path": (
            REPO
            / "results/tss_tri_objective_v3_train16_20260714/final_validation"
            / "mmlu_7b_eagle_heldout.json"
        ),
        "search_evaluations": 548,
        "search_source": (
            "results/tss_pareto_bridge_v2_train16_20260713/7b_eagle/search/"
            "mmlu_7b_eagle_pareto_bridge_v2.json"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="translation,mmlu")
    p.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma list of baseline methods to run (not including tss_breadth).",
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO / "results" / "eagle7_search_method_comparison_20260728"),
    )
    p.add_argument("--train-size", type=int, default=4)
    p.add_argument("--output-len", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-skip-layers", type=int, default=4)
    p.add_argument("--early-barrier", type=int, default=2)
    p.add_argument("--latter-barrier", type=int, default=0)
    p.add_argument(
        "--random-budget",
        type=int,
        default=48,
        help="Random Search eval budget (default 48 for ~2h runs)",
    )
    p.add_argument("--layer-step", type=int, default=1)
    p.add_argument("--skip-search", action="store_true")
    p.add_argument("--skip-heldout", action="store_true")
    p.add_argument("--table-only", action="store_true")
    return p.parse_args()


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


def load_split(
    dataset: str, split: str, *, train_size: int, seed: int, output_len: int
) -> list[DecodeItem]:
    all_items = load_dataset_split(
        dataset, split="all", train_size=train_size, seed=seed, output_len=output_len
    )[:80]
    raw = all_items[:train_size] if split == "train" else all_items[train_size:]
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


def count_search_evals(history: list[dict[str, Any]]) -> int:
    """Count non-baseline evaluated skip configs (unique sets preferred)."""
    seen: set[tuple[int, ...]] = set()
    n = 0
    for row in history:
        phase = row.get("phase")
        if phase in {"baseline", None} and not row.get("skip_layers"):
            # baseline row or SLEB round-0
            if row.get("round") == 0 or phase == "baseline":
                continue
        skip = tuple(row.get("skip_layers") or [])
        if not skip:
            continue
        if skip in seen:
            continue
        seen.add(skip)
        n += 1
    # SLEB history may not mark phase; fall back to len-1
    if n == 0 and len(history) > 1:
        return max(0, len(history) - 1)
    return n


def pick_three_win_or_best(
    history: list[dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
) -> tuple[list[int], dict[str, Any]]:
    best_tw: dict[str, Any] | None = None
    best_tw_ratio = -math.inf
    best_any: dict[str, Any] | None = None
    best_any_accept = -math.inf
    for row in history:
        skip = row.get("skip_layers") or []
        if not skip:
            continue
        entry = dict(row)
        if three_win_feasible(entry, baseline, config):
            ratio = tri_geomean_ratio(entry, baseline, config)
            if ratio > best_tw_ratio:
                best_tw_ratio = ratio
                best_tw = entry
        acc = _metric(entry, config.accept_metric)
        if acc > best_any_accept:
            best_any_accept = acc
            best_any = entry
    chosen = best_tw or best_any or dict(baseline)
    return list(chosen.get("skip_layers") or []), chosen


def random_search(
    *,
    eval_fn: Callable[[set[int]], dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    budget: int,
    seed: int,
) -> tuple[set[int], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    lo = max(0, config.early_barrier)
    hi = config.num_layers - max(0, config.latter_barrier)
    pool = list(range(lo, hi))
    history: list[dict[str, Any]] = [
        {
            "phase": "baseline",
            "round": 0,
            "skip_layers": [],
            config.accept_metric: _metric(baseline, config.accept_metric),
            "task_score": _metric(baseline, "task_score"),
            "tok_per_s": _toks(baseline),
            "accept_rate": baseline.get("accept_rate"),
            "mean_accepted_per_step": baseline.get("mean_accepted_per_step"),
        }
    ]
    cache: dict[frozenset[int], dict[str, Any]] = {frozenset(): baseline}
    evaluated = 0
    # Prefer a mix of cardinalities 1..max_skip.
    while evaluated < budget:
        k = rng.randint(1, min(config.max_skip_layers, len(pool)))
        skip = frozenset(rng.sample(pool, k))
        if skip in cache:
            continue
        try:
            m = eval_fn(set(skip))
        except Exception as exc:
            print(f"  [random] skip={sorted(skip)} failed: {exc}", flush=True)
            continue
        cache[skip] = m
        evaluated += 1
        history.append(
            {
                "phase": "random",
                "round": evaluated,
                "skip_layers": sorted(skip),
                config.accept_metric: _metric(m, config.accept_metric),
                "task_score": _metric(m, "task_score"),
                "tok_per_s": _toks(m),
                "accept_rate": m.get("accept_rate"),
                "mean_accepted_per_step": m.get("mean_accepted_per_step"),
            }
        )
        if evaluated % 20 == 0:
            print(f"  [random] evaluated={evaluated}/{budget}", flush=True)

    # Random Search with the same three-win feasibility as TSS, but no structured
    # exploration. If no random sample is three-win on train, fall back to []
    # (do NOT silently maximize accept — that makes Random look like Accept-only).
    best_tw: dict[str, Any] | None = None
    best_ratio = -math.inf
    for row in history:
        skip = row.get("skip_layers") or []
        if not skip:
            continue
        if not three_win_feasible(row, baseline, config):
            continue
        ratio = tri_geomean_ratio(row, baseline, config)
        if ratio > best_ratio:
            best_ratio = ratio
            best_tw = row
    if best_tw is None:
        print(
            "  [random] no three-win sample in budget; fall back to skip=[]",
            flush=True,
        )
        return set(), history, dict(baseline)
    print(
        f"  [random] best three-win skip={best_tw.get('skip_layers')} "
        f"tri_ratio={best_ratio:.3f}",
        flush=True,
    )
    return set(best_tw["skip_layers"]), history, best_tw


def _objective_greedy_search(
    *,
    eval_fn: Callable[[set[int]], dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    objective: str,
    phase: str,
    on_trial_error: Callable[[int, Exception], None] | None = None,
    require_strict_gain: bool = True,
    always_fill_to_max: bool = False,
) -> tuple[set[int], list[dict[str, Any]], dict[str, Any]]:
    """Forward greedy on a single objective.

    objective:
      - accept: maximize mean_accepted_per_step
      - score: maximize task_score
      - num_skip: maximize |skip| (always fill to max_skip_layers)
    Other metrics are ignored entirely.
    """
    assert objective in {"accept", "score", "num_skip"}
    lo = max(0, config.early_barrier)
    hi = config.num_layers - max(0, config.latter_barrier)
    step = max(1, config.layer_step)
    skip_set: set[int] = set()
    current = dict(baseline)
    history: list[dict[str, Any]] = [
        {
            "phase": "baseline",
            "round": 0,
            "skip_layers": [],
            config.accept_metric: _metric(baseline, config.accept_metric),
            "task_score": _metric(baseline, "task_score"),
            "tok_per_s": _toks(baseline),
            "accept_rate": baseline.get("accept_rate"),
            "mean_accepted_per_step": baseline.get("mean_accepted_per_step"),
        }
    ]

    def _obj(m: dict[str, Any], skip: list[int] | set[int]) -> float:
        if objective == "accept":
            return _metric(m, config.accept_metric)
        if objective == "score":
            return _metric(m, "task_score")
        return float(len(skip))

    for rnd in range(1, config.max_skip_layers + 1):
        best_layer: int | None = None
        best_m: dict[str, Any] | None = None
        best_val = -math.inf
        # For num_skip, still evaluate candidates; pick any (prefer higher accept only as
        # a deterministic tie-break so the path is reproducible).
        for layer in range(lo, hi, step):
            if layer in skip_set:
                continue
            trial = set(skip_set) | {layer}
            try:
                m = eval_fn(trial)
            except Exception as exc:
                if on_trial_error:
                    on_trial_error(layer, exc)
                continue
            val = _obj(m, trial)
            history.append(
                {
                    "phase": phase,
                    "round": rnd,
                    "layer": layer,
                    "skip_layers": sorted(trial),
                    config.accept_metric: _metric(m, config.accept_metric),
                    "task_score": _metric(m, "task_score"),
                    "tok_per_s": _toks(m),
                    "accept_rate": m.get("accept_rate"),
                    "mean_accepted_per_step": m.get("mean_accepted_per_step"),
                }
            )
            # num_skip: all candidates have same |S|; use layer id for determinism.
            tie = float(-layer) if objective == "num_skip" else 0.0
            if (val, tie) > (best_val, float(-best_layer) if best_layer is not None and objective == "num_skip" else 0.0):
                best_val = val
                best_layer = layer
                best_m = m
        if best_layer is None or best_m is None:
            break
        if (
            require_strict_gain
            and not always_fill_to_max
            and best_val <= _obj(current, skip_set) + 1e-12
        ):
            print(f"  [{phase} round {rnd}] no {objective} gain, stop", flush=True)
            break
        skip_set.add(best_layer)
        current = best_m
        print(
            f"  [{phase} round {rnd}] +{best_layer} |S|={len(skip_set)} "
            f"accept={_metric(current, config.accept_metric):.3f} "
            f"score={_metric(current, 'task_score'):.4f}",
            flush=True,
        )

    # Final pick: pure max of the single objective over all evaluated configs.
    best_row = None
    best_key = (-math.inf, -math.inf)
    for row in history:
        skip = row.get("skip_layers") or []
        if objective == "num_skip":
            # Prefer deepest set; ignore accept/score for selection.
            key = (float(len(skip)), 0.0)
        elif objective == "accept":
            key = (_metric(row, config.accept_metric), float(len(skip)))
        else:
            key = (_metric(row, "task_score"), float(len(skip)))
        if not skip and objective != "num_skip":
            continue
        if key > best_key:
            best_key = key
            best_row = row
    if objective == "num_skip" and skip_set:
        # Always return the filled skip set from the greedy path.
        current = dict(current)
        current["skip_layers"] = sorted(skip_set)
        return set(skip_set), history, current
    if best_row is not None:
        return set(best_row["skip_layers"]), history, best_row
    return set(skip_set), history, current


def max_skip_greedy_search(
    *,
    eval_fn: Callable[[set[int]], dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    on_trial_error: Callable[[int, Exception], None] | None = None,
) -> tuple[set[int], list[dict[str, Any]], dict[str, Any]]:
    """Greedy: pack as many skips as allowed; ignore accept/metric."""
    return _objective_greedy_search(
        eval_fn=eval_fn,
        baseline=baseline,
        config=config,
        objective="num_skip",
        phase="greedy",
        on_trial_error=on_trial_error,
        require_strict_gain=False,
        always_fill_to_max=True,
    )


def accept_only_search(
    *,
    eval_fn: Callable[[set[int]], dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    on_trial_error: Callable[[int, Exception], None] | None = None,
) -> tuple[set[int], list[dict[str, Any]], dict[str, Any]]:
    """Maximize accept only; score/throughput may collapse."""
    return _objective_greedy_search(
        eval_fn=eval_fn,
        baseline=baseline,
        config=config,
        objective="accept",
        phase="accept_only",
        on_trial_error=on_trial_error,
        require_strict_gain=True,
        always_fill_to_max=False,
    )


def metric_only_search(
    *,
    eval_fn: Callable[[set[int]], dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    on_trial_error: Callable[[int, Exception], None] | None = None,
) -> tuple[set[int], list[dict[str, Any]], dict[str, Any]]:
    """Maximize task metric only; accept/skip count may collapse."""
    return _objective_greedy_search(
        eval_fn=eval_fn,
        baseline=baseline,
        config=config,
        objective="score",
        phase="metric_only",
        on_trial_error=on_trial_error,
        require_strict_gain=True,
        always_fill_to_max=False,
    )


def run_search_method(
    *,
    method: str,
    model: Any,
    items: list[DecodeItem],
    domain: str,
    num_layers: int,
    max_skip_layers: int,
    early_barrier: int,
    latter_barrier: int,
    random_budget: int,
    seed: int,
    layer_step: int = 1,
    baseline: dict[str, Any] | None = None,
    baseline_hyp: dict[str, str] | None = None,
) -> dict[str, Any]:
    import torch
    from run_vicuna13_eagle3_skip_sweep import _collect_hypotheses, eval_skip_config

    if baseline is None:
        print(f"[{method}] baseline (no skip)...", flush=True)
        baseline = eval_skip_config(model, items, set(), baseline_hypotheses=None, domain=domain)
    if baseline_hyp is None:
        print(f"[{method}] collect baseline hypotheses...", flush=True)
        baseline_hyp = _collect_hypotheses(model, items, set())

    cfg = SlebSearchConfig(
        num_layers=num_layers,
        max_skip_layers=max_skip_layers,
        early_barrier=early_barrier,
        latter_barrier=latter_barrier,
        accept_drop_tol=0.05 if method == "sleb" else 1.0,
        score_drop_tol=0.05 if method == "sleb" else 1.0,
        score_tol_mode="relative",
        accept_metric="mean_accepted_per_step",
        score_key="task_score",
        layer_step=layer_step,
        exhaustive_singles=True,
    )

    eval_counter = {"n": 0}
    cache: dict[frozenset[int], dict[str, Any]] = {}

    def _eval_skip(skip_layers: set[int]) -> dict[str, Any]:
        key = frozenset(skip_layers)
        if key in cache:
            return cache[key]
        try:
            m = eval_skip_config(
                model, items, skip_layers, baseline_hypotheses=baseline_hyp, domain=domain
            )
        finally:
            torch.cuda.empty_cache()
        cache[key] = m
        eval_counter["n"] += 1
        return m

    def _on_err(layer: int, exc: Exception) -> None:
        print(f"  [{method}] skip+{layer} failed: {exc}", flush=True)
        torch.cuda.empty_cache()

    t0 = time.time()
    if method == "sleb":
        skip_set, history, current = sleb_layer_search(
            eval_fn=_eval_skip,
            baseline=baseline,
            config=cfg,
            on_trial_error=_on_err,
        )
        # Prefer three-win among SLEB path history; else SLEB's own current.
        skip_layers, picked = pick_three_win_or_best(history, baseline, cfg)
        if not skip_layers:
            skip_layers = sorted(skip_set)
        else:
            current = picked
            skip_set = set(skip_layers)
    elif method == "greedy":
        skip_set, history, current = max_skip_greedy_search(
            eval_fn=_eval_skip,
            baseline=baseline,
            config=cfg,
            on_trial_error=_on_err,
        )
    elif method == "accept_only":
        skip_set, history, current = accept_only_search(
            eval_fn=_eval_skip,
            baseline=baseline,
            config=cfg,
            on_trial_error=_on_err,
        )
    elif method == "metric_only":
        skip_set, history, current = metric_only_search(
            eval_fn=_eval_skip,
            baseline=baseline,
            config=cfg,
            on_trial_error=_on_err,
        )
    elif method == "random":
        skip_set, history, current = random_search(
            eval_fn=_eval_skip,
            baseline=baseline,
            config=cfg,
            budget=random_budget,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown method {method}")

    wall = time.time() - t0
    n_evals = eval_counter["n"]
    # Prefer explicit skip from search return.
    skip_layers = sorted(skip_set) if skip_set else list(current.get("skip_layers") or [])
    payload = {
        "mode": method,
        "display_name": METHOD_DISPLAY[method],
        "selection_criterion": {
            "sleb": "SLEB min combined accept/score harm within tolerances",
            "random": "Random skip-sets; keep only three-win on train else []",
            "greedy": "Forward greedy maximize |skip| up to max_skip; ignore accept/metric",
            "accept_only": "Forward greedy maximize accept only; ignore metric/toks",
            "metric_only": "Forward greedy maximize task metric only; ignore accept/toks",
        }[method],
        "baseline": strip_hyp(baseline),
        "best": strip_hyp(current),
        "skip_layers": skip_layers,
        "history": history,
        "search_evaluations": n_evals,
        "wall_s": wall,
        "search_config": {
            "max_skip_layers": max_skip_layers,
            "early_barrier": early_barrier,
            "latter_barrier": latter_barrier,
            "layer_step": layer_step,
            "random_budget": random_budget if method == "random" else None,
        },
    }
    print(
        f"[{method}] SELECTED skip={skip_layers} "
        f"accept={_metric(current, 'mean_accepted_per_step'):.3f} "
        f"score={_metric(current, 'task_score'):.4f} "
        f"tok/s={_toks(current):.1f}; evals={n_evals} wall={wall/3600:.2f}h",
        flush=True,
    )
    return payload


def run_heldout(
    *,
    model: Any,
    items: list[DecodeItem],
    domain: str,
    skip_layers: list[int],
    train_result: dict[str, Any],
    baseline: dict[str, Any] | None = None,
    baseline_hyp: dict[str, str] | None = None,
) -> dict[str, Any]:
    from run_vicuna13_eagle3_skip_sweep import _collect_hypotheses, eval_skip_config

    if baseline is None:
        baseline = eval_skip_config(
            model, items, set(), baseline_hypotheses=None, domain=domain
        )
    if baseline_hyp is None:
        baseline_hyp = _collect_hypotheses(model, items, set())
    best = eval_skip_config(
        model, items, set(skip_layers), baseline_hypotheses=baseline_hyp, domain=domain
    )
    return {
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
        "search_evaluations": train_result.get("search_evaluations"),
    }


def load_tss_row(dataset: str) -> dict[str, Any]:
    meta = TSS_EXISTING[dataset]
    held = json.loads(Path(meta["heldout_path"]).read_text())
    best = held["test_eval"]["best"]
    return {
        "method": "tss_breadth",
        "display_name": METHOD_DISPLAY["tss_breadth"],
        "dataset": dataset,
        "skip_layers": list(meta["skip_layers"]),
        "accept_len": best["mean_accepted_per_step"],
        "metric": best["task_score"],
        "throughput": best["tok_per_s"],
        "search_evaluations": meta["search_evaluations"],
        "search_source": meta["search_source"],
        "heldout_source": str(meta["heldout_path"]),
    }


def build_table(rows: list[dict[str, Any]], out_dir: Path) -> None:
    # Order methods
    order = list(METHODS) + ["tss_breadth"]
    rows_sorted = sorted(
        rows,
        key=lambda r: (r["dataset"], order.index(r["method"]) if r["method"] in order else 99),
    )
    write_json({"rows": rows_sorted}, out_dir / "comparison_table.json")

    fields = [
        "Dataset",
        "Method",
        "Accept Len.",
        "Metric",
        "Skip Layers",
        "Throughput",
        "Search Evaluations",
    ]
    csv_path = out_dir / "comparison_table.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_sorted:
            w.writerow(
                {
                    "Dataset": r["dataset"],
                    "Method": r["display_name"],
                    "Accept Len.": f"{r['accept_len']:.3f}",
                    "Metric": f"{r['metric']:.4f}",
                    "Skip Layers": str(r["skip_layers"]),
                    "Throughput": f"{r['throughput']:.1f}",
                    "Search Evaluations": r["search_evaluations"],
                }
            )

    md_lines = [
        "# Vicuna-7B EAGLE search-method comparison",
        "",
        "Held-out test=64. Baselines searched on train=4 (fast); "
        "TSS Breadth Search reused from existing train=16 results.",
        "",
    ]
    for ds in sorted({r["dataset"] for r in rows_sorted}):
        md_lines.append(f"## {ds}")
        md_lines.append("")
        md_lines.append(
            "| Method | Accept Len. | Metric | Skip Layers | Throughput | Search Evaluations |"
        )
        md_lines.append("|---|---:|---:|---|---:|---:|")
        for r in rows_sorted:
            if r["dataset"] != ds:
                continue
            md_lines.append(
                f"| {r['display_name']} | {r['accept_len']:.3f} | {r['metric']:.4f} | "
                f"`{r['skip_layers']}` | {r['throughput']:.1f} | {r['search_evaluations']} |"
            )
        md_lines.append("")
    (out_dir / "comparison_table.md").write_text("\n".join(md_lines) + "\n")
    print(f"[table] wrote {csv_path}", flush=True)
    print("\n".join(md_lines), flush=True)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if args.table_only:
        rows: list[dict[str, Any]] = []
        for ds in datasets:
            for method in methods:
                held_path = out_dir / "heldout" / f"{ds}_{method}_heldout.json"
                if not held_path.exists():
                    print(f"[warn] missing {held_path}", flush=True)
                    continue
                held = json.loads(held_path.read_text())
                best = held["test_eval"]["best"]
                rows.append(
                    {
                        "method": method,
                        "display_name": METHOD_DISPLAY[method],
                        "dataset": ds,
                        "skip_layers": held["skip_layers"],
                        "accept_len": best["mean_accepted_per_step"],
                        "metric": best["task_score"],
                        "throughput": best["tok_per_s"],
                        "search_evaluations": held.get("search_evaluations"),
                    }
                )
            rows.append(load_tss_row(ds))
        build_table(rows, out_dir)
        return

    import torch
    from run_vicuna13_eagle3_skip_sweep import (
        MODEL_PRESETS,
        _resolve_vicuna7_base,
        load_model,
    )

    cfg = dict(MODEL_PRESETS["vicuna7"])
    cfg["base_model"] = _resolve_vicuna7_base()

    model = None
    rows: list[dict[str, Any]] = []

    from run_vicuna13_eagle3_skip_sweep import _collect_hypotheses, eval_skip_config

    try:
        for ds in datasets:
            domain = SCORE_CATEGORY[ds]
            # Keep the official 16/64 split so held-out matches TSS prompts;
            # search only uses the first --train-size examples of the train split.
            full_train = load_split(
                ds,
                "train",
                train_size=16,
                seed=args.seed,
                output_len=args.output_len,
            )
            train_items = full_train[: args.train_size]
            test_items = load_split(
                ds,
                "test",
                train_size=16,
                seed=args.seed,
                output_len=args.output_len,
            )
            rand_budget = args.random_budget or 48

            shared_baseline = None
            shared_hyp = None
            held_baseline = None
            held_hyp = None

            for method in methods:
                search_path = out_dir / "search" / f"{ds}_{method}_search.json"
                held_path = out_dir / "heldout" / f"{ds}_{method}_heldout.json"
                ensure_dir(search_path.parent)
                ensure_dir(held_path.parent)

                if search_path.exists() and not args.skip_search:
                    print(f"[resume] load search {search_path}", flush=True)
                    train_result = json.loads(search_path.read_text())
                elif args.skip_search:
                    raise FileNotFoundError(f"missing search result {search_path}")
                else:
                    if model is None:
                        print("[load] Vicuna-7B + EAGLE...", flush=True)
                        model = load_model(
                            base_model=cfg["base_model"],
                            ea_model=cfg["ea_model"],
                            total_token=60,
                            use_eagle3=False,
                        )
                    if shared_baseline is None:
                        print(
                            f"[{ds}] shared train baseline (n={len(train_items)})...",
                            flush=True,
                        )
                        shared_baseline = eval_skip_config(
                            model,
                            train_items,
                            set(),
                            baseline_hypotheses=None,
                            domain=domain,
                        )
                        shared_hyp = _collect_hypotheses(model, train_items, set())
                    print(
                        f"\n===== {ds} / {method} search "
                        f"(train={len(train_items)}, max_skip={args.max_skip_layers}, "
                        f"random_budget={rand_budget}) =====",
                        flush=True,
                    )
                    train_result = run_search_method(
                        method=method,
                        model=model,
                        items=train_items,
                        domain=domain,
                        num_layers=cfg["num_layers"],
                        max_skip_layers=args.max_skip_layers,
                        early_barrier=args.early_barrier,
                        latter_barrier=args.latter_barrier,
                        random_budget=rand_budget,
                        seed=args.seed + hash(ds + method) % 10000,
                        layer_step=args.layer_step,
                        baseline=shared_baseline,
                        baseline_hyp=shared_hyp,
                    )
                    train_result.update(
                        {
                            "dataset": ds,
                            "domain": domain,
                            "size": "7b",
                            "method_backend": "eagle",
                            "preset": "vicuna7",
                            "train_size": len(train_items),
                            "split": "train",
                            "protocol_note": (
                                f"fast search on first {args.train_size} of "
                                "official train-16; heldout uses test-64"
                            ),
                        }
                    )
                    write_json(train_result, search_path)

                if held_path.exists() and not args.skip_heldout:
                    print(f"[resume] load heldout {held_path}", flush=True)
                    held = json.loads(held_path.read_text())
                elif args.skip_heldout:
                    raise FileNotFoundError(f"missing heldout {held_path}")
                else:
                    if model is None:
                        print("[load] Vicuna-7B + EAGLE...", flush=True)
                        model = load_model(
                            base_model=cfg["base_model"],
                            ea_model=cfg["ea_model"],
                            total_token=60,
                            use_eagle3=False,
                        )
                    if held_baseline is None:
                        print(
                            f"[{ds}] shared heldout baseline (n={len(test_items)})...",
                            flush=True,
                        )
                        held_baseline = eval_skip_config(
                            model,
                            test_items,
                            set(),
                            baseline_hypotheses=None,
                            domain=domain,
                        )
                        held_hyp = _collect_hypotheses(model, test_items, set())
                    print(
                        f"===== {ds} / {method} heldout "
                        f"skip={train_result['skip_layers']} n={len(test_items)} =====",
                        flush=True,
                    )
                    held = run_heldout(
                        model=model,
                        items=test_items,
                        domain=domain,
                        skip_layers=list(train_result["skip_layers"]),
                        train_result=train_result,
                        baseline=held_baseline,
                        baseline_hyp=held_hyp,
                    )
                    held.update(
                        {
                            "dataset": ds,
                            "domain": domain,
                            "size": "7b",
                            "method": method,
                            "display_name": METHOD_DISPLAY[method],
                            "split": "test",
                            "test_size": len(test_items),
                        }
                    )
                    write_json(held, held_path)

                best = held["test_eval"]["best"]
                rows.append(
                    {
                        "method": method,
                        "display_name": METHOD_DISPLAY[method],
                        "dataset": ds,
                        "skip_layers": held["skip_layers"],
                        "accept_len": best["mean_accepted_per_step"],
                        "metric": best["task_score"],
                        "throughput": best["tok_per_s"],
                        "search_evaluations": held.get("search_evaluations")
                        or train_result.get("search_evaluations"),
                    }
                )

            rows.append(load_tss_row(ds))

        build_table(rows, out_dir)
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
