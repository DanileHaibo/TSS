#!/usr/bin/env python3
"""Re-run real skip-layer search under different score (metric) tolerances.

Constraint on task_score vs train baseline (relative):
  0%   → score >= baseline          (no drop)
  10%  → score >= baseline * 0.90   (allow 10% drop)
  20%  → score >= baseline * 0.80   (allow 20% drop)
  +5%  → score >= baseline * 1.05   (must improve 5%)

Accept stays hard: accept >= baseline at every expansion step.
Score tol is also enforced at every expansion step (enforce_score_tol_on_explore),
so 0/10/20/+5% change the beam path — not a post-hoc re-rank of one pool.
Uses real max_skip_latter_search from spec_exp.sleb_skip_search.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
    _metric,
    _toks,
    max_skip_latter_search,
    score_within_tol,
)

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DOMAINS = ("translation", "qa", "rag", "mmlu")
NUM_LAYERS = 32

# (label, score_drop_tol relative, min_score_ratio for final pick)
# drop_tol feeds score_within_tol: score >= baseline * (1 - tol)
#   0% / 10% / 20% → allow that relative drop during each select iteration
#   +5% → tol=-0.05 so every iteration requires score >= 1.05 * baseline
SCORE_TOLS = (
    ("0%", 0.00, 1.00),
    ("10%", 0.10, 0.90),
    ("20%", 0.20, 0.80),
    ("+5%", -0.05, 1.05),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DOMAINS))
    p.add_argument(
        "--output-dir",
        default=str(REPO / "results" / "eagle7_score_tol_search_20260730"),
    )
    p.add_argument("--train-size", type=int, default=8)
    p.add_argument("--output-len", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-skip-layers", type=int, default=5)
    p.add_argument("--beam-width", type=int, default=3)
    p.add_argument("--early-barrier", type=int, default=2)
    p.add_argument("--latter-barrier", type=int, default=0)
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
    dataset: str,
    split: str,
    *,
    protocol_train: int,
    search_train: int,
    seed: int,
    output_len: int,
):
    all_items = load_dataset_split(
        dataset, split="all", train_size=protocol_train, seed=seed, output_len=output_len
    )[:80]
    train_full = all_items[:protocol_train]
    test = all_items[protocol_train:]
    raw = train_full[:search_train] if split == "train" else test
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


def pick_final(
    history: list[dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    *,
    min_score_ratio: float,
) -> tuple[list[int], dict[str, Any]]:
    """Final pick among history: accept>=baseline, score>=baseline*min_score_ratio."""
    init_accept = _metric(baseline, config.accept_metric)
    init_score = _metric(baseline, config.score_key)
    best = dict(baseline)
    best["skip_layers"] = []
    best_key = (-math.inf, -1, -math.inf)

    for row in history:
        skip = list(row.get("skip_layers") or [])
        acc = _metric(row, config.accept_metric)
        score = _metric(row, config.score_key)
        if not (math.isfinite(acc) and math.isfinite(score)):
            continue
        if acc < init_accept - 1e-12:
            continue
        if score < init_score * min_score_ratio - 1e-12:
            continue
        # Also respect score_drop_tol if looser than min_ratio path
        if not score_within_tol(score, init_score, config.score_drop_tol, config.score_tol_mode):
            # For +5%, min_ratio is stricter; for drops, drop_tol matches min_ratio.
            if min_score_ratio >= 1.0:
                pass  # already checked min_ratio
            else:
                continue
        key = (acc, len(skip), _toks(row))
        if key > best_key:
            best_key = key
            best = dict(row)
            best["skip_layers"] = skip

    # +5% may find nothing → return []
    if min_score_ratio > 1.0 and not best.get("skip_layers"):
        if _metric(best, config.score_key) < init_score * min_score_ratio - 1e-12:
            return [], dict(baseline) | {"skip_layers": []}
    return list(best.get("skip_layers") or []), best


def run_one_search(
    *,
    eval_fn: Callable[[set[int]], dict[str, Any]],
    baseline: dict[str, Any],
    label: str,
    score_drop_tol: float,
    min_score_ratio: float,
    max_skip: int,
    beam_width: int,
    early_barrier: int,
    latter_barrier: int,
) -> dict[str, Any]:
    cfg = SlebSearchConfig(
        num_layers=NUM_LAYERS,
        max_skip_layers=max_skip,
        early_barrier=early_barrier,
        latter_barrier=latter_barrier,
        accept_drop_tol=0.0,  # hard accept >= baseline during explore
        score_drop_tol=score_drop_tol,
        score_tol_mode="relative",
        accept_metric="mean_accepted_per_step",
        score_key="task_score",
    )
    print(
        f"  [search {label}] max_skip_latter "
        f"score_drop_tol={score_drop_tol:.2f} min_score_ratio={min_score_ratio:.2f}",
        flush=True,
    )
    t0 = time.time()
    eval_count = {"n": 0}

    def counted_eval(skip: set[int]) -> dict[str, Any]:
        eval_count["n"] += 1
        return eval_fn(skip)

    skip_set, history, current = max_skip_latter_search(
        eval_fn=counted_eval,
        baseline=baseline,
        config=cfg,
        beam_width=beam_width,
        min_skips=min(3, max_skip),
        score_first=False,
        # Apply score_drop_tol at every expansion step so 0/10/20/+5%
        # change which layers are kept in the beam (not only final re-pick).
        enforce_score_tol_on_explore=True,
        on_trial_error=lambda layer, exc: print(f"  [err]+{layer}: {exc}", flush=True),
    )
    # Re-pick final with explicit min_score_ratio (handles +5%).
    skip, best = pick_final(
        history, baseline, cfg, min_score_ratio=min_score_ratio
    )
    # If pick_final found something different from algorithm's current, prefer pick_final.
    if not skip and skip_set and min_score_ratio <= 1.0:
        # fall back to algorithm output if it satisfies tol
        if score_within_tol(
            _metric(current, cfg.score_key),
            _metric(baseline, cfg.score_key),
            score_drop_tol,
            "relative",
        ) and _metric(current, cfg.accept_metric) >= _metric(baseline, cfg.accept_metric) - 1e-12:
            skip = sorted(skip_set)
            best = current

    print(
        f"  [search {label}] SELECTED skip={skip} "
        f"train_acc={_metric(best, cfg.accept_metric):.3f} "
        f"score={_metric(best, cfg.score_key):.4f} "
        f"evals={eval_count['n']} wall={ (time.time()-t0)/60:.1f}m",
        flush=True,
    )
    return {
        "tol_label": label,
        "score_drop_tol": score_drop_tol,
        "min_score_ratio": min_score_ratio,
        "skip_layers": skip,
        "search_evaluations": eval_count["n"],
        "wall_s": time.time() - t0,
        "baseline": strip_hyp(baseline),
        "best_train": strip_hyp(best) if isinstance(best, dict) else best,
        "history_len": len(history),
        "mode": "max_skip_latter_search",
    }


def build_table(rows: list[dict[str, Any]], out_dir: Path) -> None:
    write_json({"rows": rows}, out_dir / "sensitivity_table.json")
    fields = [
        "Domain",
        "Score Tol",
        "Accept Len.",
        "Metric",
        "Skip Layers",
        "Sparsity",
        "Throughput",
        "Search Evaluations",
    ]
    with (out_dir / "sensitivity_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "Domain": r["dataset"],
                    "Score Tol": r["tol_label"],
                    "Accept Len.": f"{r['accept_len']:.3f}",
                    "Metric": f"{r['metric']:.4f}",
                    "Skip Layers": str(r["skip_layers"]),
                    "Sparsity": f"{100 * r['sparsity']:.1f}%",
                    "Throughput": f"{r['throughput']:.1f}",
                    "Search Evaluations": r["search_evaluations"],
                }
            )
    md = [
        "# Vicuna-7B EAGLE: score-tolerance sensitivity (real max_skip_latter search)",
        "",
        "Each row re-runs skip-layer search. Score constraint vs train baseline:",
        "`0%` ≥ baseline; `10%` ≥ 0.9×; `20%` ≥ 0.8×; `+5%` ≥ 1.05×.",
        "Accept kept hard ≥ baseline during search (TSS max_skip_latter).",
        "",
        "| Domain | Score Tol | Accept | Metric | Skip | Sparsity | Tok/s | Evals |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['dataset']} | {r['tol_label']} | {r['accept_len']:.3f} | "
            f"{r['metric']:.4f} | `{r['skip_layers']}` | {100*r['sparsity']:.1f}% | "
            f"{r['throughput']:.1f} | {r['search_evaluations']} |"
        )
    text = "\n".join(md) + "\n"
    (out_dir / "sensitivity_table.md").write_text(text)
    pf = Path("/root/autodl-tmp/experiments-data/paper_figure")
    pf.mkdir(parents=True, exist_ok=True)
    (pf / "tab_eagle7_score_tol_sensitivity.md").write_text(text)
    (pf / "tab_eagle7_score_tol_sensitivity.csv").write_text(
        (out_dir / "sensitivity_table.csv").read_text()
    )
    print(text, flush=True)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    if args.table_only:
        build_table(json.loads((out_dir / "sensitivity_table.json").read_text())["rows"], out_dir)
        return

    import torch
    from run_vicuna13_eagle3_skip_sweep import (
        MODEL_PRESETS,
        _collect_hypotheses,
        _resolve_vicuna7_base,
        eval_skip_config,
        load_model,
    )

    cfg = dict(MODEL_PRESETS["vicuna7"])
    cfg["base_model"] = _resolve_vicuna7_base()
    print("[load] Vicuna-7B + EAGLE...", flush=True)
    model = load_model(
        base_model=cfg["base_model"],
        ea_model=cfg["ea_model"],
        total_token=60,
        use_eagle3=False,
    )

    rows: list[dict[str, Any]] = []
    try:
        for ds in datasets:
            domain = SCORE_CATEGORY[ds]
            train_items = load_split(
                ds,
                "train",
                protocol_train=16,
                search_train=args.train_size,
                seed=args.seed,
                output_len=args.output_len,
            )
            test_items = load_split(
                ds,
                "test",
                protocol_train=16,
                search_train=args.train_size,
                seed=args.seed,
                output_len=args.output_len,
            )
            print(
                f"\n########## {ds} search_train={len(train_items)} "
                f"heldout={len(test_items)} ##########",
                flush=True,
            )
            print(f"[{ds}] train baseline...", flush=True)
            baseline = strip_hyp(
                eval_skip_config(model, train_items, set(), baseline_hypotheses=None, domain=domain)
            )
            hyp = _collect_hypotheses(model, train_items, set())
            cache: dict[frozenset[int], dict[str, Any]] = {frozenset(): baseline}

            def _eval(skip: set[int]) -> dict[str, Any]:
                key = frozenset(skip)
                if key in cache:
                    return cache[key]
                m = strip_hyp(
                    eval_skip_config(
                        model, train_items, skip, baseline_hypotheses=hyp, domain=domain
                    )
                )
                cache[key] = m
                torch.cuda.empty_cache()
                return m

            print(f"[{ds}] heldout baseline...", flush=True)
            held_base = strip_hyp(
                eval_skip_config(model, test_items, set(), baseline_hypotheses=None, domain=domain)
            )
            held_hyp = _collect_hypotheses(model, test_items, set())
            held_cache: dict[tuple[int, ...], dict[str, Any]] = {(): held_base}

            for label, drop_tol, min_ratio in SCORE_TOLS:
                tag = label.replace("%", "pct").replace("+", "p")
                search_path = out_dir / "search" / f"{ds}_{tag}_search.json"
                ensure_dir(search_path.parent)

                if search_path.exists():
                    print(f"[resume] {search_path}", flush=True)
                    payload = json.loads(search_path.read_text())
                else:
                    print(f"\n===== {ds} / score_tol={label} SEARCH =====", flush=True)
                    before = len(cache)
                    payload = run_one_search(
                        eval_fn=_eval,
                        baseline=baseline,
                        label=label,
                        score_drop_tol=drop_tol,
                        min_score_ratio=min_ratio,
                        max_skip=args.max_skip_layers,
                        beam_width=args.beam_width,
                        early_barrier=args.early_barrier,
                        latter_barrier=args.latter_barrier,
                    )
                    # count new unique evals approx
                    payload["cache_size_after"] = len(cache)
                    payload["new_unique_evals"] = len(cache) - before
                    payload["dataset"] = ds
                    write_json(payload, search_path)

                skip = list(payload["skip_layers"])
                key = tuple(skip)
                held_path = (
                    out_dir / "heldout" / f"{ds}_skip_{'_'.join(map(str,key)) or 'none'}.json"
                )
                ensure_dir(held_path.parent)
                if key in held_cache:
                    best_h = held_cache[key]
                elif held_path.exists():
                    best_h = json.loads(held_path.read_text())["best"]
                    held_cache[key] = best_h
                else:
                    print(f"[{ds} {label}] heldout skip={skip}", flush=True)
                    best_h = strip_hyp(
                        eval_skip_config(
                            model,
                            test_items,
                            set(skip),
                            baseline_hypotheses=held_hyp,
                            domain=domain,
                        )
                    )
                    best_h["skip_layers"] = skip
                    write_json({"baseline": held_base, "best": best_h}, held_path)
                    held_cache[key] = best_h

                rows.append(
                    {
                        "dataset": ds,
                        "tol_label": label,
                        "score_drop_tol": payload.get("score_drop_tol", drop_tol),
                        "min_score_ratio": payload.get("min_score_ratio", min_ratio),
                        "skip_layers": skip,
                        "sparsity": len(skip) / NUM_LAYERS,
                        "accept_len": float(best_h["mean_accepted_per_step"]),
                        "metric": float(best_h["task_score"]),
                        "throughput": float(best_h.get("tok_per_s") or 0),
                        "search_evaluations": payload.get("search_evaluations", 0),
                    }
                )

        build_table(rows, out_dir)
        print(f"[done] {out_dir}", flush=True)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
