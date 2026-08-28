#!/usr/bin/env python3
"""Evaluate many fixed EAGLE-7B skip sets with one model load per dataset."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_tss_max_toks_pipeline import load_split, strip_hyp
from spec_exp.benchmark_config import SCORE_CATEGORY
from spec_exp.io import ensure_dir, write_json
from spec_exp.transformers_compat import install_transformers_compat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--candidate-audit",
        type=Path,
        default=REPO
        / "results"
        / "tss_tri_objective_v3_train16_20260714"
        / "candidate_audit.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO
        / "results"
        / "tss_tri_objective_v3_train16_20260714"
        / "candidate_heldout",
    )
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-len", type=int, default=96)
    return parser.parse_args()


def tri_ratio(best: dict[str, Any], baseline: dict[str, Any]) -> float:
    ratios = []
    for key in ("task_score", "mean_accepted_per_step", "tok_per_s"):
        value, base = float(best[key]), float(baseline[key])
        if not math.isfinite(value) or not math.isfinite(base) or base <= 0:
            return -math.inf
        ratios.append(max(value / base, 1e-12))
    return math.prod(ratios) ** (1.0 / 3.0)


def three_win(best: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return all(
        float(best[key]) >= float(baseline[key])
        for key in ("task_score", "mean_accepted_per_step", "tok_per_s")
    )


def main() -> None:
    args = parse_args()
    install_transformers_compat()
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf-cache")
    audit = json.loads(args.candidate_audit.read_text())
    dataset_payload = audit["datasets"][args.dataset]
    candidates = [
        sorted({int(x) for x in row["skip_layers"]})
        for row in dataset_payload["candidates"][: args.max_candidates]
    ]
    candidates = list(dict.fromkeys(tuple(row) for row in candidates))
    output_dir = ensure_dir(args.output_dir / args.dataset)
    aggregate_path = output_dir / "candidate_heldout.json"

    from scripts.run_vicuna13_eagle3_skip_sweep import (
        MODEL_PRESETS,
        _collect_hypotheses,
        _resolve_vicuna7_base,
        eval_skip_config,
        load_model,
    )

    items = load_split(
        args.dataset,
        "test",
        train_size=args.train_size,
        seed=args.seed,
        output_len=args.output_len,
    )
    cfg = dict(MODEL_PRESETS["vicuna7"])
    cfg["base_model"] = _resolve_vicuna7_base()
    model = None
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    try:
        model = load_model(
            base_model=cfg["base_model"],
            ea_model=cfg["ea_model"],
            total_token=60,
            use_eagle3=False,
        )
        domain = SCORE_CATEGORY[args.dataset]
        print(f"[{args.dataset}] native baseline on {len(items)} prompts", flush=True)
        baseline = eval_skip_config(
            model, items, set(), baseline_hypotheses=None, domain=domain
        )
        baseline_hyp = _collect_hypotheses(model, items, set())
        for index, layers in enumerate(candidates, 1):
            print(
                f"[{args.dataset}] candidate {index}/{len(candidates)} skip={list(layers)}",
                flush=True,
            )
            best = eval_skip_config(
                model,
                items,
                set(layers),
                baseline_hypotheses=baseline_hyp,
                domain=domain,
            )
            result = {
                "skip_layers": list(layers),
                "baseline": strip_hyp(baseline),
                "best": strip_hyp(best),
                "three_win": three_win(best, baseline),
                "tri_geomean_ratio": tri_ratio(best, baseline),
                "delta": {
                    "task_score": best["task_score"] - baseline["task_score"],
                    "accept": best["mean_accepted_per_step"]
                    - baseline["mean_accepted_per_step"],
                    "tok_per_s": best["tok_per_s"] - baseline["tok_per_s"],
                },
            }
            results.append(result)
            tag = "_".join(str(x) for x in layers)
            write_json(result, output_dir / f"skip_{tag}.json")
            write_json(
                {
                    "schema_version": "eagle7_candidate_heldout_v1",
                    "dataset": args.dataset,
                    "protocol": {
                        "train_size": args.train_size,
                        "test_size": len(items),
                        "seed": args.seed,
                        "output_len": args.output_len,
                    },
                    "baseline": strip_hyp(baseline),
                    "results": results,
                    "complete": False,
                },
                aggregate_path,
            )
            print(
                f"  score={best['task_score']:.4f} "
                f"accept={best['mean_accepted_per_step']:.3f} "
                f"tok/s={best['tok_per_s']:.1f} three_win={result['three_win']}",
                flush=True,
            )
    finally:
        if model is not None:
            del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    ranked = sorted(
        [row for row in results if row["three_win"]],
        key=lambda row: row["tri_geomean_ratio"],
        reverse=True,
    )
    selected = ranked[0] if ranked else {
        "skip_layers": [],
        "baseline": strip_hyp(baseline),
        "best": strip_hyp(baseline),
        "three_win": True,
        "tri_geomean_ratio": 1.0,
        "delta": {"task_score": 0.0, "accept": 0.0, "tok_per_s": 0.0},
        "fallback": "native",
    }
    payload = {
        "schema_version": "eagle7_candidate_heldout_v1",
        "dataset": args.dataset,
        "protocol": {
            "train_size": args.train_size,
            "test_size": len(items),
            "seed": args.seed,
            "output_len": args.output_len,
        },
        "baseline": strip_hyp(baseline),
        "results": results,
        "selected": selected,
        "complete": True,
        "wall_s": time.perf_counter() - t0,
    }
    write_json(payload, aggregate_path)
    print(
        f"[{args.dataset}] SELECTED skip={selected['skip_layers']} "
        f"score={selected['best']['task_score']:.4f} "
        f"accept={selected['best']['mean_accepted_per_step']:.3f} "
        f"tok/s={selected['best']['tok_per_s']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
