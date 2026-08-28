#!/usr/bin/env python3
"""Build an EAGLE-7B fixed-skip candidate pool from prior runs."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("translation", "summarization", "rag", "humaneval", "qa", "mmlu")
KNOWN_SEEDS = {
    "translation": [[3, 11, 15, 25, 30], [8, 13, 25, 30], [14, 27, 28, 29, 30]],
    "summarization": [[3, 12], [5, 11, 25], [5, 15, 23]],
    "rag": [[3, 6, 14, 25, 30], [3, 15, 27, 28, 30], [2], [2, 12, 23, 29]],
    "humaneval": [[15, 20], [15, 27]],
    "qa": [[4, 9, 19, 28], [5, 10, 21, 31]],
    "mmlu": [[6, 7, 8, 14, 23]],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=REPO / "results")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results" / "tss_tri_objective_v3_train16_20260714" / "candidate_audit.json",
    )
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def metric(entry: dict[str, Any], key: str) -> float:
    try:
        return float(entry.get(key))
    except (TypeError, ValueError):
        return math.nan


def valid_layers(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        return None
    try:
        layers = tuple(sorted({int(x) for x in value}))
    except (TypeError, ValueError):
        return None
    if not layers or any(x < 2 or x >= 32 for x in layers) or len(layers) > 5:
        return None
    return layers


def train_rank(entry: dict[str, Any], baseline: dict[str, Any]) -> float:
    ratios = []
    for key in ("task_score", "mean_accepted_per_step", "tok_per_s"):
        value, base = metric(entry, key), metric(baseline, key)
        if not math.isfinite(value) or not math.isfinite(base) or base <= 0:
            return -math.inf
        ratios.append(max(value / base, 1e-9))
    return math.prod(ratios) ** (1.0 / 3.0)


def matching_search_files(root: Path, dataset: str) -> list[Path]:
    return sorted(
        path
        for path in root.glob("**/search/*.json")
        if path.name.startswith(f"{dataset}_") and "7b_eagle" in path.name
    )


def main() -> None:
    args = parse_args()
    pool: dict[str, dict[tuple[int, ...], dict[str, Any]]] = {
        dataset: {} for dataset in DATASETS
    }
    baselines: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_files: dict[str, list[str]] = defaultdict(list)

    for dataset in DATASETS:
        for path in matching_search_files(args.results_root, dataset):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            baseline = payload.get("baseline") or {}
            if not baseline:
                continue
            relative_path = str(path.relative_to(REPO))
            baselines[dataset].append(
                {
                    "source": relative_path,
                    "task_score": metric(baseline, "task_score"),
                    "mean_accepted_per_step": metric(baseline, "mean_accepted_per_step"),
                    "tok_per_s": metric(baseline, "tok_per_s"),
                }
            )
            source_files[dataset].append(relative_path)
            entries: list[tuple[str, dict[str, Any]]] = [
                ("history", row) for row in payload.get("history") or []
            ]
            for role, value in (payload.get("candidates") or {}).items():
                if role == "pareto_frontier" and isinstance(value, list):
                    entries.extend(("pareto_frontier", row) for row in value)
                elif isinstance(value, dict):
                    entries.append((role, value))
            if isinstance(payload.get("best"), dict):
                entries.append(("selected", payload["best"]))

            for role, entry in entries:
                layers = valid_layers(entry.get("skip_layers"))
                if layers is None:
                    continue
                rank = train_rank(entry, baseline)
                record = {
                    "skip_layers": list(layers),
                    "source": relative_path,
                    "source_role": role,
                    "train_score": metric(entry, "task_score"),
                    "train_accept": metric(entry, "mean_accepted_per_step"),
                    "train_tok_per_s": metric(entry, "tok_per_s"),
                    "train_tri_ratio": rank,
                    "train_three_win": all(
                        metric(entry, key) >= metric(baseline, key)
                        for key in ("task_score", "mean_accepted_per_step", "tok_per_s")
                    ),
                }
                previous = pool[dataset].get(layers)
                if previous is None or rank > previous["train_tri_ratio"]:
                    pool[dataset][layers] = record

        for seed in KNOWN_SEEDS[dataset]:
            layers = tuple(seed)
            pool[dataset].setdefault(
                layers,
                {
                    "skip_layers": seed,
                    "source": "curated_known_seed",
                    "source_role": "seed",
                    "train_score": None,
                    "train_accept": None,
                    "train_tok_per_s": None,
                    "train_tri_ratio": None,
                    "train_three_win": None,
                },
            )

    datasets_payload: dict[str, Any] = {}
    for dataset in DATASETS:
        ranked = sorted(
            pool[dataset].values(),
            key=lambda row: (
                bool(row["train_three_win"]),
                row["train_tri_ratio"] if isinstance(row["train_tri_ratio"], float) else -math.inf,
                len(row["skip_layers"]),
            ),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        curated = {tuple(x) for x in KNOWN_SEEDS[dataset]}
        selected.extend(row for row in ranked if tuple(row["skip_layers"]) in curated)
        selected.extend(row for row in ranked if row not in selected)
        selected = selected[: args.top_k]
        datasets_payload[dataset] = {
            "baselines": baselines[dataset],
            "source_files": source_files[dataset],
            "unique_candidates": len(pool[dataset]),
            "candidates": selected,
        }

    payload = {
        "schema_version": "eagle7_candidate_audit_v1",
        "protocol": {"train_size": 16, "test_size": 64, "seed": 42},
        "selection": "curated seeds plus train three-win candidates ranked by score/accept/tok_s geometric mean",
        "datasets": datasets_payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    for dataset, data in datasets_payload.items():
        print(
            f"{dataset:14} sources={len(data['source_files']):2d} "
            f"unique={data['unique_candidates']:4d} selected={len(data['candidates']):2d}"
        )


if __name__ == "__main__":
    main()
