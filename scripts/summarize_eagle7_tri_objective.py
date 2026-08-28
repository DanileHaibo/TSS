#!/usr/bin/env python3
"""Summarize EAGLE-7B tri-objective held-out results and write a manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("translation", "summarization", "rag", "humaneval", "qa", "mmlu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO / "results" / "tss_tri_objective_v3_train16_20260714",
    )
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=REPO
        / "results"
        / "tss_pareto_bridge_v2_train16_20260713"
        / "7b_eagle",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(prefix: str, value: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_score": value.get("task_score"),
        f"{prefix}_accept": value.get("mean_accepted_per_step"),
        f"{prefix}_tok_per_s": value.get("tok_per_s"),
    }


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        path = args.root / "candidate_heldout" / dataset / "candidate_heldout.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        baseline = payload["baseline"]
        selected = payload["selected"]
        best = selected["best"]
        row = {
            "dataset": dataset,
            "skip_layers": selected["skip_layers"],
            "three_win": selected.get("three_win", False),
            "fallback": selected.get("fallback"),
            "tri_geomean_ratio": selected.get("tri_geomean_ratio"),
            "speedup": best["tok_per_s"] / baseline["tok_per_s"],
            **metrics("native", baseline),
            **metrics("final", best),
        }
        v2_path = args.v2_root / "heldout" / f"{dataset}_7b_eagle_heldout.json"
        if v2_path.exists():
            v2 = json.loads(v2_path.read_text())["test_eval"]["best"]
            row.update(metrics("v2", v2))
            row["v2_skip_layers"] = v2.get("skip_layers")
        rows.append(row)

    summary = {
        "schema_version": "eagle7_tri_objective_summary_v1",
        "protocol": {"train_size": 16, "test_size": 64, "seed": 42},
        "selection_rule": (
            "Require held-out score, mean accepted per step, and tok/s each "
            "to be >= native; otherwise use native."
        ),
        "complete": len(rows) == len(DATASETS),
        "rows": rows,
    }
    summary_path = args.root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = args.root / "summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    files = sorted(
        path
        for path in args.root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "_state" not in path.parts
        and not path.name.endswith(".log")
    )
    manifest = {
        "schema_version": "experiment_manifest_v1",
        "root": str(args.root.relative_to(REPO)),
        "files": [
            {
                "path": str(path.relative_to(REPO)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    (args.root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {summary_path}, {csv_path}, manifest ({len(files)} files)")
    for row in rows:
        print(
            f"{row['dataset']:14} skip={row['skip_layers']} "
            f"score={row['native_score']:.4f}->{row['final_score']:.4f} "
            f"accept={row['native_accept']:.3f}->{row['final_accept']:.3f} "
            f"speedup={row['speedup']:.2f}x"
        )


if __name__ == "__main__":
    main()
