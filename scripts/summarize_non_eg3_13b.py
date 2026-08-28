#!/usr/bin/env python3
"""Select and summarize the best non-EAGLE3 13B result per domain."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results" / "tss_tri_objective_13b_non_eg3_20260714"
DATASETS = ("translation", "summarization", "rag", "humaneval", "qa", "mmlu")
SOURCES = (
    ("hydra13", ROOT / "hydra13_candidate_heldout"),
    ("medusa13", ROOT / "medusa13_bold_heldout"),
)


def load_candidate(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not payload.get("complete"):
        return None
    return payload.get("selected")


def main() -> None:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        choices = []
        for architecture, base in SOURCES:
            selected = load_candidate(base / dataset / "candidate_heldout.json")
            if not selected or not selected.get("skip_layers"):
                continue
            choices.append((architecture, selected))
        if not choices:
            rows.append(
                {
                    "dataset": dataset,
                    "architecture": "none",
                    "skip_layers": [],
                    "status": "no_three_win",
                }
            )
            continue
        architecture, selected = max(
            choices, key=lambda item: item[1].get("tri_geomean_ratio", 1.0)
        )
        baseline, best = selected["baseline"], selected["best"]
        rows.append(
            {
                "dataset": dataset,
                "architecture": architecture,
                "skip_layers": selected["skip_layers"],
                "status": "three_win",
                "native_score": baseline["task_score"],
                "final_score": best["task_score"],
                "native_accept": baseline["mean_accepted_per_step"],
                "final_accept": best["mean_accepted_per_step"],
                "native_tok_per_s": baseline["tok_per_s"],
                "final_tok_per_s": best["tok_per_s"],
                "speedup": best["tok_per_s"] / baseline["tok_per_s"],
                "tri_geomean_ratio": selected.get("tri_geomean_ratio"),
            }
        )
    summary = {
        "schema_version": "non_eg3_13b_tri_objective_summary_v1",
        "protocol": {"train_size": 16, "test_size": 64, "seed": 42},
        "rows": rows,
    }
    (ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    fields = sorted({key for row in rows for key in row})
    with (ROOT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    files = sorted(
        path
        for path in ROOT.rglob("*.json")
        if path.name != "manifest.json" and "_state" not in path.parts
    )
    manifest = {
        "schema_version": "experiment_manifest_v1",
        "files": [
            {
                "path": str(path.relative_to(REPO)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    for row in rows:
        if row["status"] == "three_win":
            print(
                f"{row['dataset']:14} {row['architecture']:9} "
                f"skip={row['skip_layers']} score={row['native_score']:.4f}->"
                f"{row['final_score']:.4f} accept={row['native_accept']:.3f}->"
                f"{row['final_accept']:.3f} speedup={row['speedup']:.2f}x"
            )
        else:
            print(f"{row['dataset']:14} no three-win candidate")


if __name__ == "__main__":
    main()
