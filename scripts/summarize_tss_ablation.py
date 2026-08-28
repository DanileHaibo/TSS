#!/usr/bin/env python3
"""Summarize fixed-skip layer ablations into paper-ready artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEFAULT_ROOT = REPO / "results" / "tss_ablation_20260716"
RELEASE = (
    REPO
    / "results"
    / "tss_triple_win_final_20260716"
    / "triple_win_summary.json"
)

from scripts.run_tss_ablation import ablation_variants


def ratios(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float]:
    values = {
        "score": float(candidate["task_score"]) / float(baseline["task_score"]),
        "accept": float(candidate["mean_accepted_per_step"])
        / float(baseline["mean_accepted_per_step"]),
        "throughput": float(candidate["tok_per_s"])
        / float(baseline["tok_per_s"]),
    }
    values["tri_mean"] = math.prod(values.values()) ** (1.0 / 3.0)
    return values


def three_win(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> bool:
    return all(
        float(candidate[key]) >= float(baseline[key])
        for key in ("task_score", "mean_accepted_per_step", "tok_per_s")
    )


def load_domain_output(
    root: Path, release_row: dict[str, Any]
) -> tuple[
    Path,
    dict[str, Any],
    dict[tuple[int, ...], dict[str, Any]],
]:
    size = release_row["size"].lower()
    dataset = release_row["dataset"]
    path = root / "layer" / size / dataset / "candidate_heldout.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not payload.get("complete"):
        raise RuntimeError(f"incomplete ablation output: {path}")
    baseline = payload["baseline"]
    candidate_key = "best" if release_row["size"] == "7B" else "candidate"
    results = {
        tuple(row["skip_layers"]): {
            **row,
            "candidate_metrics": row[candidate_key],
        }
        for row in payload["results"]
    }
    return path, baseline, results


def collect(root: Path) -> list[dict[str, Any]]:
    release = json.loads(RELEASE.read_text())
    rows: list[dict[str, Any]] = []
    for published in release["rows"]:
        source, baseline, results = load_domain_output(root, published)
        full_skip = list(published["skip_layers"])
        for variant, layers in ablation_variants(full_skip):
            key = tuple(layers)
            if key not in results:
                raise KeyError(
                    f"{published['size']}/{published['dataset']} missing {key}"
                )
            candidate = results[key]["candidate_metrics"]
            metric_ratios = ratios(candidate, baseline)
            rows.append(
                {
                    "size": published["size"],
                    "model": published["model"],
                    "architecture": published["architecture"],
                    "dataset": published["dataset"],
                    "variant": variant,
                    "skip_layers": layers,
                    "num_skip_layers": len(layers),
                    "sparsity": len(layers) / int(published["num_layers"]),
                    "score": float(candidate["task_score"]),
                    "accept_len": float(
                        candidate["mean_accepted_per_step"]
                    ),
                    "tok_per_s": float(candidate["tok_per_s"]),
                    "native_score": float(baseline["task_score"]),
                    "native_accept_len": float(
                        baseline["mean_accepted_per_step"]
                    ),
                    "native_tok_per_s": float(baseline["tok_per_s"]),
                    "score_ratio": metric_ratios["score"],
                    "accept_ratio": metric_ratios["accept"],
                    "throughput_ratio": metric_ratios["throughput"],
                    "tri_mean_ratio": metric_ratios["tri_mean"],
                    "three_win": three_win(candidate, baseline),
                    "source": str(source.relative_to(REPO)),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def latex_summary(rows: list[dict[str, Any]]) -> str:
    names = {
        "translation": "Translation",
        "summarization": "Summarization",
        "rag": "RAG",
        "qa": "QA",
        "mmlu": "MMLU",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Layer ablation of final TSS configurations on held-out "
            r"prompts. Critical Layer is the layer whose removal causes the "
            r"largest Tri-Mean decrease. Full and LOO values are ratios to Native.}"
        ),
        r"\label{tab:tss_layer_ablation}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        (
            r"Size & Domain & $|\mathcal{S}|$ & Full Tri-Mean & "
            r"LOO Mean & LOO Min & Critical Layer & Critical $\Delta$ \\"
        ),
        r"\midrule",
    ]
    domain_keys = list(
        dict.fromkeys((row["size"], row["dataset"]) for row in rows)
    )
    for size, dataset in domain_keys:
        group = [
            row
            for row in rows
            if row["size"] == size and row["dataset"] == dataset
        ]
        full = next(row for row in group if row["variant"] == "full")
        loo = [row for row in group if row["variant"].startswith("drop_")]
        critical = min(loo, key=lambda row: row["tri_mean_ratio"])
        drop = full["tri_mean_ratio"] - critical["tri_mean_ratio"]
        lines.append(
            f"{size} & {names[dataset]} & {full['num_skip_layers']} & "
            f"{full['tri_mean_ratio']:.3f} & "
            f"{sum(row['tri_mean_ratio'] for row in loo) / len(loo):.3f} & "
            f"{critical['tri_mean_ratio']:.3f} & "
            f"{critical['variant'].removeprefix('drop_')} & {drop:+.3f} \\\\"
        )
    lines.extend(
        [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    rows = collect(args.root)
    output = args.root / "release"
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "tss_layer_ablation_v1",
        "protocol": {
            "heldout": True,
            "seed": 42,
            "variants": "full + leave-one-out + front/back subsets",
        },
        "row_count": len(rows),
        "rows": rows,
    }
    (output / "layer_ablation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_csv(output / "layer_ablation.csv", rows)
    (output / "layer_ablation.tex").write_text(latex_summary(rows))
    readme = (
        "# TSS layer ablation\n\n"
        "This release compares each final TSS configuration against all "
        "leave-one-out variants and balanced front/back subsets on held-out "
        "prompts. The canonical detailed artifact is `layer_ablation.json`.\n"
    )
    (output / "README.md").write_text(readme)
    source_files = {
        REPO / row["source"]
        for row in rows
    }
    files = sorted(
        {
            path
            for path in output.iterdir()
            if path.name != "manifest.json"
        }
        | source_files
    )
    manifest = {
        "schema_version": "tss_ablation_manifest_v1",
        "files": [
            {
                "path": str(path.relative_to(REPO)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(rows)} ablation rows to {output}")


if __name__ == "__main__":
    main()
