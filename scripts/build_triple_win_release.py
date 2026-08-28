#!/usr/bin/env python3
"""Build the final 7B/13B held-out triple-win release artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "tss_triple_win_final_20260716"
ROOT_7B = REPO / "results" / "tss_tri_objective_v3_train16_20260714"
ROOT_13B = REPO / "results" / "tss_tri_objective_13b_non_eg3_20260714"

SOURCES_7B = {
    dataset: ROOT_7B
    / "final_validation"
    / f"{dataset}_7b_eagle_heldout.json"
    for dataset in ("translation", "summarization", "rag", "qa", "mmlu")
}
SOURCES_13B = {
    dataset: ROOT_13B
    / "samd_score_first_final_validation"
    / f"{dataset}_heldout.json"
    for dataset in ("translation", "summarization", "qa", "mmlu")
}
METRICS = {
    "translation": "BLEU",
    "summarization": "ROUGE-L",
    "rag": "F1",
    "qa": "F1",
    "mmlu": "Accuracy",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def strict_three_win(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return all(
        float(candidate[key]) >= float(baseline[key])
        for key in ("task_score", "mean_accepted_per_step", "tok_per_s")
    )


def make_row(
    *,
    size: str,
    dataset: str,
    architecture: str,
    model: str,
    num_layers: int,
    selection_train_size: int,
    skip_layers: list[int],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    if len(skip_layers) < 1:
        raise ValueError(f"{size}/{dataset}: empty skip set")
    if not strict_three_win(baseline, candidate):
        raise ValueError(f"{size}/{dataset}: source is not a strict three-win")
    ratios = {
        "score": float(candidate["task_score"]) / float(baseline["task_score"]),
        "accept": float(candidate["mean_accepted_per_step"])
        / float(baseline["mean_accepted_per_step"]),
        "throughput": float(candidate["tok_per_s"])
        / float(baseline["tok_per_s"]),
    }
    return {
        "size": size,
        "dataset": dataset,
        "architecture": architecture,
        "model": model,
        "num_layers": num_layers,
        "selection_train_size": selection_train_size,
        "heldout_size": 64,
        "seed": 42,
        "score_metric": METRICS[dataset],
        "skip_layers": skip_layers,
        "num_skip_layers": len(skip_layers),
        "sparsity": len(skip_layers) / num_layers,
        "native": {
            "score": float(baseline["task_score"]),
            "accept_len": float(baseline["mean_accepted_per_step"]),
            "tok_per_s": float(baseline["tok_per_s"]),
        },
        "tss": {
            "score": float(candidate["task_score"]),
            "accept_len": float(candidate["mean_accepted_per_step"]),
            "tok_per_s": float(candidate["tok_per_s"]),
        },
        "ratios": ratios,
        "tri_mean_ratio": math.prod(ratios.values()) ** (1.0 / 3.0),
        "three_win": True,
        "source": str(source.relative_to(REPO)),
    }


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, source in SOURCES_7B.items():
        payload = load_json(source)
        test_eval = payload["test_eval"]
        rows.append(
            make_row(
                size="7B",
                dataset=dataset,
                architecture="EAGLE",
                model="Vicuna-7B",
                num_layers=32,
                selection_train_size=16,
                skip_layers=list(payload["skip_layers"]),
                baseline=test_eval["baseline"],
                candidate=test_eval["best"],
                source=source,
            )
        )
    for dataset, source in SOURCES_13B.items():
        payload = load_json(source)
        if not payload.get("three_win"):
            raise ValueError(f"{source}: source flag is not three_win")
        rows.append(
            make_row(
                size="13B",
                dataset=dataset,
                architecture="SAMD Token-Recycle",
                model="Llama-2-13B",
                num_layers=40,
                selection_train_size=32 if dataset == "qa" else 16,
                skip_layers=list(payload["skip_layers"]),
                baseline=payload["baseline"],
                candidate=payload["candidate"],
                source=source,
            )
        )
    return rows


def latex_skip(layers: list[int]) -> str:
    return "$\\{" + ",".join(str(layer) for layer in layers) + "\\}$"


def latex_rows(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Held-out triple-win results on Spec-Bench (64 prompts). "
            r"Skip configurations are selected on 16 training prompts, except "
            r"13B QA, which uses 32 prompts. Native denotes speculative decoding "
            r"without target-layer skipping; +TSS uses score-first multi-layer "
            r"target skipping. Throughput is end-to-end on the same device. "
            r"Tri-Mean is the geometric mean of task-score, acceptance-length, "
            r"and throughput ratios relative to Native.}"
        ),
        r"\label{tab:main_tss_triple_win}",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        (
            r"Domain & Method & Accept Len. & Task Metric & Skip Layers & "
            r"Sparsity & Throughput & vs.\ Native & Tri-Mean \\"
        ),
        r"& & & & & & (tok/s) & & \\",
        r"\midrule",
    ]
    sections = (
        ("7B", "Vicuna-7B", "EAGLE", 32),
        ("13B", "Llama-2-13B", "SAMD", 40),
    )
    names = {
        "translation": "Translation",
        "summarization": "Summarization",
        "rag": "RAG",
        "qa": "QA",
        "mmlu": "MMLU",
    }
    for section_index, (size, model, method, layers) in enumerate(sections):
        if section_index:
            lines.append(r"\midrule")
        architecture = (
            "EAGLE / EAGLE+TSS"
            if size == "7B"
            else "SAMD Token-Recycle / SAMD+TSS"
        )
        lines.extend(
            [
                (
                    rf"\multicolumn{{9}}{{l}}{{\textbf{{{model}}} "
                    rf"({architecture}, $L={layers}$)}} \\"
                ),
                r"\midrule",
            ]
        )
        for row in (item for item in rows if item["size"] == size):
            native = row["native"]
            tss = row["tss"]
            lines.extend(
                [
                    (
                        f"{names[row['dataset']]} & {method} & "
                        f"{native['accept_len']:.2f} & {native['score']:.3f} & "
                        r"-- & -- & "
                        f"{native['tok_per_s']:.1f} & "
                        r"1.00$\times$ & 1.000$\times$ \\"
                    ),
                    (
                        f"& {method}+TSS & "
                        rf"\textbf{{{tss['accept_len']:.2f}}} & "
                        rf"\textbf{{{tss['score']:.3f}}} & "
                        f"{latex_skip(row['skip_layers'])} & "
                        f"{100.0 * row['sparsity']:.1f}\\% & "
                        rf"\textbf{{{tss['tok_per_s']:.1f}}} & "
                        rf"\textbf{{{row['ratios']['throughput']:.2f}$\times$}} & "
                        rf"\textbf{{{row['tri_mean_ratio']:.3f}$\times$}} \\"
                    ),
                    "",
                ]
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = (
        "size",
        "dataset",
        "architecture",
        "model",
        "selection_train_size",
        "skip_layers",
        "sparsity",
        "native_score",
        "tss_score",
        "native_accept_len",
        "tss_accept_len",
        "native_tok_per_s",
        "tss_tok_per_s",
        "throughput_ratio",
        "tri_mean_ratio",
        "source",
    )
    with (OUT / "triple_win_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "size": row["size"],
                    "dataset": row["dataset"],
                    "architecture": row["architecture"],
                    "model": row["model"],
                    "selection_train_size": row["selection_train_size"],
                    "skip_layers": ",".join(map(str, row["skip_layers"])),
                    "sparsity": row["sparsity"],
                    "native_score": row["native"]["score"],
                    "tss_score": row["tss"]["score"],
                    "native_accept_len": row["native"]["accept_len"],
                    "tss_accept_len": row["tss"]["accept_len"],
                    "native_tok_per_s": row["native"]["tok_per_s"],
                    "tss_tok_per_s": row["tss"]["tok_per_s"],
                    "throughput_ratio": row["ratios"]["throughput"],
                    "tri_mean_ratio": row["tri_mean_ratio"],
                    "source": row["source"],
                }
            )


def main() -> None:
    rows = collect_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "tss_triple_win_release_v1",
        "created_from_existing_results": True,
        "protocol": {
            "heldout_size": 64,
            "seed": 42,
            "selection_train_size": 16,
            "exceptions": {"13B/qa": {"selection_train_size": 32}},
            "strict_three_win": (
                "candidate score, accept_len, and tok_per_s are each >= native"
            ),
        },
        "coverage": {"7B": 5, "13B": 4},
        "excluded": {
            "7B": ["humaneval"],
            "13B": ["rag", "humaneval"],
        },
        "rows": rows,
    }
    (OUT / "triple_win_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_csv(rows)
    (OUT / "main_table.tex").write_text(latex_rows(rows))
    readme = (
        "# Final TSS triple-win release\n\n"
        "- 7B: Vicuna-7B + EAGLE, 5 strict held-out triple-win domains.\n"
        "- 13B: Llama-2-13B + SAMD Token-Recycle, 4 strict held-out "
        "triple-win domains.\n"
        "- Final 13B configurations all skip at least 6 of 40 layers.\n"
        "- Excluded from the main table: 7B HumanEval; 13B RAG and HumanEval.\n"
        "- `main_table.tex` replaces latency decomposition with end-to-end "
        "throughput, throughput ratio vs. Native, and Tri-Mean.\n"
        "- `triple_win_summary.json` is the canonical machine-readable result.\n"
    )
    (OUT / "README.md").write_text(readme)

    tracked = sorted(
        list(SOURCES_7B.values())
        + list(SOURCES_13B.values())
        + [
            OUT / "README.md",
            OUT / "main_table.tex",
            OUT / "triple_win_summary.csv",
            OUT / "triple_win_summary.json",
        ]
    )
    manifest = {
        "schema_version": "tss_triple_win_manifest_v1",
        "files": [
            {
                "path": str(path.relative_to(REPO)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in tracked
        ],
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    for row in rows:
        print(
            f"{row['size']:3} {row['dataset']:14} "
            f"skip={row['skip_layers']} "
            f"score={row['native']['score']:.4f}->{row['tss']['score']:.4f} "
            f"accept={row['native']['accept_len']:.3f}->"
            f"{row['tss']['accept_len']:.3f} "
            f"throughput={row['ratios']['throughput']:.2f}x "
            f"tri={row['tri_mean_ratio']:.3f}x"
        )


if __name__ == "__main__":
    main()
