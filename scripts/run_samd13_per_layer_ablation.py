#!/usr/bin/env python3
"""Llama-2-13B + SAMD[Token-Recycle] single-layer ablation on four main domains."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_samd_target_skip_search import (  # noqa: E402
    TARGETS,
    extract_user,
    render_prompt,
)
from spec_exp.benchmark_config import PRIMARY_METRIC, SCORE_CATEGORY  # noqa: E402
from spec_exp.benchmark_datasets import load_dataset_items  # noqa: E402
from spec_exp.self_spec_decode import DecodeItem  # noqa: E402
from spec_exp.transformers_compat import install_transformers_compat  # noqa: E402

DOMAINS = ("translation", "summarization", "qa", "mmlu")
DOMAIN_LABELS = {
    "translation": "Translation",
    "summarization": "Summarization",
    "qa": "QA",
    "mmlu": "MMLU",
}
NUM_LAYERS = 40
TOKEN_TREE = str(
    REPO / "data" / "Spec-Bench-repo" / "model" / "samd" / "config" / "token_recycle_4_15.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def load_items(
    dataset: str,
    *,
    num_requests: int,
    seed: int,
    output_len: int,
    target: str,
    tokenizer: Any,
) -> list[DecodeItem]:
    raw = load_dataset_items(
        dataset,
        num_requests=num_requests,
        seed=seed,
        output_len=output_len,
    )
    return [
        DecodeItem(
            request_id=item.request_id,
            prompt=render_prompt(tokenizer, extract_user(item.prompt), target),
            max_tokens=item.max_tokens,
            category=item.category,
            reference=item.reference,
        )
        for item in raw
    ]


def run_domain(
    model: Any,
    dataset: str,
    items: list[DecodeItem],
    eval_samd: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    domain = SCORE_CATEGORY[dataset]
    print(f"\n===== {dataset} n={len(items)} layers={NUM_LAYERS} =====", flush=True)
    baseline = eval_samd(
        model,
        items,
        set(),
        baseline_hypotheses=None,
        domain=domain,
    )
    rows = []
    for layer in range(NUM_LAYERS):
        started = time.perf_counter()
        try:
            metrics = eval_samd(
                model,
                items,
                {layer},
                baseline_hypotheses=None,
                domain=domain,
            )
            row = {
                "dataset": dataset,
                "metric_name": PRIMARY_METRIC[dataset],
                "layer": layer,
                "mean_accepted_per_step": metrics["mean_accepted_per_step"],
                "task_score": metrics["task_score"],
                "delta_accept": metrics["mean_accepted_per_step"]
                - baseline["mean_accepted_per_step"],
                "delta_metric": metrics["task_score"] - baseline["task_score"],
                "wall_s": metrics["wall_s"],
                "error": "",
            }
        except Exception as exc:
            row = {
                "dataset": dataset,
                "metric_name": PRIMARY_METRIC[dataset],
                "layer": layer,
                "mean_accepted_per_step": math.nan,
                "task_score": math.nan,
                "delta_accept": math.nan,
                "delta_metric": math.nan,
                "wall_s": math.nan,
                "error": str(exc),
            }
            torch.cuda.empty_cache()
        rows.append(row)
        print(
            f"  [L{layer:02d}] Δacc={row['delta_accept']:+.3f} "
            f"Δmetric={row['delta_metric']:+.4f} "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    summary = {
        "dataset": dataset,
        "metric_name": PRIMARY_METRIC[dataset],
        "num_requests": len(items),
        "num_layers": NUM_LAYERS,
        "baseline": {
            "mean_accepted_per_step": baseline["mean_accepted_per_step"],
            "task_score": baseline["task_score"],
        },
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset in DOMAINS:
        path = output_dir / f"per_layer_{dataset}.csv"
        if not path.exists():
            continue
        with path.open() as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, Any] = dict(raw)
                for key in (
                    "layer",
                    "mean_accepted_per_step",
                    "task_score",
                    "delta_accept",
                    "delta_metric",
                    "wall_s",
                ):
                    if row.get(key):
                        row[key] = (
                            int(row[key])
                            if key == "layer"
                            else float(row[key])
                        )
                rows.append(row)
    return rows


def matrix(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = np.full((len(DOMAINS), NUM_LAYERS), np.nan)
    for domain_index, dataset in enumerate(DOMAINS):
        for row in rows:
            if row["dataset"] == dataset:
                values[domain_index, int(row["layer"])] = float(row[key])
    return values


def plot_heatmap(rows: list[dict[str, Any]], output: Path) -> None:
    from matplotlib import font_manager

    for font_path in Path("/root/.fonts/arial_alias").glob("*.ttf"):
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    panels = (
        ("delta_accept", r"$\Delta$ Accept Length  (skip L $-$ baseline)"),
        ("delta_metric", r"$\Delta$ Domain Metric  (skip L $-$ baseline)"),
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.5, 6.4),
        sharex=True,
        gridspec_kw={"hspace": 0.38},
    )
    for ax, (key, title) in zip(axes, panels):
        values = matrix(rows, key)
        vmax = max(float(np.nanmax(np.abs(values))), 1e-6)
        image = ax.imshow(
            values,
            aspect="auto",
            cmap="RdBu",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_yticks(range(len(DOMAINS)))
        ax.set_yticklabels([DOMAIN_LABELS[dataset] for dataset in DOMAINS])
        ax.set_title(title, loc="left", fontweight="bold", fontsize=16)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.012)
        colorbar.ax.tick_params(labelsize=12)
        colorbar.set_label("bad  ←  0  →  good", fontsize=12)
    axes[-1].set_xlabel("Skipped target layer index", fontsize=15)
    axes[-1].set_xticks(range(0, NUM_LAYERS, 2))
    axes[-1].set_xticklabels([str(index) for index in range(0, NUM_LAYERS, 2)])
    fig.suptitle(
        "Llama-2-13B + SAMD  ·  Single-layer skip ablation across domains",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.savefig(
        output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    install_transformers_compat()
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    cfg = TARGETS["llama2_13b"]
    summaries: dict[str, Any] = {}

    if not args.plot_only:
        from transformers import AutoTokenizer

        from scripts.run_hydra_samd_skip_greedy import eval_samd, load_samd_model

        print("[INFO] loading Llama-2-13B + SAMD[Token-Recycle] ...", flush=True)
        started = time.time()
        tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
        model = load_samd_model(
            cfg["path"],
            eagle_path=None,
            size=cfg["size"],
            tree_method="token_recycle",
            use_safetensors=False,
            token_tree_path=TOKEN_TREE,
            cache_type="static",
            max_cache_len=2048,
        )
        print(f"[INFO] loaded in {time.time() - started:.1f}s", flush=True)
        try:
            for dataset in DOMAINS:
                csv_path = args.output_dir / f"per_layer_{dataset}.csv"
                summary_path = (
                    args.output_dir / f"per_layer_{dataset}_summary.json"
                )
                if csv_path.exists() and summary_path.exists():
                    summaries[dataset] = json.loads(summary_path.read_text())
                    print(f"[SKIP] {dataset} already done", flush=True)
                    continue
                items = load_items(
                    dataset,
                    num_requests=args.num_requests,
                    seed=args.seed,
                    output_len=args.output_len,
                    target="llama2_13b",
                    tokenizer=tokenizer,
                )
                summary, rows = run_domain(model, dataset, items, eval_samd)
                write_csv(csv_path, rows)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                summaries[dataset] = summary
        finally:
            del model
            torch.cuda.empty_cache()
        (args.output_dir / "per_layer_all_summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )

    rows = load_rows(args.output_dir)
    if not rows:
        raise SystemExit("No per-layer rows found")
    write_csv(args.output_dir / "per_layer_all_domains.csv", rows)
    plot_heatmap(
        rows,
        args.output_dir / "fig_samd_13b_layer_ablation_heatmap",
    )
    print("[OK] SAMD 13B two-panel heatmap written", flush=True)


if __name__ == "__main__":
    main()
