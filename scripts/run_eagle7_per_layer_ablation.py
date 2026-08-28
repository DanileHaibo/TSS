#!/usr/bin/env python3
"""Vicuna-7B + EAGLE single-target-layer ablation on four main domains."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_eagle3_13b_per_layer_ablation import (  # noqa: E402
    apply_style,
    load_all_rows,
    matrix,
    run_domain,
    write_csv,
)
from scripts.run_vicuna13_eagle3_skip_sweep import (  # noqa: E402
    MODEL_PRESETS,
    _resolve_vicuna7_base,
    load_model,
)

DOMAINS = ("translation", "summarization", "qa", "mmlu")
DOMAIN_SHORT = {
    "translation": "Translation",
    "summarization": "Summarization",
    "qa": "QA",
    "mmlu": "MMLU",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--max-loss-tokens", type=int, default=128)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def plot_heatmap(rows: list[dict], output: Path) -> None:
    apply_style()
    metrics = [
        (
            "delta_accept",
            r"$\Delta$ Accept Length  (skip L $-$ baseline)",
            "RdBu",
        ),
        (
            "delta_metric",
            r"$\Delta$ Domain Metric  (skip L $-$ baseline)",
            "RdBu",
        ),
        (
            "delta_loss",
            r"$\Delta$ CE Loss  (skip L $-$ baseline)",
            "RdBu_r",
        ),
    ]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.5, 7.2),
        sharex=True,
        gridspec_kw={"hspace": 0.30},
    )
    ylabels = [DOMAIN_SHORT[dataset] for dataset in DOMAINS]
    for ax, (key, title, cmap) in zip(axes, metrics):
        values = matrix(rows, list(DOMAINS), key, 32)
        vmax = max(float(np.nanmax(np.abs(values))), 1e-6)
        image = ax.imshow(
            values,
            aspect="auto",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_yticks(range(len(DOMAINS)))
        ax.set_yticklabels(ylabels)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.015, pad=0.01)
        colorbar.ax.tick_params(labelsize=8)
        colorbar.set_label("bad  ←  0  →  good", fontsize=8)

    axes[-1].set_xlabel("Skipped target layer index")
    axes[-1].set_xticks(range(0, 32, 2))
    axes[-1].set_xticklabels([str(index) for index in range(0, 32, 2)])
    fig.suptitle(
        "Vicuna-7B + EAGLE  ·  Single-layer skip ablation across domains",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.01,
        0.01,
        "Blue is favorable (higher accept/metric or lower loss); red is unfavorable.",
        fontsize=8,
        color="#555555",
        style="italic",
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
    cfg = dict(MODEL_PRESETS["vicuna7"])
    cfg["base_model"] = _resolve_vicuna7_base()
    num_layers = int(cfg["num_layers"])

    if not args.plot_only:
        print("[INFO] loading Vicuna-7B + EAGLE ...", flush=True)
        started = time.time()
        model = load_model(
            base_model=cfg["base_model"],
            ea_model=cfg["ea_model"],
            total_token=args.total_token,
            use_eagle3=False,
        )
        print(
            f"[INFO] loaded in {time.time() - started:.1f}s "
            f"num_layers={num_layers}",
            flush=True,
        )
        summaries = {}
        try:
            for dataset in DOMAINS:
                csv_path = args.output_dir / f"per_layer_{dataset}.csv"
                json_path = (
                    args.output_dir / f"per_layer_{dataset}_summary.json"
                )
                if (
                    csv_path.exists()
                    and json_path.exists()
                    and csv_path.stat().st_size > 100
                ):
                    print(f"[skip] {dataset} already complete", flush=True)
                    summaries[dataset] = json.loads(
                        json_path.read_text()
                    )["summary"]
                    continue
                summary, rows = run_domain(
                    model,
                    dataset,
                    num_layers=num_layers,
                    num_requests=args.num_requests,
                    seed=args.seed,
                    output_len=args.output_len,
                    max_loss_tokens=args.max_loss_tokens,
                    chat_template=cfg["chat_template"],
                )
                write_csv(csv_path, rows)
                json_path.write_text(
                    json.dumps(
                        {
                            "summary": summary,
                            "baseline": summary["baseline"],
                        },
                        indent=2,
                    )
                    + "\n"
                )
                summaries[dataset] = summary
        finally:
            del model
            torch.cuda.empty_cache()

        all_rows, _ = load_all_rows(args.output_dir, list(DOMAINS))
        write_csv(args.output_dir / "per_layer_all_domains.csv", all_rows)
        (args.output_dir / "per_layer_all_summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )

    rows, _ = load_all_rows(args.output_dir, list(DOMAINS))
    if not rows:
        raise SystemExit(f"No per-layer CSV found in {args.output_dir}")
    if not any(math.isfinite(float(row["delta_accept"])) for row in rows):
        raise SystemExit("No finite per-layer values found")
    plot_heatmap(
        rows,
        args.output_dir / "fig_eagle_7b_layer_ablation_heatmap",
    )
    print("[OK] 7B heatmap written", flush=True)


if __name__ == "__main__":
    main()
