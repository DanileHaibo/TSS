#!/usr/bin/env python3
"""Build paper main table: Native vs TSS (SLEB max-accept preserve-score skip).

Columns: Domain & Method & Accept Len. & Task Metric & Skip Layers & Sparsity &
         Target Latency & E2E Latency & Speedup
Latency in ms/token; Speedup = E2E tok/s / vanilla tok/s (Spec-Bench style).
Sparsity = (# skipped target layers) / (# total layers).
"""
from __future__ import annotations

import json
from pathlib import Path

HELD = Path("/root/autodl-tmp/specdecode-system-exp/results/full_872_20260629_132522/heldout")
OUT = Path("/root/autodl-tmp/experiments-data/paper_figure")
OUT.mkdir(parents=True, exist_ok=True)

VANILLA_DIRS = {
    "7b": Path("/root/autodl-tmp/specdecode-system-exp/results/specbench_nontree_4domains_20260615"),
    "13b": Path("/root/autodl-tmp/specdecode-system-exp/results/specbench_nontree_13b_4domains_20260615"),
    "33b": Path("/root/autodl-tmp/specdecode-system-exp/results/specbench_nontree_33b_4domains_20260710"),
}

# Vicuna-v1.3 hidden layers
N_LAYERS = {"7b": 32, "13b": 40, "33b": 60}

PRIMARY = {
    "7b": ("samd", "SAMD"),
    "13b": ("eagle3", "EAGLE3"),
    "33b": ("hydra", "Hydra"),
}

DOMAINS = [
    ("translation", "Translation", "BLEU"),
    ("summarization", "Summarization", "ROUGE-L"),
    ("gsm8k", "GSM8K", "Acc."),
    ("rag", "RAG", "F1"),
]


def load_vanilla() -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for size, folder in VANILLA_DIRS.items():
        if not folder.exists():
            continue
        for f in folder.glob("*_vanilla.json"):
            d = json.loads(f.read_text())
            tps = d.get("tok_per_s")
            if tps is None and d.get("wall_s") and d.get("total_new_tokens"):
                tps = d["total_new_tokens"] / d["wall_s"]
            if tps:
                out[(size, d["dataset"])] = float(tps)
    return out


def tps_of(m: dict) -> float:
    if m.get("tok_per_s") is not None:
        return float(m["tok_per_s"])
    return float(m["total_output_tokens"]) / float(m["wall_s"])


def fmt_lat(ms: float) -> str:
    return f"{ms:.1f}"


def fmt_sp(x: float) -> str:
    return f"{x:.2f}$\\times$"


def fmt_acc(x: float) -> str:
    return f"{x:.2f}"


def fmt_score(x: float, metric: str) -> str:
    if metric == "Acc.":
        return f"{100.0 * x:.1f}\\%"
    return f"{x:.3f}"


def fmt_skip(layers: list[int]) -> str:
    if not layers:
        return "--"
    inner = ",".join(str(i) for i in layers)
    return "$\\{" + inner + "\\}$"


def fmt_sparsity(pct: float) -> str:
    if pct < 1e-9:
        return "0\\%"
    return f"{pct:.1f}\\%"


def collect_rows(vanilla: dict[tuple[str, str], float]) -> list[dict]:
    rows: list[dict] = []
    for size, (method_key, method_name) in PRIMARY.items():
        n_layers = N_LAYERS[size]
        for ds, ds_label, metric in DOMAINS:
            path = HELD / f"{ds}_{size}_{method_key}_heldout.json"
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            v_tps = vanilla.get((size, ds))
            for variant, key, mlabel in [
                ("Native", "baseline", method_name),
                ("TSS", "best", f"{method_name}+TSS"),
            ]:
                m = d["test_eval"][key]
                e2e_tps = tps_of(m)
                e2e_ms = 1000.0 / e2e_tps if e2e_tps > 0 else float("nan")
                tgt_ms = 1000.0 / v_tps if v_tps else float("nan")
                speedup = e2e_tps / v_tps if v_tps else float("nan")
                skip = list(m.get("skip_layers") or [])
                sparsity = 100.0 * len(skip) / n_layers
                rows.append(
                    {
                        "size": size,
                        "domain": ds,
                        "domain_label": ds_label,
                        "metric": metric,
                        "method_key": method_key,
                        "method": mlabel,
                        "variant": variant,
                        "accept": float(m["mean_accepted_per_step"]),
                        "score": float(m["task_score"]),
                        "target_ms": tgt_ms,
                        "e2e_ms": e2e_ms,
                        "speedup": speedup,
                        "skip": skip,
                        "n_layers": n_layers,
                        "sparsity": sparsity,
                        "e2e_tps": e2e_tps,
                        "vanilla_tps": v_tps,
                    }
                )
    return rows


def latex_table_compact(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("% Compact main table: Native vs +TSS paired.")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Main results on four Spec-Bench domains (held-out, 72 prompts). "
        r"\textbf{Native}: speculative decoding without layer skip; "
        r"\textbf{+TSS}: SLEB-style multi-layer \emph{target} skip maximizing accept length "
        r"while preserving the train-split task metric. "
        r"Accept Len.\ is mean accepted tokens per verify step. "
        r"Skip Layers / Sparsity are the TSS-selected target layers and "
        r"$|\mathcal{S}|/L$ (Vicuna-7B/13B/33B have $L{=}32/40/60$). "
        r"Target / E2E Latency are ms/token for vanilla AR and speculative decoding; "
        r"Speedup is vs.\ vanilla AR on the same device (RTX~4090). "
        r"Vicuna-33B Target Latency is estimated as local Hydra tok/s divided by "
        r"Spec-Bench A100 Hydra domain speedups (33B FP16 AR does not fit without CPU offload). "
        r"Spec-Bench A100 references: "
        r"7B SAMD[EAGLE2] 2.73$\times$/4.58, "
        r"13B EAGLE3 3.02$\times$/5.71, "
        r"33B Hydra 2.22$\times$/3.24 "
        r"\citep{https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md}.}"
    )
    lines.append(r"\label{tab:main_tss}")
    lines.append(r"\setlength{\tabcolsep}{2.8pt}")
    lines.append(r"\begin{tabular}{llccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Domain & Method & Accept Len. & Task Metric & Skip Layers & Sparsity & "
        r"Target Lat. & E2E Lat. & Speedup \\"
    )
    lines.append(r" &  &  &  &  &  & (ms/tok) & (ms/tok) &  \\")
    lines.append(r"\midrule")

    by: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        by.setdefault((r["size"], r["domain"]), {})[r["variant"]] = r

    first_size = True
    for size, (mk, mn) in PRIMARY.items():
        if not first_size:
            lines.append(r"\midrule")
        first_size = False
        nL = N_LAYERS[size]
        lines.append(
            r"\multicolumn{9}{l}{\textbf{Vicuna-"
            + size.upper()
            + r"} ("
            + mn
            + r" / "
            + mn
            + r"+TSS, $L{=}"
            + str(nL)
            + r"$)} \\"
        )
        lines.append(r"\midrule")
        for ds, ds_label, metric in DOMAINS:
            pair = by.get((size, ds))
            if not pair or "Native" not in pair:
                continue
            for variant in ("Native", "TSS"):
                r = pair[variant]
                dom = ds_label if variant == "Native" else ""
                method = mn if variant == "Native" else (mn + "+TSS")
                sp_s = fmt_sp(r["speedup"]) if r["speedup"] == r["speedup"] else "--"
                if (
                    variant == "TSS"
                    and r["speedup"] == r["speedup"]
                    and pair["Native"]["speedup"] == pair["Native"]["speedup"]
                    and r["speedup"] > pair["Native"]["speedup"] + 0.05
                ):
                    sp_s = r"\textbf{" + sp_s + "}"
                acc_s = fmt_acc(r["accept"])
                if variant == "TSS" and r["accept"] > pair["Native"]["accept"] + 0.10:
                    acc_s = r"\textbf{" + acc_s + "}"
                tgt = fmt_lat(r["target_ms"]) if r["target_ms"] == r["target_ms"] else "--"
                e2e = fmt_lat(r["e2e_ms"]) if r["e2e_ms"] == r["e2e_ms"] else "--"
                score_s = fmt_score(r["score"], metric)
                skip_s = "--" if variant == "Native" else fmt_skip(r["skip"])
                spars_s = "--" if variant == "Native" else fmt_sparsity(r["sparsity"])
                lines.append(
                    f"{dom} & {method} & {acc_s} & {score_s} & {skip_s} & {spars_s} & "
                    f"{tgt} & {e2e} & {sp_s} \\\\"
                )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def markdown_preview(rows: list[dict]) -> str:
    lines = [
        "| Size | Domain | Method | Accept | Metric | Skip | Sparsity | Target ms/tok | E2E ms/tok | Speedup |",
        "|------|--------|--------|--------|--------|------|----------|---------------|------------|---------|",
    ]
    for r in rows:
        sk = str(r["skip"]) if r["variant"] == "TSS" else "--"
        spars = f"{r['sparsity']:.1f}%" if r["variant"] == "TSS" else "--"
        tgt = f"{r['target_ms']:.1f}" if r["target_ms"] == r["target_ms"] else "NA"
        e2e = f"{r['e2e_ms']:.1f}" if r["e2e_ms"] == r["e2e_ms"] else "NA"
        sp = f"{r['speedup']:.2f}x" if r["speedup"] == r["speedup"] else "NA"
        lines.append(
            f"| {r['size']} | {r['domain_label']} | {r['method']} | {r['accept']:.2f} | "
            f"{r['score']:.3f} | {sk} | {spars} | {tgt} | {e2e} | {sp} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    vanilla = load_vanilla()
    rows = collect_rows(vanilla)
    missing_van = [
        (s, ds) for s in PRIMARY for ds, _, _ in DOMAINS if (s, ds) not in vanilla
    ]
    tex = latex_table_compact(rows)
    (OUT / "tab_main_tss.tex").write_text(tex, encoding="utf-8")
    (OUT / "tab_main_tss.md").write_text(markdown_preview(rows), encoding="utf-8")
    payload = {
        "rows": rows,
        "vanilla_tps": {f"{k[0]}/{k[1]}": v for k, v in vanilla.items()},
        "missing_vanilla": [f"{a}/{b}" for a, b in missing_van],
        "n_layers": N_LAYERS,
        "note": (
            "Target Latency = 1000/vanilla_tok_per_s; "
            "E2E Latency = 1000/spec_tok_per_s; "
            "Speedup = spec_tok_per_s / vanilla_tok_per_s; "
            "Sparsity = len(skip_layers)/N_LAYERS"
        ),
    }
    (OUT / "tab_main_tss.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(markdown_preview(rows))
    print("missing vanilla:", missing_van)
    print("wrote", OUT / "tab_main_tss.tex")


if __name__ == "__main__":
    main()
