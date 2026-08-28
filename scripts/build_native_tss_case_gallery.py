#!/usr/bin/env python3
"""Collect Native vs TSS answers and render a clear multi-case gallery."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spec_exp.benchmark_config import SCORE_CATEGORY  # noqa: E402
from spec_exp.benchmark_datasets import load_dataset_split  # noqa: E402
from spec_exp.io import ensure_dir, write_json  # noqa: E402
from spec_exp.self_spec_decode import DecodeItem  # noqa: E402
from spec_exp.task_score import score_generation_detail  # noqa: E402
from spec_exp.transformers_compat import install_transformers_compat  # noqa: E402

EAGLE7_SKIPS = {
    "translation": [3, 25, 30],
    "summarization": [15, 22, 23],
    "qa": [6, 8, 19, 21, 27],
    "mmlu": [3, 7, 9, 14, 20],
}
SAMD13_SKIPS = {
    "translation": [7, 11, 20, 31, 35, 38],
    "summarization": [12, 20, 27, 33, 34, 38],
    "qa": [9, 10, 11, 19, 28, 38],
    "mmlu": [7, 23, 24, 26, 28, 32, 33],
}
LABELS = {
    "translation": "Translation",
    "summarization": "Summarization",
    "qa": "QA",
    "mmlu": "MMLU",
}
METRIC = {
    "translation": "BLEU",
    "summarization": "ROUGE-L",
    "qa": "F1",
    "mmlu": "Acc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--action",
        choices=["collect7", "collect13", "render", "all"],
        default="all",
    )
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-len", type=int, default=96)
    return parser.parse_args()


def extract_user(prompt: str) -> str:
    if "<|im_start|>user\n" in prompt:
        return prompt.split("<|im_start|>user\n", 1)[1].split(
            "\n<|im_start|>assistant", 1
        )[0]
    if "[INST]" in prompt:
        return prompt.split("[INST]", 1)[1].split("[/INST]", 1)[0].strip()
    if " USER: " in prompt and " ASSISTANT:" in prompt:
        return prompt.split(" USER: ", 1)[1].rsplit(" ASSISTANT:", 1)[0].strip()
    if prompt.startswith("USER: ") and " ASSISTANT:" in prompt:
        return prompt[len("USER: ") :].rsplit(" ASSISTANT:", 1)[0].strip()
    return prompt


def wrap_vicuna(user: str) -> str:
    return (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's "
        f"questions. USER: {user.strip()} ASSISTANT:"
    )


def wrap_llama2(user: str) -> str:
    return f"<s>[INST] {user.strip()} [/INST] "


def load_items(
    dataset: str,
    *,
    style: str,
    num_requests: int,
    train_size: int,
    seed: int,
    output_len: int,
) -> list[DecodeItem]:
    raw = load_dataset_split(
        dataset,
        split="all",
        train_size=train_size,
        seed=seed,
        output_len=output_len,
    )[:80][train_size:][:num_requests]
    items = []
    for item in raw:
        user = extract_user(item.prompt)
        prompt = wrap_vicuna(user) if style == "vicuna" else wrap_llama2(user)
        items.append(
            DecodeItem(
                request_id=item.request_id,
                prompt=prompt,
                max_tokens=item.max_tokens,
                category=item.category,
                reference=item.reference,
            )
        )
    return items


def clean_problem(dataset: str, prompt: str) -> str:
    user = extract_user(prompt)
    if dataset == "translation":
        return user.removeprefix("Translate German to English: ").strip()
    if dataset == "summarization":
        for prefix in (
            "Summarize the following article in one sentence:\n",
            "Summarize:\n",
            "Summarize: ",
        ):
            if user.startswith(prefix):
                user = user[len(prefix) :]
        return user.strip()[:520]
    if dataset == "mmlu":
        question = user.split("Question:", 1)[-1]
        return question.rsplit("Answer:", 1)[0].strip()
    if dataset == "qa":
        user = user.replace("Answer the question concisely.\n\n", "")
        user = user.replace("Question: ", "")
        return user.strip()
    return user.strip()


def looks_degenerate(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    if compact.count("#") >= 8:
        return True
    if len(set(compact)) <= 3 and len(compact) >= 12:
        return True
    # long repeated boilerplate after a short answer
    if "The 2019" in text or "World Outlook" in text:
        return True
    return False


def clean_answer(dataset: str, text: str) -> str:
    text = text.strip()
    text = re.sub(r"^Sure[!.,]?\s*(Here's|Here is).*?:\s*", "", text, flags=re.I)
    text = text.split("Here's a breakdown", 1)[0].strip()
    if dataset == "translation":
        # Keep only the first sentence-ish chunk for display.
        text = text.strip().strip('"')
        text = re.split(r"\n\n|\. The 20|\. This study", text, maxsplit=1)[0].strip()
        if not text.endswith("."):
            text = text + ("." if len(text.split()) <= 20 else "")
    if dataset == "qa":
        text = text.split("\n")[0].strip()
        text = re.split(r"\. The 20", text, maxsplit=1)[0].strip()
    if dataset == "mmlu":
        letter = re.findall(r"(?:^|\b)([A-D])(?:[.)\s]|$)", text)
        first = text.split("\n")[0].strip()
        if letter:
            return f"{letter[0]} — {first[:220]}"
    return text[:320]


def score_one(
    dataset: str, hypothesis: str, reference: str | None, request_id: str, question: str
) -> float:
    return float(
        score_generation_detail(
            category=SCORE_CATEGORY[dataset],
            hypothesis=hypothesis,
            reference=reference,
            request_id=request_id,
            question=question,
        ).primary
    )


def pick_cases(
    payload: dict[str, Any],
    items: list[DecodeItem],
    *,
    model_tag: str,
    method: str,
    max_cases: int = 2,
) -> list[dict[str, Any]]:
    dataset = payload["dataset"]
    base = payload["baseline"]["hypotheses"]
    cand = payload["candidate"]["hypotheses"]
    scored: list[dict[str, Any]] = []
    for item in items:
        rid = item.request_id
        if rid not in base or rid not in cand:
            continue
        native = base[rid]
        tss = cand[rid]
        if native.strip() == tss.strip():
            continue
        # TSS must be readable; Native may be degenerate (useful contrast).
        if looks_degenerate(tss):
            continue
        native_disp = clean_answer(dataset, native)
        tss_disp = clean_answer(dataset, tss)
        if native_disp.strip() == tss_disp.strip():
            continue
        if looks_degenerate(tss_disp):
            continue
        native_score = score_one(dataset, native, item.reference, rid, item.prompt)
        tss_score = score_one(dataset, tss, item.reference, rid, item.prompt)
        # Prefer clear qualitative contrast where TSS improves the metric.
        if tss_score + 1e-6 < native_score:
            continue
        contrast = (tss_score - native_score) + 0.05 * min(len(native_disp), 80) / 80
        if looks_degenerate(native) and not looks_degenerate(tss):
            contrast += 0.35  # Native drifted / hallucinated; TSS stayed on task
        if dataset == "mmlu":
            n_letter = re.findall(r"\b([A-D])\b", native_disp)
            t_letter = re.findall(r"\b([A-D])\b", tss_disp)
            if n_letter and t_letter and n_letter[0] != t_letter[0]:
                contrast += 0.5
            elif n_letter and t_letter and n_letter[0] == t_letter[0]:
                # Same letter but different wording is weaker for showcase.
                contrast -= 0.2
        # For display, keep a short Native snippet that still shows the failure mode.
        if looks_degenerate(native):
            native_disp = clean_answer(dataset, native.split("\n")[0][:180] + " …")
        scored.append(
            {
                "model": model_tag,
                "method": method,
                "dataset": dataset,
                "domain": LABELS[dataset],
                "request_id": rid,
                "problem": clean_problem(dataset, item.prompt),
                "reference": str(item.reference) if item.reference is not None else "",
                "native_answer": native_disp,
                "tss_answer": tss_disp,
                "native_raw": native,
                "tss_raw": tss,
                "metric": METRIC[dataset],
                "native_score": native_score,
                "tss_score": tss_score,
                "skip_layers": payload["skip_layers"],
                "overall": {
                    "native_accept": payload["baseline"]["mean_accepted_per_step"],
                    "tss_accept": payload["candidate"]["mean_accepted_per_step"],
                    "native_score": payload["baseline"]["task_score"],
                    "tss_score": payload["candidate"]["task_score"],
                    "native_toks": payload["baseline"]["tok_per_s"],
                    "tss_toks": payload["candidate"]["tok_per_s"],
                },
                "contrast": contrast,
            }
        )
    scored.sort(key=lambda row: row["contrast"], reverse=True)
    # diversify: keep one stronger TSS-win and one interesting rewrite if available
    picked: list[dict[str, Any]] = []
    for row in scored:
        if len(picked) >= max_cases:
            break
        if not picked:
            picked.append(row)
            continue
        if row["tss_score"] >= row["native_score"] and picked[0]["tss_score"] < picked[0]["native_score"]:
            picked.append(row)
        elif abs(row["tss_score"] - picked[0]["tss_score"]) > 1e-6 or row["request_id"] != picked[0]["request_id"]:
            if all(row["request_id"] != p["request_id"] for p in picked):
                picked.append(row)
    return picked[:max_cases]


def eval_eagle_with_hypotheses(
    model: Any,
    items: list[DecodeItem],
    skip_layers: set[int],
    *,
    domain: str,
    baseline_hypotheses: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Like eval_skip_config, but keep per-request hypotheses for case studies."""
    import math
    import time

    import torch
    from eagle.model.utils import reset_tree_mode
    from eagle3_resource_profile_pod_v2 import clear_kv

    from scripts.run_vicuna13_eagle3_skip_sweep import traced_eagenerate_skip
    from spec_exp.task_score import mean_task_score_detailed

    tokenizer = model.get_tokenizer()
    hypotheses: dict[str, str] = {}
    references: dict[str, str | None] = {}
    questions: dict[str, str] = {}
    total_drafted = total_accepted = total_verify = total_out = 0
    started = time.perf_counter()
    for item in items:
        ids = tokenizer(item.prompt, return_tensors="pt").input_ids.to("cuda")
        plen = int(ids.shape[1])
        max_length = plen + item.max_tokens + 256
        clear_kv(model)
        reset_tree_mode(model)
        model.ea_layer.reset_kv()
        out_ids, trace = traced_eagenerate_skip(
            model,
            ids,
            request_id=item.request_id,
            skip_layers=skip_layers,
            temperature=0.0,
            max_new_tokens=item.max_tokens,
            max_length=max_length,
        )
        gen_ids = out_ids[0, plen:].tolist()
        hypotheses[item.request_id] = (
            tokenizer.decode(gen_ids, skip_special_tokens=True) if gen_ids else ""
        )
        references[item.request_id] = item.reference
        questions[item.request_id] = item.prompt
        total_out += len(gen_ids)
        total_verify += len(trace)
        total_drafted += sum(int(row["drafted_len"]) for row in trace)
        total_accepted += sum(int(row["accepted_len"]) for row in trace)
        torch.cuda.empty_cache()
    wall = time.perf_counter() - started
    has_ref = any(value for value in references.values())
    score = mean_task_score_detailed(
        category=domain,
        hypotheses=hypotheses,
        references=references,
        baseline_hypotheses=None if has_ref else baseline_hypotheses,
        questions=questions,
    )
    return {
        "skip_layers": sorted(skip_layers),
        "num_skip_layers": len(skip_layers),
        "accept_rate": total_accepted / total_drafted if total_drafted else math.nan,
        "mean_accepted_per_step": 1.0 + total_accepted / max(total_verify, 1),
        "task_score": score["mean_score"],
        "wall_s": wall,
        "total_output_tokens": total_out,
        "tok_per_s": total_out / wall if wall > 0 else math.nan,
        "num_verify_steps": total_verify,
        "hypotheses": hypotheses,
    }


def collect_eagle7(output_dir: Path, args: argparse.Namespace) -> None:
    import torch

    from scripts.run_tss_max_toks_pipeline import load_split
    from scripts.run_vicuna13_eagle3_skip_sweep import MODEL_PRESETS, load_model

    raw_dir = ensure_dir(output_dir / "raw")
    print("[INFO] loading Vicuna-7B + EAGLE ...", flush=True)
    cfg = MODEL_PRESETS["vicuna7"]
    model = load_model(
        base_model=cfg["base_model"],
        ea_model=cfg["ea_model"],
        total_token=60,
        use_eagle3=False,
    )
    try:
        for dataset, skip in EAGLE7_SKIPS.items():
            out = raw_dir / f"eagle7_{dataset}_outputs.json"
            if out.exists():
                payload = json.loads(out.read_text())
                if payload.get("baseline", {}).get("hypotheses"):
                    print(f"[SKIP] {out.name}", flush=True)
                    continue
            items = load_split(
                dataset,
                "test",
                train_size=args.train_size,
                seed=args.seed,
                output_len=args.output_len,
            )[: args.num_requests]
            domain = SCORE_CATEGORY[dataset]
            print(f"[7B-EAGLE] {dataset} skip={skip} n={len(items)}", flush=True)
            baseline = eval_eagle_with_hypotheses(
                model, items, set(), domain=domain
            )
            candidate = eval_eagle_with_hypotheses(
                model,
                items,
                set(skip),
                domain=domain,
                baseline_hypotheses=baseline.get("hypotheses"),
            )
            write_json(
                {
                    "model": "Vicuna-7B",
                    "method": "EAGLE",
                    "dataset": dataset,
                    "split": "test64",
                    "skip_layers": skip,
                    "baseline": baseline,
                    "candidate": candidate,
                },
                out,
            )
            print(
                f"  saved hyps={len(baseline['hypotheses'])} "
                f"accept {baseline['mean_accepted_per_step']:.3f}→"
                f"{candidate['mean_accepted_per_step']:.3f}",
                flush=True,
            )
    finally:
        del model
        torch.cuda.empty_cache()


def collect_samd13(output_dir: Path, args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    from scripts.run_hydra_samd_skip_greedy import eval_samd, load_samd_model
    from scripts.run_samd_target_skip_search import TARGETS, extract_user, render_prompt

    raw_dir = ensure_dir(output_dir / "raw")
    # Reuse prior translation/mmlu if present in old case-study folder.
    legacy = REPO / "results" / "tss_case_studies_20260716" / "raw"
    for dataset in ("translation", "mmlu"):
        src = legacy / f"{dataset}_outputs.json"
        dst = raw_dir / f"samd13_{dataset}_outputs.json"
        if dst.exists():
            continue
        if src.exists():
            payload = json.loads(src.read_text())
            payload["model"] = "Llama-2-13B"
            payload["method"] = "SAMD"
            write_json(payload, dst)
            print(f"[REUSE] {src.name} -> {dst.name}", flush=True)

    cfg = TARGETS["llama2_13b"]
    token_tree = str(
        REPO
        / "data"
        / "Spec-Bench-repo"
        / "model"
        / "samd"
        / "config"
        / "token_recycle_4_15.json"
    )
    need = [
        dataset
        for dataset in SAMD13_SKIPS
        if not (raw_dir / f"samd13_{dataset}_outputs.json").exists()
    ]
    if not need:
        print("[SKIP] all SAMD13 outputs present", flush=True)
        return
    print("[INFO] loading Llama-2-13B + SAMD ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    model = load_samd_model(
        cfg["path"],
        eagle_path=None,
        size="13b",
        tree_method="token_recycle",
        use_safetensors=False,
        token_tree_path=token_tree,
        cache_type="static",
        max_cache_len=2048,
    )
    try:
        for dataset in need:
            skip = SAMD13_SKIPS[dataset]
            out = raw_dir / f"samd13_{dataset}_outputs.json"
            items = load_items(
                dataset,
                style="llama2",
                num_requests=args.num_requests,
                train_size=args.train_size,
                seed=args.seed,
                output_len=args.output_len,
            )
            # re-render with tokenizer chat template path for consistency
            items = [
                DecodeItem(
                    request_id=item.request_id,
                    prompt=render_prompt(
                        tokenizer, extract_user(item.prompt), "llama2_13b"
                    ),
                    max_tokens=item.max_tokens,
                    category=item.category,
                    reference=item.reference,
                )
                for item in items
            ]
            domain = SCORE_CATEGORY[dataset]
            print(f"[13B-SAMD] {dataset} skip={skip} n={len(items)}", flush=True)
            baseline = eval_samd(
                model, items, set(), baseline_hypotheses=None, domain=domain
            )
            candidate = eval_samd(
                model,
                items,
                set(skip),
                baseline_hypotheses=baseline.get("hypotheses"),
                domain=domain,
            )
            write_json(
                {
                    "model": "Llama-2-13B",
                    "method": "SAMD",
                    "target": "llama2_13b",
                    "dataset": dataset,
                    "split": "test64",
                    "skip_layers": skip,
                    "baseline": baseline,
                    "candidate": candidate,
                },
                out,
            )
    finally:
        del model
        torch.cuda.empty_cache()


def wrap(text: str, width: int = 88) -> str:
    return "\n".join(
        textwrap.fill(line, width=width) if line.strip() else ""
        for line in text.splitlines()
    )


def render_gallery(cases: list[dict[str, Any]], output: Path) -> None:
    n = len(cases)
    fig_h = 2.55 * n + 0.8
    fig, axes = plt.subplots(n, 1, figsize=(12.8, fig_h))
    if n == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        title = (
            f"{case['model']} + {case['method']}  ·  {case['domain']}  ·  "
            f"{case['request_id']}  ·  skip={case['skip_layers']}"
        )
        ax.text(0.01, 0.96, title, fontsize=11, fontweight="bold", va="top")
        score_line = (
            f"{case['metric']}: Native {case['native_score']:.3f} → "
            f"TSS {case['tss_score']:.3f}"
        )
        ax.text(0.99, 0.96, score_line, fontsize=10, ha="right", va="top", color="#334155")

        def panel(x: float, y: float, w: float, h: float, color: str, label: str, body: str) -> None:
            patch = FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.02",
                linewidth=1.0,
                edgecolor="#CBD5E1",
                facecolor=color,
            )
            ax.add_patch(patch)
            ax.text(x + 0.012, y + h - 0.045, label, fontsize=9, fontweight="bold", color="#0F172A")
            ax.text(
                x + 0.012,
                y + h - 0.09,
                wrap(body, width=52 if w < 0.5 else 100),
                fontsize=8.2,
                va="top",
                family="DejaVu Sans",
                color="#111827",
            )

        panel(0.01, 0.55, 0.98, 0.30, "#F8FAFC", "Problem", case["problem"][:420])
        panel(0.01, 0.08, 0.485, 0.42, "#FEF2F2", "Native", case["native_answer"])
        panel(0.505, 0.08, 0.485, 0.42, "#ECFDF5", "TSS", case["tss_answer"])
        if case.get("reference"):
            ax.text(
                0.01,
                0.02,
                "Ref: " + wrap(str(case["reference"])[:180], width=140),
                fontsize=7.5,
                color="#64748B",
                va="bottom",
            )
    fig.suptitle(
        "Native vs TSS qualitative cases (held-out outputs)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_markdown(cases: list[dict[str, Any]], output: Path) -> None:
    lines = ["# Native vs TSS Case Gallery", ""]
    for index, case in enumerate(cases, start=1):
        lines.extend(
            [
                f"## {index}. {case['model']} + {case['method']} / {case['domain']}",
                f"- id: `{case['request_id']}`",
                f"- skip layers: `{case['skip_layers']}`",
                f"- {case['metric']}: Native **{case['native_score']:.3f}** → TSS **{case['tss_score']:.3f}**",
                "",
                "**Problem**",
                "",
                case["problem"],
                "",
                "**Native**",
                "",
                case["native_answer"],
                "",
                "**TSS**",
                "",
                case["tss_answer"],
                "",
            ]
        )
        if case.get("reference"):
            lines.extend(["**Reference**", "", str(case["reference"]), ""])
        lines.append("---")
        lines.append("")
    output.write_text("\n".join(lines))


def build_cases(output_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_dir = output_dir / "raw"
    cases: list[dict[str, Any]] = []
    # 7B
    for dataset in EAGLE7_SKIPS:
        path = raw_dir / f"eagle7_{dataset}_outputs.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        items = load_items(
            dataset,
            style="vicuna",
            num_requests=args.num_requests,
            train_size=args.train_size,
            seed=args.seed,
            output_len=args.output_len,
        )
        cases.extend(
            pick_cases(payload, items, model_tag="Vicuna-7B", method="EAGLE", max_cases=2)
        )
    # 13B
    for dataset in SAMD13_SKIPS:
        path = raw_dir / f"samd13_{dataset}_outputs.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        items = load_items(
            dataset,
            style="llama2",
            num_requests=args.num_requests,
            train_size=args.train_size,
            seed=args.seed,
            output_len=args.output_len,
        )
        cases.extend(
            pick_cases(payload, items, model_tag="Llama-2-13B", method="SAMD", max_cases=2)
        )
    return cases


def main() -> None:
    args = parse_args()
    install_transformers_compat()
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.action in {"collect7", "all"}:
        collect_eagle7(args.output_dir, args)
    if args.action in {"collect13", "all"}:
        collect_samd13(args.output_dir, args)
    if args.action in {"render", "all"}:
        cases = build_cases(args.output_dir, args)
        if not cases:
            raise SystemExit("No cases available to render")
        write_json({"cases": cases}, args.output_dir / "case_gallery.json")
        render_gallery(cases, args.output_dir / "fig_native_tss_case_gallery")
        render_markdown(cases, args.output_dir / "case_gallery.md")
        # also split figures for 7B / 13B
        cases7 = [case for case in cases if case["model"] == "Vicuna-7B"]
        cases13 = [case for case in cases if case["model"] == "Llama-2-13B"]
        if cases7:
            render_gallery(cases7, args.output_dir / "fig_native_tss_cases_7b")
        if cases13:
            render_gallery(cases13, args.output_dir / "fig_native_tss_cases_13b")
        print(f"[OK] {len(cases)} cases -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
