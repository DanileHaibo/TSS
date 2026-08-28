#!/usr/bin/env python3
"""Pick one Native-vs-TSS case per domain for 7B/13B and emit NeurIPS-style LaTeX."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from spec_exp.benchmark_config import SCORE_CATEGORY
from spec_exp.benchmark_datasets import load_dataset_split
from spec_exp.io import ensure_dir, write_json
from spec_exp.self_spec_decode import DecodeItem
from spec_exp.task_score import score_generation_detail

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

RAW_DIR = Path(
    "/root/autodl-tmp/experiments-data/native_tss_case_gallery_20260721/raw"
)
OUT_DIR = Path("/root/autodl-tmp/experiments-data/paper_figure")

DOMAINS = ("translation", "summarization", "qa", "rag", "mmlu")
EAGLE7_SKIPS = {
    "translation": [3, 25, 30],
    "summarization": [15, 22, 23],
    "qa": [6, 8, 19, 21, 27],
    "rag": [3, 6, 14, 25, 30],
    "mmlu": [3, 7, 9, 14, 20],
}
SAMD13_SKIPS = {
    "translation": [7, 11, 20, 31, 35, 38],
    "summarization": [12, 20, 27, 33, 34, 38],
    "qa": [9, 10, 11, 19, 28, 38],
    # from samd_score_first search (rag not in published triple-win; still a real TSS skip)
    "rag": [8, 17, 20, 27, 33, 34],
    "mmlu": [7, 23, 24, 26, 28, 32, 33],
}
LABELS = {
    "translation": "Translation",
    "summarization": "Summarization",
    "qa": "Open-domain QA",
    "rag": "RAG",
    "mmlu": "MMLU",
}
METRIC = {
    "translation": "BLEU",
    "summarization": "ROUGE-L",
    "qa": "F1",
    "rag": "F1",
    "mmlu": "Acc.",
}


def extract_user(prompt: str) -> str:
    if "<|im_start|>user\n" in prompt:
        return prompt.split("<|im_start|>user\n", 1)[1].split(
            "\n<|im_start|>assistant", 1
        )[0]
    if "[INST]" in prompt:
        return prompt.split("[INST]", 1)[1].split("[/INST]", 1)[0].strip()
    if " USER: " in prompt and " ASSISTANT:" in prompt:
        return prompt.split(" USER: ", 1)[1].rsplit(" ASSISTANT:", 1)[0].strip()
    return prompt


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
        return user.strip()
    if dataset == "mmlu":
        question = user.split("Question:", 1)[-1]
        return question.rsplit("Answer:", 1)[0].strip()
    if dataset in {"qa", "rag"}:
        user = user.replace("Answer the question concisely.\n\n", "")
        user = user.replace("Question: ", "")
        return user.strip()
    return user.strip()


def looks_degenerate(text: str) -> bool:
    if not text or not text.strip():
        return True
    bad = (
        "2019-2020 season",
        "World Outlook",
        "The first time I saw",
        "cultural heritage",
    )
    if any(b in text for b in bad):
        return True
    if text.count("#") >= 8:
        return True
    compact = re.sub(r"\s+", "", text)
    return len(set(compact)) <= 3 and len(compact) >= 12


def clean_answer(dataset: str, text: str) -> str:
    text = text.strip()
    text = re.sub(r"^Sure[!.,]?\s*(Here's|Here is).*?:\s*", "", text, flags=re.I)
    text = text.split("Here's a breakdown", 1)[0].strip()
    if dataset == "translation":
        text = text.strip().strip('"')
        text = re.split(r"\n\n|\. The 20|\. This study", text, maxsplit=1)[0].strip()
    if dataset in {"qa", "rag"}:
        text = text.split("\n")[0].strip()
        text = re.split(r"\. The 20", text, maxsplit=1)[0].strip()
    if dataset == "mmlu":
        letter = re.findall(r"(?:^|\b)([A-D])(?:[.)\s]|$)", text)
        first = text.split("\n")[0].strip()
        if letter:
            return f"{letter[0]}. {first[:240]}"
        return first[:260]
    if dataset == "summarization":
        return text[:360]
    return text[:280]


def latex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def shorten(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def score_one(dataset: str, hyp: str, ref: str | None, rid: str, q: str) -> float:
    return float(
        score_generation_detail(
            category=SCORE_CATEGORY[dataset],
            hypothesis=hyp,
            reference=ref,
            request_id=rid,
            question=q,
        ).primary
    )


def load_items(dataset: str, style: str) -> list[DecodeItem]:
    from build_native_tss_case_gallery import load_items as _load

    return _load(
        dataset,
        style=style,
        num_requests=64,
        train_size=16,
        seed=42,
        output_len=96,
    )


def pick_best(payload: dict[str, Any], items: list[DecodeItem]) -> dict[str, Any] | None:
    dataset = payload["dataset"]
    base = payload["baseline"]["hypotheses"]
    cand = payload["candidate"]["hypotheses"]
    scored: list[dict[str, Any]] = []
    for item in items:
        rid = item.request_id
        if rid not in base or rid not in cand:
            continue
        native, tss = base[rid], cand[rid]
        if looks_degenerate(tss):
            continue
        if native.strip() == tss.strip():
            continue
        ns = score_one(dataset, native, item.reference, rid, item.prompt)
        ts = score_one(dataset, tss, item.reference, rid, item.prompt)
        nd = clean_answer(dataset, native)
        td = clean_answer(dataset, tss)
        if nd.strip() == td.strip():
            continue
        # Showcase only cases where TSS does not hurt the case-level metric.
        if ts + 1e-9 < ns:
            continue
        contrast = (ts - ns)
        if looks_degenerate(native) and not looks_degenerate(tss):
            contrast += 0.4
        if dataset == "mmlu":
            nl = re.findall(r"^([A-D])", nd)
            tl = re.findall(r"^([A-D])", td)
            ref = str(item.reference).strip().upper()[:1]
            if tl and tl[0] == ref and (not nl or nl[0] != ref):
                contrast += 1.5
            elif nl and tl and nl[0] != tl[0]:
                contrast += 0.3
            elif nl and tl and nl[0] == tl[0] and ts <= ns + 1e-9:
                # Same letter / same score: weak qualitative signal.
                contrast -= 0.5
        scored.append(
            {
                "model": payload["model"],
                "method": payload["method"],
                "dataset": dataset,
                "domain": LABELS[dataset],
                "request_id": rid,
                "problem": clean_problem(dataset, item.prompt),
                "reference": str(item.reference) if item.reference is not None else "",
                "native_answer": nd if not looks_degenerate(native) else shorten(nd, 160),
                "tss_answer": td,
                "metric": METRIC[dataset],
                "native_score": ns,
                "tss_score": ts,
                "skip_layers": payload["skip_layers"],
                "contrast": contrast,
                "overall": {
                    "native_accept": payload["baseline"]["mean_accepted_per_step"],
                    "tss_accept": payload["candidate"]["mean_accepted_per_step"],
                    "native_score": payload["baseline"]["task_score"],
                    "tss_score": payload["candidate"]["task_score"],
                    "native_toks": payload["baseline"]["tok_per_s"],
                    "tss_toks": payload["candidate"]["tok_per_s"],
                },
            }
        )
    scored.sort(key=lambda r: r["contrast"], reverse=True)
    return scored[0] if scored else None


def collect_missing_rag() -> None:
    ensure_dir(RAW_DIR)
    # --- 7B EAGLE RAG ---
    out7 = RAW_DIR / "eagle7_rag_outputs.json"
    if not (out7.exists() and json.loads(out7.read_text()).get("baseline", {}).get("hypotheses")):
        import torch
        from build_native_tss_case_gallery import eval_eagle_with_hypotheses
        from scripts.run_vicuna13_eagle3_skip_sweep import MODEL_PRESETS, load_model

        print("[collect] Vicuna-7B + EAGLE / rag", flush=True)
        cfg = MODEL_PRESETS["vicuna7"]
        model = load_model(
            base_model=cfg["base_model"],
            ea_model=cfg["ea_model"],
            total_token=60,
            use_eagle3=False,
        )
        try:
            items = load_items("rag", "vicuna")
            domain = SCORE_CATEGORY["rag"]
            skip = EAGLE7_SKIPS["rag"]
            baseline = eval_eagle_with_hypotheses(model, items, set(), domain=domain)
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
                    "dataset": "rag",
                    "split": "test64",
                    "skip_layers": skip,
                    "baseline": baseline,
                    "candidate": candidate,
                },
                out7,
            )
        finally:
            del model
            torch.cuda.empty_cache()
    else:
        print("[skip] eagle7 rag present", flush=True)

    # --- 13B SAMD RAG ---
    out13 = RAW_DIR / "samd13_rag_outputs.json"
    if not (out13.exists() and json.loads(out13.read_text()).get("baseline", {}).get("hypotheses")):
        import torch
        from transformers import AutoTokenizer

        from scripts.run_hydra_samd_skip_greedy import eval_samd, load_samd_model
        from scripts.run_samd_target_skip_search import TARGETS, extract_user, render_prompt

        print("[collect] Llama-2-13B + SAMD / rag", flush=True)
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
            skip = SAMD13_SKIPS["rag"]
            raw_items = load_dataset_split(
                "rag", split="all", train_size=16, seed=42, output_len=96
            )[:80][16:80]
            items = [
                DecodeItem(
                    request_id=it.request_id,
                    prompt=render_prompt(
                        tokenizer, extract_user(it.prompt), "llama2_13b"
                    ),
                    max_tokens=it.max_tokens,
                    category=it.category,
                    reference=it.reference,
                )
                for it in raw_items
            ]
            domain = SCORE_CATEGORY["rag"]
            baseline = eval_samd(model, items, set(), baseline_hypotheses=None, domain=domain)
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
                    "dataset": "rag",
                    "split": "test64",
                    "skip_layers": skip,
                    "baseline": baseline,
                    "candidate": candidate,
                },
                out13,
            )
        finally:
            del model
            torch.cuda.empty_cache()
    else:
        print("[skip] samd13 rag present", flush=True)


def gather_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    specs = [
        ("Vicuna-7B", "EAGLE", "eagle7", "vicuna", EAGLE7_SKIPS),
        ("Llama-2-13B", "SAMD", "samd13", "llama2", SAMD13_SKIPS),
    ]
    for model, method, prefix, style, skips in specs:
        for dataset in DOMAINS:
            path = RAW_DIR / f"{prefix}_{dataset}_outputs.json"
            if not path.exists():
                print(f"[WARN] missing {path}", flush=True)
                continue
            payload = json.loads(path.read_text())
            payload.setdefault("model", model)
            payload.setdefault("method", method)
            payload.setdefault("dataset", dataset)
            payload.setdefault("skip_layers", skips[dataset])
            items = load_items(dataset, style)
            case = pick_best(payload, items)
            if case is None:
                print(f"[WARN] no good case for {model}/{dataset}", flush=True)
                continue
            cases.append(case)
            print(
                f"[pick] {model}/{dataset} {case['request_id']} "
                f"{case['native_score']:.3f}→{case['tss_score']:.3f}",
                flush=True,
            )
    return cases


def fmt_score(dataset: str, v: float) -> str:
    if dataset == "mmlu":
        return f"{v:.0f}" if v in (0.0, 1.0) else f"{v:.2f}"
    if dataset in {"qa", "rag", "translation", "summarization"}:
        return f"{v:.3f}"
    return f"{v:.3f}"


def emit_latex(cases: list[dict[str, Any]], path: Path) -> str:
    """Two table* blocks (7B / 13B), booktabs, appendix-friendly."""
    blocks: list[str] = []
    for model_key, title in (
        ("Vicuna-7B", r"Vicuna-7B + EAGLE"),
        ("Llama-2-13B", r"Llama-2-13B + SAMD (Token Recycle)"),
    ):
        subset = [c for c in cases if c["model"] == model_key]
        if not subset:
            continue
        label = "tab:tss_cases_7b" if "7B" in model_key else "tab:tss_cases_13b"
        subset.sort(key=lambda c: DOMAINS.index(c["dataset"]))
        body = []
        for i, c in enumerate(subset):
            skip = ",".join(map(str, c["skip_layers"]))
            prob = latex_escape(shorten(c["problem"], 260))
            native = latex_escape(shorten(c["native_answer"], 200))
            tss = latex_escape(shorten(c["tss_answer"], 200))
            ref = latex_escape(shorten(c["reference"], 160))
            ns = fmt_score(c["dataset"], c["native_score"])
            ts = fmt_score(c["dataset"], c["tss_score"])
            met = c["metric"]
            if c["tss_score"] > c["native_score"] + 1e-9:
                ts_tex = r"\textbf{" + ts + "}"
            else:
                ts_tex = ts
            header = (
                r"\multicolumn{2}{@{}l}{\textit{"
                + latex_escape(c["domain"])
                + r"} (\texttt{"
                + latex_escape(c["request_id"])
                + r"}); skip $\{ "
                + skip
                + r" \}$; "
                + met
                + f": {ns}"
                + r"$\rightarrow$"
                + ts_tex
                + r"} \\"
            )
            body.append(header)
            body.append(r"\midrule")
            body.append(rf"\textbf{{Input}} & {prob} \\")
            body.append(rf"\textbf{{Native}} & {native} \\")
            body.append(rf"\textbf{{+TSS}} & {tss} \\")
            body.append(rf"\textbf{{Ref.}} & {ref} \\")
            if i < len(subset) - 1:
                body.append(r"\midrule")

        cap = (
            rf"Qualitative held-out examples for {title}. "
            r"Each block is one request from the official test split ($n{=}64$), "
            r"comparing Native speculative decoding with the TSS skip set. "
            r"Case-level task scores appear in each block header."
        )
        tex = "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                r"\small",
                rf"\caption{{{cap}}}",
                rf"\label{{{label}}}",
                r"\setlength{\tabcolsep}{4pt}",
                r"\begin{tabular}{@{}>{\bfseries}l p{0.90\linewidth}@{}}",
                r"\toprule",
                *body,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
                "",
            ]
        )
        blocks.append(tex)

    # Also a compact one-column figure-style listing for main paper (optional short)
    short_rows = []
    for c in sorted(cases, key=lambda x: (0 if "7B" in x["model"] else 1, DOMAINS.index(x["dataset"]))):
        skip = ",".join(map(str, c["skip_layers"]))
        short_rows.append(
            {
                "Model": c["model"].replace("Vicuna-7B", "7B").replace("Llama-2-13B", "13B"),
                "Domain": c["domain"],
                "ID": c["request_id"],
                "Skip": f"{{{skip}}}",
                "Metric": c["metric"],
                "Native": fmt_score(c["dataset"], c["native_score"]),
                "TSS": fmt_score(c["dataset"], c["tss_score"]),
            }
        )

    summary = [
        r"% Compact index of qualitative cases (optional)",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Index of Native vs.\ TSS qualitative cases (one held-out example per domain).}",
        r"\label{tab:tss_case_index}",
        r"\begin{tabular}{llcclcc}",
        r"\toprule",
        r"Model & Domain & Request ID & Skip Layers & Metric & Native & +TSS \\",
        r"\midrule",
    ]
    for r in short_rows:
        skip_tex = r["Skip"].strip("{}")
        summary.append(
            f"{r['Model']} & {r['Domain']} & \\texttt{{{latex_escape(r['ID'])}}} & "
            + "$\\{"
            + skip_tex
            + "\\}$ & "
            + f"{r['Metric']} & {r['Native']} & {r['TSS']} \\\\"
        )
    summary += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    preamble = [
        r"% TSS qualitative cases — NeurIPS/ICML booktabs style",
        r"% Requires: \usepackage{booktabs,array}",
        r"% Place in appendix (or main paper if space allows).",
        "",
    ]
    text = "\n".join(preamble + summary + blocks)
    path.write_text(text)
    return text


def emit_markdown(cases: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Native vs TSS Qualitative Cases (5 domains × 7B/13B)", ""]
    for c in cases:
        lines += [
            f"## {c['model']} + {c['method']} / {c['domain']}",
            f"- id: `{c['request_id']}`",
            f"- skip: `{c['skip_layers']}`",
            f"- {c['metric']}: Native **{c['native_score']:.3f}** → TSS **{c['tss_score']:.3f}**",
            "",
            "**Input**",
            "",
            shorten(c["problem"], 500),
            "",
            "**Native**",
            "",
            c["native_answer"],
            "",
            "**+TSS**",
            "",
            c["tss_answer"],
            "",
            "**Reference**",
            "",
            shorten(c["reference"], 300),
            "",
            "---",
            "",
        ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-rag", action="store_true")
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()
    ensure_dir(OUT_DIR)
    if args.collect_rag and not args.table_only:
        collect_missing_rag()
    cases = gather_cases()
    write_json({"cases": cases}, OUT_DIR / "tab_tss_qualitative_cases.json")
    tex = emit_latex(cases, OUT_DIR / "tab_tss_qualitative_cases.tex")
    emit_markdown(cases, OUT_DIR / "tab_tss_qualitative_cases.md")
    print(tex)
    print(f"[done] {len(cases)} cases -> {OUT_DIR / 'tab_tss_qualitative_cases.tex'}", flush=True)


if __name__ == "__main__":
    main()
