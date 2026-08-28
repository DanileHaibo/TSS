#!/usr/bin/env python3
"""Download and preprocess paper datasets into ``data/`` (no model weights).

Paper domains (Vicuna-7B / Llama-2-13B):
  translation, summarization, qa, rag, mmlu  (+ optional gsm8k / humaneval)

Usage:
  python scripts/download_and_preprocess_datasets.py
  python scripts/download_and_preprocess_datasets.py --skip-eagle
  TSS_DATA_DIR=/path/to/data python scripts/download_and_preprocess_datasets.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spec_exp.data_paths import data_root  # noqa: E402

SPEC_BENCH_GIT = "https://github.com/hemingkx/Spec-Bench.git"
EAGLE_GIT = "https://github.com/SafeAILab/EAGLE.git"
EAGLE_SUBSETS = ("gsm8k", "humaneval", "sum")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[write] {path}  ({len(rows)} rows)", flush=True)


def download_spec_bench(root: Path, *, force: bool) -> Path:
    out = root / "spec_bench" / "question.jsonl"
    clone = root / "Spec-Bench-repo"
    if out.is_file() and not force:
        print(f"[skip] Spec-Bench already at {out}", flush=True)
        return out
    if not (clone / "data" / "spec_bench" / "question.jsonl").is_file():
        if clone.exists() and force:
            shutil.rmtree(clone)
        if not clone.exists():
            run(["git", "clone", "--depth", "1", SPEC_BENCH_GIT, str(clone)])
    src = clone / "data" / "spec_bench" / "question.jsonl"
    if not src.is_file():
        raise FileNotFoundError(f"Spec-Bench jsonl missing after clone: {src}")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    cats: dict[str, int] = {}
    for line in out.read_text().splitlines():
        if not line.strip():
            continue
        cat = json.loads(line).get("category", "?")
        cats[cat] = cats.get(cat, 0) + 1
    print(f"[spec-bench] copied {out} categories={cats}", flush=True)
    return out


def download_eagle_jsonl(root: Path, *, force: bool) -> Path:
    dest = root / "eagle"
    clone = root / "EAGLE-repo"
    ready = all((dest / name / "question.jsonl").is_file() for name in EAGLE_SUBSETS)
    if ready and not force:
        print(f"[skip] EAGLE jsonl already under {dest}", flush=True)
        return dest
    if not (clone / "eagle" / "data" / "gsm8k" / "question.jsonl").is_file():
        if clone.exists() and force:
            shutil.rmtree(clone)
        if not clone.exists():
            run(["git", "clone", "--depth", "1", EAGLE_GIT, str(clone)])
    src_root = clone / "eagle" / "data"
    for name in EAGLE_SUBSETS:
        src = src_root / name / "question.jsonl"
        if not src.is_file():
            raise FileNotFoundError(f"EAGLE subset missing: {src}")
        dst = dest / name / "question.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n = sum(1 for line in dst.read_text().splitlines() if line.strip())
        print(f"[eagle] {name}: {n} rows -> {dst}", flush=True)
    return dest


def download_nq_open(root: Path, *, force: bool, max_rows: int) -> Path:
    out = root / "nq_open" / "validation.jsonl"
    if out.is_file() and not force:
        print(f"[skip] NQ-Open already at {out}", flush=True)
        return out
    from datasets import load_dataset

    print("[hf] load_dataset('nq_open', split='validation')", flush=True)
    ds = load_dataset("nq_open", split="validation", trust_remote_code=True)
    rows: list[dict] = []
    for ex in ds:
        q = ex.get("question") or ex.get("question_text", "")
        ans = ex.get("answer") or (ex.get("answers", [""])[0] if ex.get("answers") else "")
        if isinstance(ans, list):
            ans = ans[0] if ans else ""
        rows.append({"question": q, "reference": str(ans)})
        if len(rows) >= max_rows:
            break
    write_jsonl(out, rows)
    return out


def download_mmlu(root: Path, *, force: bool) -> Path:
    out = root / "mmlu" / "test.jsonl"
    if out.is_file() and not force:
        print(f"[skip] MMLU already at {out}", flush=True)
        return out
    from datasets import load_dataset

    print("[hf] load_dataset('cais/mmlu', 'all', split='test')", flush=True)
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = []
    for row in ds:
        rows.append(
            {
                "subject": row["subject"],
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": int(row["answer"]),
            }
        )
    write_jsonl(out, rows)
    return out


def write_manifest(root: Path) -> None:
    files = []
    for path in sorted(root.rglob("*.jsonl")):
        if ".git" in path.parts or "EAGLE-repo" in path.parts or "Spec-Bench-repo" in path.parts:
            continue
        n = sum(1 for line in path.read_text().splitlines() if line.strip())
        files.append({"path": str(path.relative_to(root)), "num_rows": n})
    payload = {
        "data_root": str(root),
        "split_protocol": {
            "seed": 42,
            "pool": "shuffle all items, take first 80, train=first 16, test=next 64",
            "exception": "Llama-2-13B QA selection_train_size=32",
        },
        "files": files,
    }
    path = root / "MANIFEST.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[manifest] {path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download + preprocess TSS paper datasets")
    p.add_argument("--data-dir", type=Path, default=None, help="Override TSS_DATA_DIR / data/")
    p.add_argument("--skip-spec-bench", action="store_true")
    p.add_argument("--skip-eagle", action="store_true")
    p.add_argument("--skip-nq", action="store_true")
    p.add_argument("--skip-mmlu", action="store_true")
    p.add_argument("--nq-max-rows", type=int, default=5000)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_dir:
        os.environ["TSS_DATA_DIR"] = str(args.data_dir.resolve())
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    print(f"[data-root] {root}", flush=True)

    if not args.skip_spec_bench:
        download_spec_bench(root, force=args.force)
    if not args.skip_eagle:
        download_eagle_jsonl(root, force=args.force)
    if not args.skip_nq:
        download_nq_open(root, force=args.force, max_rows=args.nq_max_rows)
    if not args.skip_mmlu:
        download_mmlu(root, force=args.force)
    write_manifest(root)
    print("[done] datasets ready. See DATASETS.md", flush=True)


if __name__ == "__main__":
    main()
