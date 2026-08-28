"""Resolve dataset directories without hard-coding machine-specific paths.

Lookup order:
  1. ``TSS_DATA_DIR`` / ``EAGLE_DATA_DIR`` environment variables
  2. Repo-local ``data/`` produced by ``scripts/download_and_preprocess_datasets.py``
  3. Legacy AutoDL paths used in the original experiments
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_LEGACY_EAGLE = Path("/root/autodl-tmp/eagle3-system-exp/repos/EAGLE/eagle/data")
_LEGACY_SPEC = (
    Path("/root/autodl-tmp/specdecode-system-exp/data/Spec-Bench-repo/data/spec_bench/question.jsonl"),
    Path("/root/autodl-tmp/specdecode-system-exp/data/spec_bench/question.jsonl"),
)
_LEGACY_NQ = Path("/root/autodl-tmp/specdecode-system-exp/data/nq_open/validation.jsonl")


def data_root() -> Path:
    env = os.environ.get("TSS_DATA_DIR")
    if env:
        return Path(env)
    return REPO_ROOT / "data"


def spec_bench_question_jsonl() -> Path:
    candidates = [
        data_root() / "spec_bench" / "question.jsonl",
        data_root() / "Spec-Bench-repo" / "data" / "spec_bench" / "question.jsonl",
        REPO_ROOT / "data" / "spec_bench" / "question.jsonl",
        *_LEGACY_SPEC,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def eagle_data_dir() -> Path:
    env = os.environ.get("EAGLE_DATA_DIR")
    if env:
        return Path(env)
    local = data_root() / "eagle"
    if (local / "gsm8k" / "question.jsonl").is_file():
        return local
    if _LEGACY_EAGLE.is_dir():
        return _LEGACY_EAGLE
    return local


def nq_open_jsonl() -> Path:
    candidates = [
        data_root() / "nq_open" / "validation.jsonl",
        _LEGACY_NQ,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def mmlu_jsonl() -> Path:
    return data_root() / "mmlu" / "test.jsonl"
