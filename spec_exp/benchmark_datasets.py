"""Load benchmark datasets into unified DecodeItem list."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from spec_exp.data_paths import eagle_data_dir, mmlu_jsonl, nq_open_jsonl, spec_bench_question_jsonl
from spec_exp.self_spec_decode import DecodeItem

EAGLE_DATA = eagle_data_dir()
SPEC_BENCH_DATA = spec_bench_question_jsonl()


def _qwen_chat_prompt(user_text: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.\n"
        f"<|im_start|>user\n{user_text}\n"
        "<|im_start|>assistant\n"
    )


def _ref_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        return raw[0] if len(raw) == 1 else str(raw)
    return str(raw)


def _humaneval_official_task_id(prompt: str) -> str | None:
    import re

    m = re.search(r"def\s+(\w+)\s*\(", prompt)
    if not m:
        return None
    try:
        from human_eval.data import read_problems
    except ImportError:
        return None
    entry = m.group(1)
    for tid, prob in read_problems().items():
        if prob["entry_point"] == entry:
            return tid
    return None


def load_spec_bench_category(
    category: str,
    *,
    num_requests: int | None,
    seed: int,
    output_len: int,
    use_chat_template: bool = True,
) -> list[DecodeItem]:
    path = spec_bench_question_jsonl()
    if not path.is_file():
        raise FileNotFoundError(
            f"Spec-Bench question.jsonl not found at {path}. "
            "Run: python scripts/download_and_preprocess_datasets.py"
        )
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("category") == category:
            rows.append(r)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_requests:
        rows = rows[:num_requests]
    items: list[DecodeItem] = []
    for r in rows:
        user = r["turns"][0]
        prompt = _qwen_chat_prompt(user) if use_chat_template else user
        ref = _ref_text(r.get("reference"))
        items.append(
            DecodeItem(
                request_id=f"sb_{category}_{r['question_id']}",
                prompt=prompt,
                max_tokens=output_len,
                category=category,
                reference=ref or None,
            )
        )
    return items


def load_gsm8k(*, num_requests: int | None, seed: int, output_len: int) -> list[DecodeItem]:
    path = eagle_data_dir() / "gsm8k" / "question.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_requests:
        rows = rows[:num_requests]
    items: list[DecodeItem] = []
    for r in rows:
        ref = _ref_text(r.get("reference"))
        items.append(
            DecodeItem(
                request_id=f"gsm_{r['question_id']}",
                prompt=_qwen_chat_prompt(r["turns"][0]),
                max_tokens=output_len,
                category="math_reasoning",
                reference=ref or None,
            )
        )
    return items


def load_humaneval(*, num_requests: int | None, seed: int, output_len: int) -> list[DecodeItem]:
    path = eagle_data_dir() / "humaneval" / "question.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_requests:
        rows = rows[:num_requests]
    items: list[DecodeItem] = []
    for r in rows:
        ref = _ref_text(r.get("reference"))
        prompt = r["turns"][0]
        official_tid = _humaneval_official_task_id(prompt)
        rid = official_tid if official_tid else f"he_{r['question_id']}"
        items.append(
            DecodeItem(
                request_id=rid,
                prompt=prompt,
                max_tokens=min(output_len, 256),
                category="code",
                reference=ref or None,
            )
        )
    return items


def load_translation(*, num_requests: int | None, seed: int, output_len: int) -> list[DecodeItem]:
    return load_spec_bench_category(
        "translation", num_requests=num_requests, seed=seed, output_len=min(output_len, 128)
    )


def load_summarization(*, num_requests: int | None, seed: int, output_len: int) -> list[DecodeItem]:
    path = eagle_data_dir() / "sum" / "question.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_requests:
        rows = rows[:num_requests]
    items: list[DecodeItem] = []
    for r in rows:
        ref = _ref_text(r.get("reference"))
        user = r["turns"][0]
        if not user.lower().startswith("summarize"):
            user = f"Summarize: {user}"
        items.append(
            DecodeItem(
                request_id=f"sum_{r['question_id']}",
                prompt=_qwen_chat_prompt(user),
                max_tokens=output_len,
                category="summarization",
                reference=ref or None,
            )
        )
    return items


def load_qa(*, num_requests: int | None, seed: int, output_len: int) -> list[DecodeItem]:
    import os

    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))

    rows: list[dict] = []
    cache = nq_open_jsonl()
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:
        from datasets import load_dataset

        ds = load_dataset("nq_open", split="validation", trust_remote_code=True)
        for ex in ds:
            q = ex.get("question") or ex.get("question_text", "")
            ans = ex.get("answer") or (ex.get("answers", [""])[0] if ex.get("answers") else "")
            if isinstance(ans, list):
                ans = ans[0] if ans else ""
            rows.append({"question": q, "reference": str(ans)})
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("\n".join(json.dumps(r) for r in rows[:5000]) + "\n", encoding="utf-8")

    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_requests:
        rows = rows[:num_requests]
    return [
        DecodeItem(
            request_id=f"nq_{i}",
            prompt=_qwen_chat_prompt(f"Answer the question concisely.\n\nQuestion: {r['question']}"),
            max_tokens=min(output_len, 64),
            category="qa",
            reference=r.get("reference"),
        )
        for i, r in enumerate(rows)
    ]


def load_mmlu(*, num_requests: int | None, seed: int, output_len: int) -> list[DecodeItem]:
    import os

    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))

    cache = mmlu_jsonl()
    if cache.is_file():
        rows = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
    else:
        from datasets import load_dataset

        rows = list(load_dataset("cais/mmlu", "all", split="test"))
    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_requests:
        rows = rows[:num_requests]

    labels = "ABCD"
    items: list[DecodeItem] = []
    for i, row in enumerate(rows):
        choices = row["choices"]
        options = "\n".join(f"{labels[j]}. {choice}" for j, choice in enumerate(choices))
        user = (
            "Answer the multiple-choice question. Respond with only the letter "
            "A, B, C, or D.\n\n"
            f"Question: {row['question']}\n{options}\nAnswer:"
        )
        items.append(
            DecodeItem(
                request_id=f"mmlu_{row['subject']}_{i}",
                prompt=_qwen_chat_prompt(user),
                max_tokens=min(output_len, 32),
                category="multiple_choice",
                reference=labels[int(row["answer"])],
            )
        )
    return items


def load_rag(*, num_requests: int | None, seed: int, output_len: int) -> list[DecodeItem]:
    return load_spec_bench_category(
        "rag", num_requests=num_requests, seed=seed, output_len=output_len
    )


LOADERS: dict[str, Any] = {
    "gsm8k": load_gsm8k,
    "humaneval": load_humaneval,
    "translation": load_translation,
    "summarization": load_summarization,
    "qa": load_qa,
    "mmlu": load_mmlu,
    "rag": load_rag,
    # backward-compat aliases
    "natural_questions": load_qa,
    "cnn_dailymail": load_summarization,
}


def load_dataset_items(
    dataset: str,
    *,
    num_requests: int | None = None,
    seed: int = 42,
    output_len: int = 128,
    prompt_style: str = "qwen",
) -> list[DecodeItem]:
    if dataset not in LOADERS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {sorted(LOADERS)}")
    return LOADERS[dataset](num_requests=num_requests, seed=seed, output_len=output_len)


def load_dataset_split(
    dataset: str,
    *,
    split: str,
    train_size: int = 8,
    seed: int = 42,
    output_len: int = 96,
) -> list[DecodeItem]:
    """Split shuffled dataset: first train_size for skip search, rest for held-out eval."""
    all_items = load_dataset_items(dataset, num_requests=None, seed=seed, output_len=output_len)
    if split == "train":
        return all_items[:train_size]
    if split == "test":
        return all_items[train_size:]
    if split == "all":
        return all_items
    raise ValueError(f"split must be train|test|all, got {split!r}")
