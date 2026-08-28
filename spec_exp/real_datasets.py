"""Real-dataset prompt selection for ctx-bucketed speculative decoding experiments."""
from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_DATA = Path(__file__).resolve().parents[1] / "data"
SPEC_BENCH = REPO_DATA / "spec_bench" / "question.jsonl"
SPEC_BENCH_FALLBACK = REPO_DATA / "Spec-Bench-repo" / "data" / "spec_bench" / "question.jsonl"
LONGBENCH_DIR = REPO_DATA / "longbench"

# Each ctx bucket maps to preferred dataset sources (first available wins).
CTX_BUCKET_SOURCES: dict[int, list[dict[str, str]]] = {
    1024: [
        {"kind": "specbench", "category": "rag"},
        {"kind": "specbench", "category": "summarization"},
        {"kind": "specbench", "category": "qa"},
    ],
    2048: [
        {"kind": "specbench", "category": "summarization"},
        {"kind": "specbench", "category": "rag"},
        {"kind": "longbench", "file": "qasper.jsonl"},
    ],
    4096: [
        {"kind": "longbench", "file": "gov_report.jsonl"},
        {"kind": "specbench", "category": "summarization", "repeat": "3"},
    ],
    8192: [
        {"kind": "longbench", "file": "gov_report.jsonl"},
        {"kind": "longbench", "file": "multifieldqa_en.jsonl"},
        {"kind": "specbench", "category": "summarization", "repeat": "6"},
    ],
    16384: [
        {"kind": "longbench", "file": "gov_report.jsonl", "repeat": "2"},
        {"kind": "specbench", "category": "summarization", "repeat": "12"},
    ],
}


@dataclass(frozen=True)
class RealDataItem:
    request_id: str
    prompt: str
    category: str
    dataset: str
    target_ctx: int
    prompt_tokens: int
    max_tokens: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _specbench_path() -> Path:
    if SPEC_BENCH.is_file():
        return SPEC_BENCH
    if SPEC_BENCH_FALLBACK.is_file():
        return SPEC_BENCH_FALLBACK
    raise FileNotFoundError("SpecBench question.jsonl not found; run run_qwen3_specbench_suite.sh first")


def load_specbench_rows(category: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(_read_jsonl(_specbench_path())):
        cat = row.get("category", "unknown")
        if category and cat != category:
            continue
        turns = row.get("turns") or []
        if not turns:
            continue
        prompt = turns[0] if isinstance(turns[0], str) else str(turns[0])
        rows.append({"id": f"specbench-{idx:05d}", "category": cat, "prompt": prompt})
    return rows


def load_longbench_rows(filename: str) -> list[dict[str, Any]]:
    path = LONGBENCH_DIR / filename
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(_read_jsonl(path)):
        ctx = row.get("context") or ""
        inp = row.get("input") or row.get("question") or ""
        prompt = f"{ctx}\n\n{inp}".strip() if inp else ctx
        if not prompt:
            continue
        rows.append(
            {
                "id": f"longbench-{filename}-{idx:05d}",
                "category": filename.replace(".jsonl", ""),
                "prompt": prompt,
            }
        )
    return rows


def _token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def _fit_to_ctx(tokenizer: Any, text: str, target_ctx: int) -> tuple[str, int]:
    """Truncate or pad (repeat tail) to land near target_ctx tokens."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) >= target_ctx:
        trimmed = ids[:target_ctx]
        return tokenizer.decode(trimmed, skip_special_tokens=True), target_ctx
    # Pad by repeating a suffix chunk from the same real text (not random noise).
    if not ids:
        pad_text = " " * 80
        ids = tokenizer.encode(pad_text, add_special_tokens=False)
    chunk = ids[max(0, len(ids) - min(256, len(ids))):]
    out = list(ids)
    while len(out) < target_ctx:
        need = target_ctx - len(out)
        out.extend(chunk[:need])
    out = out[:target_ctx]
    return tokenizer.decode(out, skip_special_tokens=True), target_ctx


def _pick_median_prompt(candidates: list[dict[str, Any]], tokenizer: Any, target_ctx: int) -> dict[str, Any] | None:
    if not candidates:
        return None
    scored = [(abs(_token_len(tokenizer, c["prompt"]) - target_ctx), c) for c in candidates]
    scored.sort(key=lambda x: x[0])
    # Prefer samples whose raw length is within 2x of target; else take closest.
    pool = [c for d, c in scored if d <= target_ctx] or [scored[0][1]]
    mid = pool[len(pool) // 2]
    return mid


def _materialize_source(tokenizer: Any, source: dict[str, str], target_ctx: int) -> list[dict[str, Any]]:
    kind = source["kind"]
    repeat = int(source.get("repeat", "1"))
    if kind == "specbench":
        rows = load_specbench_rows(source.get("category"))
    elif kind == "longbench":
        rows = load_longbench_rows(source["file"])
    else:
        return []
    if repeat > 1 and rows:
        expanded: list[dict[str, Any]] = []
        for r in rows:
            expanded.append(
                {
                    **r,
                    "prompt": (r["prompt"] + "\n\n") * repeat,
                    "id": f"{r['id']}_x{repeat}",
                }
            )
        rows = expanded
    out: list[dict[str, Any]] = []
    for r in rows:
        prompt, ntok = _fit_to_ctx(tokenizer, r["prompt"], target_ctx)
        out.append({**r, "prompt": prompt, "prompt_tokens": ntok, "target_ctx": target_ctx})
    return out


def build_ctx_manifest(tokenizer: Any, ctx_buckets: list[int] | None = None) -> list[dict[str, Any]]:
    """One representative median prompt per ctx bucket with dataset provenance."""
    buckets = ctx_buckets or sorted(CTX_BUCKET_SOURCES)
    manifest: list[dict[str, Any]] = []
    for ctx in buckets:
        chosen = None
        source_used = None
        for src in CTX_BUCKET_SOURCES.get(ctx, []):
            items = _materialize_source(tokenizer, src, ctx)
            pick = _pick_median_prompt(items, tokenizer, ctx)
            if pick:
                chosen = pick
                source_used = src
                break
        if not chosen:
            continue
        manifest.append(
            {
                "target_ctx": ctx,
                "dataset": chosen.get("category", "?"),
                "source": source_used,
                "request_id": chosen["id"],
                "prompt_tokens": chosen["prompt_tokens"],
                "prompt_chars": len(chosen["prompt"]),
            }
        )
    return manifest


def load_ctx_bucket_items(
    tokenizer: Any,
    target_ctx: int,
    *,
    num_requests: int = 16,
    output_len: int = 64,
    seed: int = 0,
    category: str | None = None,
) -> list[RealDataItem]:
    """Load real prompts for a ctx bucket; pick num_requests samples nearest median length."""
    sources = CTX_BUCKET_SOURCES.get(target_ctx, [{"kind": "specbench", "category": category or "qa"}])
    pool: list[dict[str, Any]] = []
    for src in sources:
        if category and src.get("kind") == "specbench":
            src = {**src, "category": category}
        pool.extend(_materialize_source(tokenizer, src, target_ctx))
        if pool:
            break
    if not pool:
        raise RuntimeError(f"No prompts for ctx={target_ctx}")

    lengths = [p["prompt_tokens"] for p in pool]
    median_len = int(statistics.median(lengths))
    pool.sort(key=lambda p: abs(p["prompt_tokens"] - median_len))
    rng = random.Random(seed)
    if len(pool) > num_requests:
        head = pool[: max(num_requests * 2, num_requests)]
        rng.shuffle(head)
        pool = head[:num_requests]
    else:
        pool = pool[:num_requests]

    items: list[RealDataItem] = []
    for p in pool:
        items.append(
            RealDataItem(
                request_id=p["id"],
                prompt=p["prompt"],
                category=p.get("category", "?"),
                dataset=str(p.get("source", sources[0]).get("kind", "?")),
                target_ctx=target_ctx,
                prompt_tokens=int(p["prompt_tokens"]),
                max_tokens=output_len,
            )
        )
    return items


def load_specbench_by_category(
    tokenizer: Any,
    category: str,
    *,
    num_requests: int = 16,
    output_len: int = 256,
    seed: int = 0,
) -> list[RealDataItem]:
    rows = load_specbench_rows(category)
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:num_requests]
    items: list[RealDataItem] = []
    for r in rows:
        ntok = _token_len(tokenizer, r["prompt"])
        items.append(
            RealDataItem(
                request_id=r["id"],
                prompt=r["prompt"],
                category=r["category"],
                dataset="specbench",
                target_ctx=ntok,
                prompt_tokens=ntok,
                max_tokens=output_len,
            )
        )
    return items
