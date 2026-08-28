from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpecBenchItem:
    request_id: str
    prompt: str
    category: str
    max_tokens: int
    reference: str | None = None


def load_specbench(
    dataset_path: str | Path,
    *,
    output_len: int = 256,
    category: str | None = None,
    num_requests: int | None = None,
    seed: int = 0,
    shuffle: bool = True,
) -> list[SpecBenchItem]:
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"SpecBench dataset not found: {path}")

    rows: list[SpecBenchItem] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            cat = row.get("category", "unknown")
            if category and cat != category:
                continue
            turns = row.get("turns")
            if not turns:
                raise ValueError(f"Line {idx} missing 'turns'")
            prompt = turns[0] if isinstance(turns[0], str) else str(turns[0])
            refs = row.get("reference")
            reference = None
            if isinstance(refs, list) and refs:
                reference = refs[0] if isinstance(refs[0], str) else str(refs[0])
            elif isinstance(refs, str):
                reference = refs
            rows.append(
                SpecBenchItem(
                    request_id=f"specbench-{idx:05d}",
                    prompt=prompt,
                    category=cat,
                    max_tokens=output_len,
                    reference=reference,
                )
            )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if num_requests is not None:
        rows = rows[:num_requests]
    return rows
