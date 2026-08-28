from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadItem:
    request_id: str
    prompt: str
    max_tokens: int
    workload: str


SHORT_PROMPTS = [
    "Write a concise definition of speculative decoding.",
    "List three benefits of batching in LLM serving.",
    "Answer in one paragraph: why is GPU memory bandwidth important?",
    "Give a short example of a Python context manager.",
]

LONG_CONTEXT_SEED = (
    "Speculative decoding uses a small draft model to propose multiple future "
    "tokens and a larger target model to verify those tokens in fewer decoding "
    "rounds. In serving systems, the benefit depends on acceptance rate, draft "
    "cost, verify cost, batching behavior, KV cache pressure, and scheduler "
    "overheads. The experiment should separate algorithmic acceptance effects "
    "from systems effects such as memory traffic and launch overhead. "
)

MEDIUM_PROMPTS = [
    "You are writing an engineering report about GPU inference systems. "
    "Explain the experimental setup, key metrics, bottlenecks, and future work.",
    "Draft a detailed technical note comparing autoregressive decoding and "
    "speculative decoding for online LLM serving.",
    "Write a structured analysis of how heterogeneous request behavior affects "
    "batched decoding throughput.",
]


def make_workload(kind: str, num_requests: int, seed: int = 0) -> list[WorkloadItem]:
    rng = random.Random(seed)
    items: list[WorkloadItem] = []
    for idx in range(num_requests):
        request_id = f"{kind}-{idx:05d}"
        if kind == "short_prompt_short_output":
            prompt = rng.choice(SHORT_PROMPTS)
            max_tokens = 32
        elif kind == "long_prompt_short_output":
            repeat = rng.randint(10, 14)
            prompt = (
                "Read the following system note and summarize the main risk in two sentences.\n\n"
                + LONG_CONTEXT_SEED * repeat
                + "\n\nSummary:"
            )
            max_tokens = 48
        elif kind == "medium_prompt_long_output":
            prompt = rng.choice(MEDIUM_PROMPTS)
            prompt += "\nUse concrete examples and keep the discussion technical."
            max_tokens = 192
        else:
            raise ValueError(
                f"Unknown workload {kind!r}. Expected one of: "
                "short_prompt_short_output, long_prompt_short_output, medium_prompt_long_output"
            )
        items.append(WorkloadItem(request_id=request_id, prompt=prompt, max_tokens=max_tokens, workload=kind))
    return items


def batched(items: list[WorkloadItem], batch_size: int) -> list[list[WorkloadItem]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

