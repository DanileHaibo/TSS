"""Benchmark constants: official K, dataset metadata, scoring categories."""
from __future__ import annotations

# Spec-Bench Table-2 style K (no chat / mt_bench).
OFFICIAL_K: dict[str, int] = {
    "gsm8k": 2,
    "humaneval": 4,
    "translation": 2,
    "summarization": 2,
    "qa": 1,
    "mmlu": 1,
    "rag": 2,
}

# Active evaluation suite (instruction-following / mt_bench excluded).
OFFICIAL_DATASETS: tuple[str, ...] = (
    "gsm8k",
    "humaneval",
    "translation",
    "summarization",
    "qa",
    "mmlu",
    "rag",
)

DATASET_LABELS: dict[str, str] = {
    "gsm8k": "Math / GSM8K",
    "humaneval": "Code / HumanEval",
    "translation": "Translation",
    "summarization": "Summarization",
    "qa": "QA / NaturalQuestions",
    "mmlu": "Knowledge / MMLU",
    "rag": "RAG",
}

# Maps dataset name -> task_score category string.
SCORE_CATEGORY: dict[str, str] = {
    "gsm8k": "math_reasoning",
    "humaneval": "code",
    "translation": "translation",
    "summarization": "summarization",
    "qa": "qa",
    "mmlu": "multiple_choice",
    "rag": "rag",
}

# Primary metric name per dataset (for reporting).
PRIMARY_METRIC: dict[str, str] = {
    "gsm8k": "accuracy",
    "humaneval": "pass@1",
    "translation": "bleu",
    "summarization": "rougeL",
    "qa": "f1",
    "mmlu": "accuracy",
    "rag": "f1",
}

FIXED_SKIP_LAYERS: dict[str, list[int]] = {
    "gsm8k": [3, 6, 9, 12, 15, 18, 21, 24],
    "humaneval": [2, 5, 8, 11, 14, 17, 20, 23, 26],
    "translation": [3, 6, 9, 12, 15, 18, 21, 24, 27],
    "summarization": [3, 6, 9, 12, 15, 18, 21, 24, 27],
    "qa": [4, 7, 10, 13, 16, 19, 22, 25],
    "rag": [3, 6, 9, 12, 15, 18, 21, 24, 27],
}

NUM_TARGET_LAYERS = 28

TARGET_MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
DRAFT_MODEL = "/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct"
VLLM_DRAFT_MODEL = "/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct-vllm-draft"

QWEN3_TARGET = "/root/autodl-tmp/models/Qwen3-8B"
QWEN3_DRAFT = "/root/autodl-tmp/models/Qwen3-0.6B"

LLAMA_BASE_MODEL = "/root/autodl-tmp/modelscope/LLM-Research/Meta-Llama-3___1-8B-Instruct"
LLAMA_EAGLE3_DRAFT = (
    "/root/autodl-tmp/hf/models--lmsys--sglang-EAGLE3-LLaMA3.1-Instruct-8B/snapshots/"
    "28a53ce8911434c031d7c78392abb26d898ec293"
)
