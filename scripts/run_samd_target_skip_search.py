#!/usr/bin/env python3
"""Run model-free SAMD[Token-Recycle] skip search on compatible Llama targets."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spec_exp.benchmark_config import SCORE_CATEGORY
from spec_exp.benchmark_datasets import load_dataset_items
from spec_exp.io import ensure_dir, write_json
from spec_exp.self_spec_decode import DecodeItem
from spec_exp.transformers_compat import install_transformers_compat

TARGETS = {
    "llama2_13b": {
        "path": "/root/autodl-tmp/models/Llama-2-13b-chat-hf",
        "num_layers": 40,
        "size": "13b",
    },
    "llama31_8b": {
        "path": "/root/autodl-tmp/models/Llama-3.1-8B-Instruct",
        "num_layers": 32,
        "size": "8b",
    },
}


def extract_user(prompt: str) -> str:
    if "<|im_start|>user\n" in prompt:
        return prompt.split("<|im_start|>user\n", 1)[1].split(
            "\n<|im_start|>assistant", 1
        )[0]
    return prompt


def render_prompt(tokenizer, user: str, target: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    if target == "llama2_13b":
        return f"<s>[INST] {user.strip()} [/INST] "
    raise ValueError(f"No chat template available for {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-len", type=int, default=96)
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--max-skip-layers", type=int, default=12)
    parser.add_argument("--min-skips", type=int, default=6)
    parser.add_argument("--score-drop-tol", type=float, default=0.20)
    parser.add_argument("--accept-drop-tol", type=float, default=0.08)
    parser.add_argument(
        "--search-mode",
        choices=["max_skip_latter", "score_first"],
        default="max_skip_latter",
    )
    args = parser.parse_args()
    install_transformers_compat()
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    cfg = TARGETS[args.target]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    raw = load_dataset_items(
        args.dataset,
        num_requests=80,
        seed=42,
        output_len=args.output_len,
    )[: args.train_size]
    items = [
        DecodeItem(
            request_id=item.request_id,
            prompt=render_prompt(tokenizer, extract_user(item.prompt), args.target),
            max_tokens=item.max_tokens,
            category=item.category,
            reference=item.reference,
        )
        for item in raw
    ]
    from scripts.run_hydra_samd_skip_greedy import greedy_search, load_samd_model

    model = load_samd_model(
        cfg["path"],
        eagle_path=None,
        size=cfg["size"],
        tree_method="token_recycle",
        use_safetensors=False if args.target == "llama2_13b" else None,
        token_tree_path=(
            "token_recycle_4_15.json" if args.target == "llama2_13b" else None
        ),
    )
    try:
        result = greedy_search(
            method="samd",
            model=model,
            items=items,
            domain=SCORE_CATEGORY[args.dataset],
            num_layers=cfg["num_layers"],
            layer_step=1,
            layer_step_override=1,
            exhaustive_singles=True,
            score_drop_tol=args.score_drop_tol,
            score_tol_mode="relative",
            accept_mode="baseline",
            accept_drop_tol=args.accept_drop_tol,
            pick_metric="tok_per_s",
            max_skip_layers=args.max_skip_layers,
            min_skips=args.min_skips,
            max_rounds=args.max_skip_layers,
            search_mode=args.search_mode,
            early_barrier=2,
            latter_barrier=0,
        )
    finally:
        del model
        import torch

        torch.cuda.empty_cache()
    result.update(
        {
            "target": args.target,
            "target_path": cfg["path"],
            "draft_method": "samd_token_recycle",
            "dataset": args.dataset,
            "train_size": len(items),
            "min_skips": args.min_skips,
            "search_mode": args.search_mode,
        }
    )
    output_dir = ensure_dir(args.output_dir)
    output = output_dir / f"{args.dataset}_{args.target}_samd_token_recycle.json"
    write_json(result, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "skip_layers": result.get("best", {}).get("skip_layers"),
                "baseline": result.get("baseline"),
                "best": result.get("best"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
