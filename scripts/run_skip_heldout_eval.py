#!/usr/bin/env python3
"""Evaluate fixed skip layers from skip-search runs on held-out test split (72 prompts)."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "data" / "Spec-Bench-repo"
sys.path.insert(0, str(SPEC))
sys.path.insert(0, str(REPO))

from spec_exp.transformers_compat import install_transformers_compat
from spec_exp.benchmark_config import SCORE_CATEGORY
from spec_exp.benchmark_datasets import load_dataset_split
from spec_exp.io import ensure_dir, write_json
from spec_exp.self_spec_decode import DecodeItem

from scripts.run_hydra_samd_skip_greedy import (  # noqa: E402
    EAGLE_PATHS,
    eval_hydra,
    eval_samd,
    load_hydra_model,
    load_samd_model,
    resolve_hydra_path,
    resolve_vicuna,
    resolve_vicuna13,
    resolve_vicuna33,
    resolve_vicuna7,
)
from scripts.run_specbench_nontree_4domains import DRAFTER68  # noqa: E402
from scripts.run_specbench_nontree_skip_greedy import MethodRunner, evaluate as evaluate_nontree  # noqa: E402
from scripts.run_vicuna13_eagle3_skip_sweep import (  # noqa: E402
    MODEL_PRESETS,
    eval_skip_config,
    load_items_for_preset,
    load_model as load_eagle3_model,
)


@dataclass
class HeldoutJob:
    tag: str
    kind: str  # hydra | samd | eagle3 | nontree
    dataset: str
    source_json: Path
    size: str = "7b"
    method: str = ""

    @property
    def out_name(self) -> str:
        return f"{self.tag}_heldout.json"


def parse_tree_hydra_samd(path: Path) -> HeldoutJob | None:
    m = re.match(
        r"^(?P<ds>[^_]+)_(?P<size>7b|13b|33b)_(?P<method>hydra|samd)_skip_greedy\.json$",
        path.name,
    )
    if not m:
        return None
    return HeldoutJob(
        tag=f"{m.group('ds')}_{m.group('size')}_{m.group('method')}",
        kind=m.group("method"),
        dataset=m.group("ds"),
        source_json=path,
        size=m.group("size"),
        method=m.group("method"),
    )


def parse_tree_eagle3(path: Path) -> HeldoutJob | None:
    m = re.match(r"^vicuna13_skip_max_accept_(?P<ds>.+)\.json$", path.name)
    if not m:
        return None
    return HeldoutJob(
        tag=f"{m.group('ds')}_13b_eagle3",
        kind="eagle3",
        dataset=m.group("ds"),
        source_json=path,
        size="13b",
        method="eagle3",
    )


def parse_nontree(path: Path) -> HeldoutJob | None:
    m = re.match(
        r"^(?P<ds>[^_]+)_(?:(?P<size>13b|33b)_)?(?P<method>sps|pld|lookahead)_skip_greedy\.json$",
        path.name,
    )
    if not m:
        return None
    size = m.group("size") or "7b"
    return HeldoutJob(
        tag=f"{m.group('ds')}_{size}_{m.group('method')}" if size != "7b" else f"{m.group('ds')}_{m.group('method')}",
        kind="nontree",
        dataset=m.group("ds"),
        source_json=path,
        size=size,
        method=m.group("method"),
    )


def discover_jobs(tree_dir: Path, nontree_dirs: list[tuple[str, Path]]) -> list[HeldoutJob]:
    jobs: list[HeldoutJob] = []
    for path in sorted(tree_dir.glob("*.json")):
        job = parse_tree_hydra_samd(path) or parse_tree_eagle3(path)
        if job:
            jobs.append(job)
    for _size, ndir in nontree_dirs:
        if not ndir.exists():
            continue
        for path in sorted(ndir.glob("*_skip_greedy.json")):
            job = parse_nontree(path)
            if job:
                jobs.append(job)
    return jobs


def load_search_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "best" not in data:
        raise ValueError(f"missing best in {path}")
    return data


def load_test_items_tree(dataset: str, *, seed: int, output_len: int, train_size: int) -> list[DecodeItem]:
    raw = load_dataset_split(dataset, split="test", train_size=train_size, seed=seed, output_len=output_len)
    wrapped: list[DecodeItem] = []
    for item in raw:
        user = item.prompt
        if "<|im_start|>user" in user:
            user = user.split("<|im_start|>user\n", 1)[1].split("\n<|im_start|>assistant")[0]
        from scripts.run_hydra_samd_skip_greedy import wrap_vicuna_prompt

        wrapped.append(
            DecodeItem(
                request_id=item.request_id,
                prompt=wrap_vicuna_prompt(user),
                max_tokens=item.max_tokens,
                category=item.category,
                reference=item.reference,
            )
        )
    return wrapped


def load_test_items_eagle3(dataset: str, *, seed: int, output_len: int, train_size: int) -> list[DecodeItem]:
    cfg = MODEL_PRESETS["vicuna13"]
    raw = load_dataset_split(dataset, split="test", train_size=train_size, seed=seed, output_len=output_len)
    from scripts.run_vicuna13_eagle3_skip_sweep import _extract_user_text, wrap_chat_prompt

    return [
        DecodeItem(
            request_id=item.request_id,
            prompt=wrap_chat_prompt(_extract_user_text(item.prompt), cfg["chat_template"]),
            max_tokens=item.max_tokens,
            category=item.category,
            reference=item.reference,
        )
        for item in raw
    ]


def load_test_items_nontree(dataset: str, *, seed: int, output_len: int, train_size: int) -> list[DecodeItem]:
    raw = load_dataset_split(dataset, split="test", train_size=train_size, seed=seed, output_len=output_len)
    from scripts.run_specbench_nontree_4domains import wrap_vicuna

    out: list[DecodeItem] = []
    for item in raw:
        user = item.prompt
        if "<|im_start|>user" in user:
            user = user.split("<|im_start|>user\n", 1)[1].split("\n<|im_start|>assistant")[0]
        out.append(
            DecodeItem(
                request_id=item.request_id,
                prompt=wrap_vicuna(user),
                max_tokens=item.max_tokens,
                category=item.category,
                reference=item.reference,
            )
        )
    return out


def metrics_delta(test_baseline: dict[str, Any], test_best: dict[str, Any]) -> dict[str, Any]:
    ba = test_baseline.get("mean_accepted_per_step")
    sa = test_best.get("mean_accepted_per_step")
    bs = test_baseline.get("task_score")
    ss = test_best.get("task_score")
    return {
        "delta_mean_accepted_per_step": (sa - ba) if ba is not None and sa is not None else math.nan,
        "delta_task_score": (ss - bs) if bs is not None and ss is not None else math.nan,
        "score_preserved": ss >= bs if bs is not None and ss is not None else None,
    }


def run_tree_job(
    job: HeldoutJob,
    *,
    model: Any,
    test_items: list[DecodeItem],
    skip_layers: set[int],
    domain: str,
    eval_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    baseline = eval_fn(model, test_items, set(), baseline_hypotheses=None, domain=domain)
    baseline_hyp = baseline.pop("hypotheses", None)
    best = eval_fn(
        model,
        test_items,
        skip_layers,
        baseline_hypotheses=baseline_hyp,
        domain=domain,
    )
    best.pop("hypotheses", None)
    return {"baseline": baseline, "best": best, "delta": metrics_delta(baseline, best)}


def run_eagle3_job(
    model: Any,
    test_items: list[DecodeItem],
    skip_layers: set[int],
    domain: str,
) -> dict[str, Any]:
    from scripts.run_vicuna13_eagle3_skip_sweep import _collect_hypotheses

    baseline = eval_skip_config(model, test_items, set(), baseline_hypotheses=None, domain=domain)
    baseline_hyp = _collect_hypotheses(model, test_items, set())
    best = eval_skip_config(
        model,
        test_items,
        skip_layers,
        baseline_hypotheses=baseline_hyp,
        domain=domain,
    )
    return {"baseline": baseline, "best": best, "delta": metrics_delta(baseline, best)}


def run_nontree_job(
    runner: MethodRunner,
    test_items: list[DecodeItem],
    skip_layers: set[int],
    domain: str,
) -> dict[str, Any]:
    baseline = evaluate_nontree(runner, test_items, domain, set())
    best = evaluate_nontree(runner, test_items, domain, skip_layers)
    baseline.pop("hypotheses", None)
    best.pop("hypotheses", None)
    return {"baseline": baseline, "best": best, "delta": metrics_delta(baseline, best)}


def build_payload(
    job: HeldoutJob,
    search: dict[str, Any],
    test_eval: dict[str, Any],
    *,
    train_size: int,
    test_size: int,
    seed: int,
    output_len: int,
) -> dict[str, Any]:
    train_baseline = search.get("baseline", {})
    train_best = search.get("best", {})
    skip_layers = train_best.get("skip_layers") or search.get("skip_layers") or []
    return {
        "tag": job.tag,
        "kind": job.kind,
        "method": job.method,
        "size": job.size,
        "dataset": job.dataset,
        "domain": SCORE_CATEGORY[job.dataset],
        "source_json": str(job.source_json),
        "split": {"seed": seed, "train_size": train_size, "test_size": test_size, "output_len": output_len},
        "skip_layers": skip_layers,
        "train_eval": {
            "baseline_mean_acc": train_baseline.get("mean_accepted_per_step"),
            "best_mean_acc": train_best.get("mean_accepted_per_step"),
            "baseline_task_score": train_baseline.get("task_score"),
            "best_task_score": train_best.get("task_score"),
        },
        "test_eval": test_eval,
    }


def summarize(out_dir: Path) -> str:
    rows = [
        "# Skip held-out test eval (72 prompts)",
        "",
        "Skip layers fixed from 8-prompt search; evaluated on remaining 72 prompts per domain.",
        "",
        "| tag | skip_layers | train Δacc | test base acc | test best acc | test Δacc | train score | test base score | test best score | test Δscore |",
        "|-----|-------------|------------|---------------|---------------|-----------|-------------|-----------------|-----------------|-------------|",
    ]
    for path in sorted(out_dir.glob("*_heldout.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        tr = d.get("train_eval", {})
        te = d.get("test_eval", {})
        tb, best = te.get("baseline", {}), te.get("best", {})
        delta = te.get("delta", {})
        train_dacc = (tr.get("best_mean_acc") or 0) - (tr.get("baseline_mean_acc") or 0)
        rows.append(
            f"| {d['tag']} | `{d.get('skip_layers', [])}` | {train_dacc:+.2f} | "
            f"{tb.get('mean_accepted_per_step', float('nan')):.2f} | "
            f"{best.get('mean_accepted_per_step', float('nan')):.2f} | "
            f"{delta.get('delta_mean_accepted_per_step', float('nan')):+.2f} | "
            f"{tr.get('baseline_task_score', float('nan')):.3f} | "
            f"{tb.get('task_score', float('nan')):.3f} | "
            f"{best.get('task_score', float('nan')):.3f} | "
            f"{delta.get('delta_task_score', float('nan')):+.3f} |"
        )
    text = "\n".join(rows) + "\n"
    (out_dir / "SUMMARY.md").write_text(text, encoding="utf-8")
    return text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tree-dir", default=str(REPO / "results/vicuna_max_accept_relaxed_20260611"))
    p.add_argument("--nontree-dir", default=str(REPO / "results/nontree_skip_max_accept_20260615"))
    p.add_argument(
        "--extra-nontree-dirs",
        default="",
        help="comma size:path pairs, e.g. 13b:/path/to/13b,33b:/path/to/33b",
    )
    p.add_argument("--output-dir", default=str(REPO / f"results/skip_heldout_eval_{time.strftime('%Y%m%d')}"))
    p.add_argument("--train-size", type=int, default=8)
    p.add_argument("--output-len", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", default="", help="comma tags, e.g. rag_7b_hydra,gsm8k_sps")
    p.add_argument(
        "--kinds",
        default="",
        help="comma kinds to run: hydra,samd,eagle3,nontree (default: all)",
    )
    p.add_argument(
        "--worker-group",
        default="",
        help="internal: run a single kind:size:method group in-process",
    )
    p.add_argument(
        "--no-subprocess",
        action="store_true",
        help="run all groups in one process (may OOM on large models)",
    )
    return p.parse_args()


def run_group_jobs(
    args: argparse.Namespace,
    group_jobs: list[HeldoutJob],
    *,
    kind: str,
    size: str,
    method: str,
    out_dir: Path,
    failures_path: Path,
) -> None:
    print(f"\n=== group kind={kind} size={size} method={method} n={len(group_jobs)} ===", flush=True)
    model = runner = None
    try:
        if kind == "hydra":
            vicuna = resolve_vicuna(size)
            model = load_hydra_model(vicuna, hydra_path=resolve_hydra_path(size))
            eval_fn = eval_hydra
        elif kind == "samd":
            vicuna = resolve_vicuna(size)
            eagle_path = EAGLE_PATHS.get(size, EAGLE_PATHS["7b"])
            model = load_samd_model(vicuna, eagle_path=eagle_path, size=size)
            eval_fn = eval_samd
        elif kind == "eagle3":
            cfg = MODEL_PRESETS["vicuna13"]
            model = load_eagle3_model(
                base_model=cfg["base_model"],
                ea_model=cfg["ea_model"],
                total_token=cfg.get("total_token", 60),
                use_eagle3=cfg.get("use_eagle3", True),
            )
        elif kind == "nontree":
            from scripts.run_specbench_nontree_4domains import resolve_vicuna as resolve_vicuna_nontree

            runner = MethodRunner(method, resolve_vicuna_nontree(size), DRAFTER68)
        else:
            raise ValueError(kind)

        for job in group_jobs:
            out_path = out_dir / job.out_name
            if out_path.exists() and not args.force:
                print(f"[skip] {job.tag} exists", flush=True)
                continue

            search = load_search_payload(job.source_json)
            skip_layers = set(search.get("best", {}).get("skip_layers") or search.get("skip_layers") or [])
            domain = SCORE_CATEGORY[job.dataset]

            if kind == "eagle3":
                test_items = load_test_items_eagle3(
                    job.dataset, seed=args.seed, output_len=args.output_len, train_size=args.train_size
                )
            elif kind == "nontree":
                test_items = load_test_items_nontree(
                    job.dataset, seed=args.seed, output_len=args.output_len, train_size=args.train_size
                )
            else:
                test_items = load_test_items_tree(
                    job.dataset, seed=args.seed, output_len=args.output_len, train_size=args.train_size
                )

            print(
                f"[run] {job.tag} test_n={len(test_items)} skip={sorted(skip_layers)} ...",
                flush=True,
            )
            t0 = time.perf_counter()
            try:
                if kind == "eagle3":
                    test_eval = run_eagle3_job(model, test_items, skip_layers, domain)
                elif kind == "nontree":
                    test_eval = run_nontree_job(runner, test_items, skip_layers, domain)
                else:
                    test_eval = run_tree_job(
                        job,
                        model=model,
                        test_items=test_items,
                        skip_layers=skip_layers,
                        domain=domain,
                        eval_fn=eval_fn,
                    )

                payload = build_payload(
                    job,
                    search,
                    test_eval,
                    train_size=args.train_size,
                    test_size=len(test_items),
                    seed=args.seed,
                    output_len=args.output_len,
                )
                payload["wall_s"] = time.perf_counter() - t0
                write_json(payload, out_path)
                d = payload["test_eval"]["delta"]
                print(
                    f"[done] {job.tag} test Δacc={d.get('delta_mean_accepted_per_step'):+.3f} "
                    f"Δscore={d.get('delta_task_score'):+.4f} ({payload['wall_s']:.1f}s)",
                    flush=True,
                )
            except Exception as exc:
                msg = f"[FAILED] {job.tag}: {exc}"
                print(msg, flush=True)
                with failures_path.open("a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")
                if kind == "samd" and model is not None:
                    model.cache = None
                    if hasattr(model, "draft"):
                        model.draft.reset()
            torch.cuda.empty_cache()
    finally:
        if model is not None:
            del model
        if runner is not None:
            runner.close()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def spawn_group_worker(args: argparse.Namespace, kind: str, size: str, method: str) -> int:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--tree-dir",
        args.tree_dir,
        "--nontree-dir",
        args.nontree_dir,
        "--extra-nontree-dirs",
        args.extra_nontree_dirs,
        "--output-dir",
        args.output_dir,
        "--train-size",
        str(args.train_size),
        "--output-len",
        str(args.output_len),
        "--seed",
        str(args.seed),
        "--worker-group",
        f"{kind}:{size}:{method}",
    ]
    if args.force:
        cmd.append("--force")
    if args.only:
        cmd.extend(["--only", args.only])
    if args.kinds:
        cmd.extend(["--kinds", args.kinds])
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cu13 = "/root/autodl-tmp/conda/envs/specdecode/lib/python3.11/site-packages/nvidia/cu13/lib"
    nccl = "/root/autodl-tmp/venvs/specbench/lib/python3.11/site-packages/nvidia/nccl/lib"
    env["LD_LIBRARY_PATH"] = f"{cu13}:{nccl}:{env.get('LD_LIBRARY_PATH', '')}"
    print(f"[spawn] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env).returncode


def main() -> None:
    install_transformers_compat()
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    out_dir = ensure_dir(args.output_dir)
    tree_dir = Path(args.tree_dir)
    nontree_dirs: list[tuple[str, Path]] = [("7b", Path(args.nontree_dir))]
    for part in args.extra_nontree_dirs.split(","):
        part = part.strip()
        if not part:
            continue
        size, path = part.split(":", 1)
        nontree_dirs.append((size.strip(), Path(path.strip())))
    jobs = discover_jobs(tree_dir, nontree_dirs)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    if only:
        jobs = [j for j in jobs if j.tag in only]
    kinds = {x.strip() for x in args.kinds.split(",") if x.strip()}
    if kinds:
        jobs = [j for j in jobs if j.kind in kinds]

    print(f"[INFO] {len(jobs)} held-out jobs -> {out_dir}", flush=True)
    failures_path = out_dir / "failures.log"

    grouped: dict[tuple[str, str, str], list[HeldoutJob]] = {}
    for job in jobs:
        key = (job.kind, job.size, job.method)
        grouped.setdefault(key, []).append(job)

    if args.worker_group:
        kind, size, method = args.worker_group.split(":", 2)
        group_jobs = grouped.get((kind, size, method), [])
        if not group_jobs:
            print(f"[WARN] no jobs for worker group {args.worker_group}", flush=True)
            return
        run_group_jobs(
            args,
            group_jobs,
            kind=kind,
            size=size,
            method=method,
            out_dir=out_dir,
            failures_path=failures_path,
        )
        return

    for (kind, size, method), group_jobs in grouped.items():
        if args.no_subprocess:
            run_group_jobs(
                args,
                group_jobs,
                kind=kind,
                size=size,
                method=method,
                out_dir=out_dir,
                failures_path=failures_path,
            )
            continue
        rc = spawn_group_worker(args, kind, size, method)
        if rc != 0:
            msg = f"[FAILED] group {kind}:{size}:{method} rc={rc}"
            print(msg, flush=True)
            with failures_path.open("a", encoding="utf-8") as fh:
                fh.write(msg + "\n")

    summary = summarize(out_dir)
    print(summary, flush=True)
    print(f"Done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
