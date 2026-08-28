#!/usr/bin/env python3
"""Prepare and execute resumable four-GPU TSS ablation tasks."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_ROOT = REPO / "results" / "tss_ablation_20260716"
RELEASE = (
    REPO
    / "results"
    / "tss_triple_win_final_20260716"
    / "triple_win_summary.json"
)
PY_EAGLE = "/root/autodl-tmp/venvs/eagle3/bin/python"
PY_SAMD = "/root/autodl-tmp/conda/envs/specdecode/bin/python"


def ablation_variants(
    skip_layers: list[int],
) -> list[tuple[str, list[int]]]:
    """Name full, leave-one-out, and balanced front/back subsets."""
    full = tuple(sorted(skip_layers))
    candidates: list[tuple[str, tuple[int, ...]]] = [("full", full)]
    candidates.extend(
        (
            f"drop_{removed}",
            tuple(layer for layer in full if layer != removed),
        )
        for removed in full
    )
    midpoint = (len(full) + 1) // 2
    candidates.extend(
        (
            ("front_half", full[:midpoint]),
            ("back_half", full[-midpoint:]),
        )
    )
    unique: dict[tuple[int, ...], str] = {}
    for name, layers in candidates:
        if layers and layers not in unique:
            unique[layers] = name
    return [(name, list(layers)) for layers, name in unique.items()]


def ablation_candidates(skip_layers: list[int]) -> list[list[int]]:
    return [layers for _, layers in ablation_variants(skip_layers)]


def task(
    *,
    task_id: str,
    phase: str,
    command: list[str],
    output_hint: str,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "phase": phase,
        "command": command,
        "output_hint": output_hint,
    }


def prepare(root: Path) -> None:
    release = json.loads(RELEASE.read_text())
    rows = release["rows"]
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []

    audit = {
        "schema_version": "tss_ablation_candidates_v1",
        "datasets": {},
    }
    for row in rows:
        candidates = ablation_candidates(row["skip_layers"])
        dataset = row["dataset"]
        if row["size"] == "7B":
            audit["datasets"][dataset] = {
                "candidates": [
                    {"skip_layers": layers, "source": "layer_ablation"}
                    for layers in candidates
                ]
            }
            output_dir = root / "layer" / "7b"
            command = [
                PY_EAGLE,
                "-u",
                str(REPO / "scripts" / "eval_eagle7_candidate_pool.py"),
                "--dataset",
                dataset,
                "--candidate-audit",
                str(inputs / "eagle7_candidates.json"),
                "--output-dir",
                str(output_dir),
                "--max-candidates",
                str(len(candidates)),
                "--train-size",
                "16",
                "--seed",
                "42",
                "--output-len",
                "96",
            ]
            tasks.append(
                task(
                    task_id=f"layer__7b__{dataset}",
                    phase="layer",
                    command=command,
                    output_hint=str(
                        output_dir / dataset / "candidate_heldout.json"
                    ),
                )
            )
        else:
            candidate_file = inputs / f"samd13_{dataset}_candidates.json"
            candidate_file.write_text(
                json.dumps(
                    {
                        "dataset": dataset,
                        "candidates": candidates,
                    },
                    indent=2,
                )
                + "\n"
            )
            train_size = int(row["selection_train_size"])
            output = (
                root
                / "layer"
                / "13b"
                / dataset
                / "candidate_heldout.json"
            )
            command = [
                PY_SAMD,
                "-u",
                str(REPO / "scripts" / "eval_samd_target_candidate_pool.py"),
                "--target",
                "llama2_13b",
                "--dataset",
                dataset,
                "--candidates-json",
                str(candidate_file),
                "--output",
                str(output),
                "--train-size",
                str(train_size),
                "--output-len",
                "96",
            ]
            tasks.append(
                task(
                    task_id=f"layer__13b__{dataset}",
                    phase="layer",
                    command=command,
                    output_hint=str(output),
                )
            )
    (inputs / "eagle7_candidates.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    for dataset in ("translation", "qa"):
        for variant, mode, extra in (
            ("v3_full", "tri_objective_v3", []),
            ("v3_no_accept", "tri_objective_v3", ["--disable-accept-track"]),
            ("v3_no_refine", "tri_objective_v3", ["--disable-refine"]),
            ("pareto_v2", "pareto_bridge_v2", []),
        ):
            output_dir = root / "search" / "7b" / dataset / variant
            command = [
                PY_EAGLE,
                "-u",
                str(REPO / "scripts" / "run_tss_max_toks_pipeline.py"),
                "--jobs",
                "7b_eagle",
                "--datasets",
                dataset,
                "--train-size",
                "16",
                "--seed",
                "42",
                "--output-len",
                "96",
                "--max-skip-layers",
                "5",
                "--search-mode",
                mode,
                "--pareto-beam-width",
                "4",
                "--refine-top-k",
                "8",
                "--refine-max-evals",
                "180",
                "--output-dir",
                str(output_dir),
                *extra,
            ]
            tasks.append(
                task(
                    task_id=f"search__7b__{dataset}__{variant}",
                    phase="search",
                    command=command,
                    output_hint=str(output_dir),
                )
            )

    search_13b = (
        ("translation", "score16_min6", "score_first", 16, 6),
        ("translation", "score16_min3", "score_first", 16, 3),
        ("translation", "accept16_min6", "max_skip_latter", 16, 6),
        ("qa", "score16_min6", "score_first", 16, 6),
        ("qa", "score32_min6", "score_first", 32, 6),
        ("qa", "accept16_min6", "max_skip_latter", 16, 6),
    )
    for dataset, variant, mode, train_size, min_skips in search_13b:
        output_dir = root / "search" / "13b" / dataset / variant
        command = [
            PY_SAMD,
            "-u",
            str(REPO / "scripts" / "run_samd_target_skip_search.py"),
            "--target",
            "llama2_13b",
            "--dataset",
            dataset,
            "--output-dir",
            str(output_dir),
            "--search-mode",
            mode,
            "--train-size",
            str(train_size),
            "--min-skips",
            str(min_skips),
            "--max-skip-layers",
            "12",
        ]
        tasks.append(
            task(
                task_id=f"search__13b__{dataset}__{variant}",
                phase="search",
                command=command,
                output_hint=str(output_dir),
            )
        )

    root.mkdir(parents=True, exist_ok=True)
    (root / "tasks.json").write_text(
        json.dumps(
            {
                "schema_version": "tss_ablation_tasks_v1",
                "created_at": time.time(),
                "tasks": tasks,
            },
            indent=2,
        )
        + "\n"
    )
    (root / "logs").mkdir(exist_ok=True)
    (root / "state").mkdir(exist_ok=True)
    (root / "claims").mkdir(exist_ok=True)
    print(
        f"prepared {sum(t['phase'] == 'layer' for t in tasks)} layer and "
        f"{sum(t['phase'] == 'search' for t in tasks)} search tasks at {root}"
    )


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def try_claim(root: Path, item: dict[str, Any], gpu: int) -> Path | None:
    claim = root / "claims" / f"{item['id']}.json"
    if claim.exists():
        try:
            payload = json.loads(claim.read_text())
            if process_alive(int(payload["pid"])):
                return None
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
        claim.unlink(missing_ok=True)
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w") as handle:
        json.dump(
            {
                "pid": os.getpid(),
                "gpu": gpu,
                "host": socket.gethostname(),
                "claimed_at": time.time(),
            },
            handle,
        )
    return claim


def worker(root: Path, phase: str, gpu: int, *, retry_failed: bool) -> None:
    tasks = json.loads((root / "tasks.json").read_text())["tasks"]
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HOME": "/root/autodl-tmp/hf-cache",
            "TRANSFORMERS_CACHE": "/root/autodl-tmp/hf-cache",
            "HF_ENDPOINT": "https://hf-mirror.com",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    while True:
        selected = None
        claim = None
        for item in tasks:
            if item["phase"] != phase:
                continue
            state_path = root / "state" / f"{item['id']}.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state.get("status") == "success" or not retry_failed:
                    continue
            claim = try_claim(root, item, gpu)
            if claim is not None:
                selected = item
                break
        if selected is None:
            print(f"[worker gpu={gpu}] no pending {phase} tasks", flush=True)
            return

        log_path = root / "logs" / f"{selected['id']}__gpu{gpu}.log"
        state_path = root / "state" / f"{selected['id']}.json"
        started = time.time()
        print(f"[worker gpu={gpu}] START {selected['id']}", flush=True)
        with log_path.open("w") as log:
            completed = subprocess.run(
                selected["command"],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        state = {
            "id": selected["id"],
            "phase": phase,
            "gpu": gpu,
            "status": "success" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "started_at": started,
            "finished_at": time.time(),
            "elapsed_s": time.time() - started,
            "command": selected["command"],
            "output_hint": selected["output_hint"],
            "log": str(log_path),
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        claim.unlink(missing_ok=True)
        print(
            f"[worker gpu={gpu}] {state['status'].upper()} "
            f"{selected['id']} ({state['elapsed_s']:.1f}s)",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "worker", "status", "cancel")
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phase", choices=("layer", "search"))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root)
        return
    if args.action == "status":
        tasks = json.loads((args.root / "tasks.json").read_text())["tasks"]
        for item in tasks:
            state_path = args.root / "state" / f"{item['id']}.json"
            status = (
                json.loads(state_path.read_text()).get("status", "unknown")
                if state_path.exists()
                else "pending"
            )
            print(f"{status:8} {item['phase']:6} {item['id']}")
        return
    if args.action == "cancel":
        if args.phase is None:
            parser.error("cancel requires --phase")
        tasks = json.loads((args.root / "tasks.json").read_text())["tasks"]
        for item in tasks:
            if item["phase"] != args.phase:
                continue
            state_path = args.root / "state" / f"{item['id']}.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state.get("status") == "success":
                    continue
            state_path.write_text(
                json.dumps(
                    {
                        "id": item["id"],
                        "phase": item["phase"],
                        "status": "cancelled",
                        "reason": "scope_reduced_to_single_gpu_layer_ablation",
                        "finished_at": time.time(),
                    },
                    indent=2,
                )
                + "\n"
            )
        return
    if args.phase is None or args.gpu is None:
        parser.error("worker requires --phase and --gpu")
    worker(
        args.root,
        args.phase,
        args.gpu,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()
