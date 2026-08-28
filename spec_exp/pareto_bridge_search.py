"""Versioned Pareto/bridge target-layer search.

This module intentionally lives beside ``sleb_skip_search`` so the historical
``max_metrics_balanced`` implementation remains bit-for-bit reproducible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from spec_exp.sleb_skip_search import (
    SlebSearchConfig,
    _metric,
    _toks,
    score_within_tol,
)


ALGORITHM_NAME = "pareto_bridge_v2"
ALGORITHM_VERSION = "2.0.0"
SCHEMA_VERSION = "search_result_v2"


@dataclass(frozen=True)
class ParetoBridgeOptions:
    """Options that affect search coverage and therefore cache identity."""

    bridge_score_drop_tol: float = 0.05
    bridge_score_tol_mode: str = "relative"
    beam_width_per_objective: int = 2
    refine_top_k: int = 4
    refine_max_evals: int = 80
    refine_unused_layers: int = 8
    score_weight: float = 0.50
    skip_weight: float = 0.25
    accept_weight: float = 0.25


def allowed_layers(config: SlebSearchConfig) -> list[int]:
    """Return every eligible layer; v2 never truncates the scan."""
    lo = max(0, config.early_barrier)
    hi = config.num_layers - max(0, config.latter_barrier)
    step = max(1, int(config.layer_step))
    return list(range(hi - 1, lo - 1, -step))


def _is_finite(value: float) -> bool:
    return value == value and math.isfinite(value)


def _strict_score_feasible(
    entry: dict[str, Any],
    *,
    baseline_score: float,
    config: SlebSearchConfig,
) -> bool:
    score = _metric(entry, config.score_key)
    return score_within_tol(score, baseline_score, 0.0, "absolute")


def _bridge_score_feasible(
    entry: dict[str, Any],
    *,
    baseline_score: float,
    config: SlebSearchConfig,
    options: ParetoBridgeOptions,
) -> bool:
    score = _metric(entry, config.score_key)
    return score_within_tol(
        score,
        baseline_score,
        options.bridge_score_drop_tol,
        options.bridge_score_tol_mode,
    )


def _summary(
    entry: dict[str, Any],
    config: SlebSearchConfig,
    *,
    role: str,
    balance_score: float | None = None,
) -> dict[str, Any]:
    layers = sorted(entry.get("skip_layers") or [])
    out = {
        "role": role,
        "skip_layers": layers,
        "num_skip_layers": len(layers),
        config.score_key: _metric(entry, config.score_key),
        config.accept_metric: _metric(entry, config.accept_metric),
        "mean_accepted_per_step": entry.get("mean_accepted_per_step"),
        "accept_rate": entry.get("accept_rate"),
        "tok_per_s": _toks(entry),
    }
    if balance_score is not None:
        out["balance_score"] = float(balance_score)
    move = entry.get("refine_move")
    if move is not None:
        out["refine_move"] = move
    return out


def pareto_frontier_3d(
    entries: list[dict[str, Any]],
    config: SlebSearchConfig,
) -> list[dict[str, Any]]:
    """Return non-dominated entries maximizing score, skip count, and accept."""
    unique: dict[tuple[int, ...], dict[str, Any]] = {}
    for entry in entries:
        key = tuple(sorted(entry.get("skip_layers") or []))
        previous = unique.get(key)
        if previous is None:
            unique[key] = entry
            continue
        prev_key = (
            _metric(previous, config.score_key),
            _metric(previous, config.accept_metric),
            _toks(previous),
        )
        cur_key = (
            _metric(entry, config.score_key),
            _metric(entry, config.accept_metric),
            _toks(entry),
        )
        if cur_key > prev_key:
            unique[key] = entry

    points = list(unique.values())
    frontier: list[dict[str, Any]] = []
    for candidate in points:
        cs = _metric(candidate, config.score_key)
        ca = _metric(candidate, config.accept_metric)
        cn = len(candidate.get("skip_layers") or [])
        if not (_is_finite(cs) and _is_finite(ca)):
            continue
        dominated = False
        for other in points:
            if other is candidate:
                continue
            os = _metric(other, config.score_key)
            oa = _metric(other, config.accept_metric)
            on = len(other.get("skip_layers") or [])
            if not (_is_finite(os) and _is_finite(oa)):
                continue
            if (
                os >= cs
                and on >= cn
                and oa >= ca
                and (os > cs or on > cn or oa > ca)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda e: (
            _metric(e, config.score_key),
            len(e.get("skip_layers") or []),
            _metric(e, config.accept_metric),
        ),
        reverse=True,
    )


def _balance_scores(
    frontier: list[dict[str, Any]],
    *,
    baseline_score: float,
    config: SlebSearchConfig,
    options: ParetoBridgeOptions,
) -> dict[tuple[int, ...], float]:
    if not frontier:
        return {}
    max_score = max(_metric(e, config.score_key) for e in frontier)
    max_accept = max(_metric(e, config.accept_metric) for e in frontier)
    max_skip = max(len(e.get("skip_layers") or []) for e in frontier)
    score_span = max(max_score - baseline_score, 0.0)
    weight_sum = options.score_weight + options.skip_weight + options.accept_weight
    if weight_sum <= 0:
        raise ValueError("Pareto balance weights must sum to a positive value")

    out: dict[tuple[int, ...], float] = {}
    for entry in frontier:
        score = _metric(entry, config.score_key)
        accept = _metric(entry, config.accept_metric)
        n_skip = len(entry.get("skip_layers") or [])
        # A baseline-preserving score starts at 0.5 and the best observed score
        # reaches 1.0. This avoids unstable raw ratios when the baseline is near 0.
        score_norm = (
            1.0
            if score_span <= 1e-12
            else 0.5 + 0.5 * max(0.0, score - baseline_score) / score_span
        )
        accept_norm = accept / max_accept if max_accept > 1e-12 else 1.0
        skip_norm = n_skip / max_skip if max_skip > 0 else 1.0
        score_norm = max(score_norm, 1e-9)
        accept_norm = max(accept_norm, 1e-9)
        skip_norm = max(skip_norm, 1e-9)
        weighted_product = (
            score_norm**options.score_weight
            * skip_norm**options.skip_weight
            * accept_norm**options.accept_weight
        )
        out[tuple(sorted(entry.get("skip_layers") or []))] = weighted_product ** (
            1.0 / weight_sum
        )
    return out


def pick_from_strict_pool(
    entries: list[dict[str, Any]],
    *,
    baseline_score: float,
    config: SlebSearchConfig,
    options: ParetoBridgeOptions,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    strict = [
        entry
        for entry in entries
        if _strict_score_feasible(
            entry, baseline_score=baseline_score, config=config
        )
    ]
    if not strict:
        raise ValueError("strict candidate pool unexpectedly empty")
    frontier = pareto_frontier_3d(strict, config)
    balances = _balance_scores(
        frontier,
        baseline_score=baseline_score,
        config=config,
        options=options,
    )

    max_metrics = max(
        strict,
        key=lambda e: (
            _metric(e, config.score_key),
            len(e.get("skip_layers") or []),
            _metric(e, config.accept_metric),
        ),
    )
    max_skip = max(
        strict,
        key=lambda e: (
            len(e.get("skip_layers") or []),
            _metric(e, config.score_key),
            _metric(e, config.accept_metric),
        ),
    )
    max_accept = max(
        strict,
        key=lambda e: (
            _metric(e, config.accept_metric),
            _metric(e, config.score_key),
            len(e.get("skip_layers") or []),
        ),
    )
    balanced = max(
        frontier,
        key=lambda e: (
            balances[tuple(sorted(e.get("skip_layers") or []))],
            _metric(e, config.score_key),
            len(e.get("skip_layers") or []),
            _metric(e, config.accept_metric),
        ),
    )

    candidates = {
        "max_metrics": _summary(max_metrics, config, role="max_metrics"),
        "max_skip": _summary(max_skip, config, role="max_skip"),
        "max_accept": _summary(max_accept, config, role="max_accept"),
        "balanced": _summary(
            balanced,
            config,
            role="balanced",
            balance_score=balances[
                tuple(sorted(balanced.get("skip_layers") or []))
            ],
        ),
        "pareto_frontier": [
            _summary(
                entry,
                config,
                role="pareto",
                balance_score=balances[
                    tuple(sorted(entry.get("skip_layers") or []))
                ],
            )
            for entry in frontier
        ],
    }
    return balanced, candidates, frontier


def _select_objective_beams(
    candidates: dict[frozenset[int], dict[str, Any]],
    *,
    config: SlebSearchConfig,
    width: int,
) -> tuple[list[tuple[frozenset[int], dict[str, Any]]], dict[str, list[list[int]]]]:
    width = max(1, int(width))
    items = list(candidates.items())

    def score(entry: dict[str, Any]) -> float:
        value = _metric(entry, config.score_key)
        return value if _is_finite(value) else -1e18

    def accept(entry: dict[str, Any]) -> float:
        value = _metric(entry, config.accept_metric)
        return value if _is_finite(value) else -1e18

    rankings = {
        "metrics": sorted(
            items,
            key=lambda kv: (
                score(kv[1]),
                accept(kv[1]),
                _toks(kv[1]),
                sum(kv[0]) / max(len(kv[0]), 1),
            ),
            reverse=True,
        ),
        "accept": sorted(
            items,
            key=lambda kv: (
                accept(kv[1]),
                score(kv[1]),
                _toks(kv[1]),
                sum(kv[0]) / max(len(kv[0]), 1),
            ),
            reverse=True,
        ),
        # At a fixed depth every candidate has the same |S|. Throughput and
        # score margin identify paths most likely to remain feasible deeper.
        "continuation": sorted(
            items,
            key=lambda kv: (
                _toks(kv[1]),
                score(kv[1]),
                accept(kv[1]),
                sum(kv[0]) / max(len(kv[0]), 1),
            ),
            reverse=True,
        ),
    }

    merged: list[tuple[frozenset[int], dict[str, Any]]] = []
    seen: set[frozenset[int]] = set()
    snapshot: dict[str, list[list[int]]] = {}
    for role, ranked in rankings.items():
        selected = ranked[:width]
        snapshot[role] = [sorted(key) for key, _ in selected]
        for key, metrics in selected:
            if key in seen:
                continue
            seen.add(key)
            merged.append((key, metrics))
    return merged, snapshot


def pareto_bridge_v2_search(
    *,
    eval_fn: Callable[[set[int]], dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    options: ParetoBridgeOptions | None = None,
    on_trial_error: Callable[[int, Exception], None] | None = None,
) -> tuple[
    set[int],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Run full-scan, bridge-feasible multi-beam search and Pareto refinement."""
    options = options or ParetoBridgeOptions()
    baseline = dict(baseline)
    baseline.setdefault("tok_per_s", _toks(baseline))
    baseline["skip_layers"] = []
    baseline["num_skip_layers"] = 0
    baseline_score = _metric(baseline, config.score_key)
    baseline_accept = _metric(baseline, config.accept_metric)
    layers = allowed_layers(config)

    memo: dict[frozenset[int], dict[str, Any]] = {frozenset(): baseline}
    history: list[dict[str, Any]] = []
    eval_stats = {
        "trials_requested": 0,
        "trials_executed": 0,
        "trials_cache_hits": 0,
        "trials_failed": 0,
        "trials_bridge_feasible": 0,
        "trials_strict_feasible": 1,
        "trials_refine": 0,
    }
    beam_snapshots: list[dict[str, Any]] = []

    def history_row(
        *,
        phase: str,
        depth: int,
        skip_set: frozenset[int],
        metrics: dict[str, Any] | None,
        parent: frozenset[int] | None,
        move: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "phase": phase,
            "round": depth,
            "skip_layers": sorted(skip_set),
            "num_skip_layers": len(skip_set),
            "parent_skip_layers": sorted(parent) if parent is not None else None,
            "bridge_feasible": False,
            "final_feasible": False,
            "selected": False,
        }
        if metrics is not None:
            row.update(
                {
                    config.score_key: _metric(metrics, config.score_key),
                    config.accept_metric: _metric(metrics, config.accept_metric),
                    "mean_accepted_per_step": metrics.get(
                        "mean_accepted_per_step"
                    ),
                    "accept_rate": metrics.get("accept_rate"),
                    "tok_per_s": _toks(metrics),
                }
            )
            row["bridge_feasible"] = _bridge_score_feasible(
                metrics,
                baseline_score=baseline_score,
                config=config,
                options=options,
            )
            row["final_feasible"] = _strict_score_feasible(
                metrics, baseline_score=baseline_score, config=config
            )
        if move is not None:
            row["refine_move"] = move
        if error is not None:
            row["error"] = error
        return row

    history.append(
        history_row(
            phase="baseline",
            depth=0,
            skip_set=frozenset(),
            metrics=baseline,
            parent=None,
        )
    )

    def evaluate(
        skip_set: frozenset[int],
        *,
        phase: str,
        depth: int,
        parent: frozenset[int] | None,
        move: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        eval_stats["trials_requested"] += 1
        if skip_set in memo:
            eval_stats["trials_cache_hits"] += 1
            return memo[skip_set]
        try:
            metrics = dict(eval_fn(set(skip_set)))
        except Exception as exc:
            eval_stats["trials_failed"] += 1
            history.append(
                history_row(
                    phase=phase,
                    depth=depth,
                    skip_set=skip_set,
                    metrics=None,
                    parent=parent,
                    move=move,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            if on_trial_error:
                added = sorted(skip_set - (parent or frozenset()))
                on_trial_error(added[-1] if added else -1, exc)
            return None
        metrics.setdefault("tok_per_s", _toks(metrics))
        metrics["skip_layers"] = sorted(skip_set)
        metrics["num_skip_layers"] = len(skip_set)
        if move is not None:
            metrics["refine_move"] = move
        memo[skip_set] = metrics
        eval_stats["trials_executed"] += 1
        row = history_row(
            phase=phase,
            depth=depth,
            skip_set=skip_set,
            metrics=metrics,
            parent=parent,
            move=move,
        )
        history.append(row)
        if row["bridge_feasible"]:
            eval_stats["trials_bridge_feasible"] += 1
        if row["final_feasible"]:
            eval_stats["trials_strict_feasible"] += 1
        return metrics

    print(
        f"  [{ALGORITHM_NAME}] full-scan layers={len(layers)} "
        f"depth={config.max_skip_layers} beam/objective={options.beam_width_per_objective}; "
        f"bridge score tol={options.bridge_score_drop_tol}"
        f"({options.bridge_score_tol_mode}); final score>={baseline_score:.4f}",
        flush=True,
    )

    beam: list[tuple[frozenset[int], dict[str, Any]]] = [
        (frozenset(), baseline)
    ]
    for depth in range(1, config.max_skip_layers + 1):
        candidates: dict[frozenset[int], dict[str, Any]] = {}
        requested_this_depth: set[frozenset[int]] = set()
        for parent_set, _parent_metrics in beam:
            for layer in layers:
                if layer in parent_set:
                    continue
                child = frozenset(set(parent_set) | {layer})
                if child in requested_this_depth:
                    eval_stats["trials_cache_hits"] += 1
                    continue
                requested_this_depth.add(child)
                metrics = evaluate(
                    child,
                    phase="explore_multi_beam",
                    depth=depth,
                    parent=parent_set,
                )
                if metrics is None:
                    continue
                if _bridge_score_feasible(
                    metrics,
                    baseline_score=baseline_score,
                    config=config,
                    options=options,
                ):
                    candidates[child] = metrics

        if not candidates:
            print(
                f"  [{ALGORITHM_NAME} depth {depth}] no bridge-feasible child; stop",
                flush=True,
            )
            break
        beam, snapshot = _select_objective_beams(
            candidates,
            config=config,
            width=options.beam_width_per_objective,
        )
        beam_snapshots.append(
            {
                "depth": depth,
                "candidate_count": len(candidates),
                "roles": snapshot,
                "merged_parent_count": len(beam),
            }
        )
        top = max(
            beam,
            key=lambda kv: (
                _metric(kv[1], config.score_key),
                _metric(kv[1], config.accept_metric),
            ),
        )
        print(
            f"  [{ALGORITHM_NAME} depth {depth}] candidates={len(candidates)} "
            f"merged_beam={len(beam)} top={sorted(top[0])} "
            f"score={_metric(top[1], config.score_key):.4f} "
            f"accept={_metric(top[1], config.accept_metric):.3f}",
            flush=True,
        )

    strict_entries = [
        metrics
        for metrics in memo.values()
        if _strict_score_feasible(
            metrics, baseline_score=baseline_score, config=config
        )
    ]
    balanced, candidates_payload, frontier = pick_from_strict_pool(
        strict_entries,
        baseline_score=baseline_score,
        config=config,
        options=options,
    )

    # Local refinement around the best Pareto points and three extrema.
    shortlist_keys: list[tuple[int, ...]] = []
    for role in ("balanced", "max_metrics", "max_skip", "max_accept"):
        key = tuple(candidates_payload[role]["skip_layers"])
        if key not in shortlist_keys:
            shortlist_keys.append(key)
    for item in candidates_payload["pareto_frontier"]:
        key = tuple(item["skip_layers"])
        if key not in shortlist_keys:
            shortlist_keys.append(key)
        if len(shortlist_keys) >= max(1, options.refine_top_k):
            break
    shortlist_keys = shortlist_keys[: max(1, options.refine_top_k)]

    single_layer_ranked = sorted(
        [
            (next(iter(key)), value)
            for key, value in memo.items()
            if len(key) == 1
        ],
        key=lambda pair: (
            _metric(pair[1], config.score_key),
            _metric(pair[1], config.accept_metric),
            _toks(pair[1]),
        ),
        reverse=True,
    )
    promising_layers = [
        layer for layer, _ in single_layer_ranked[: options.refine_unused_layers]
    ]
    refine_seen: set[frozenset[int]] = set()
    refine_budget = max(0, int(options.refine_max_evals))
    refine_executed = 0
    for key in shortlist_keys:
        if refine_executed >= refine_budget:
            break
        base_set = frozenset(key)
        unused = [
            layer
            for layer in promising_layers + layers
            if layer not in base_set
        ]
        # Preserve order while limiting swaps/adds to the strongest singles.
        unused = list(dict.fromkeys(unused))[: options.refine_unused_layers]
        moves: list[tuple[frozenset[int], dict[str, Any]]] = []
        for removed in sorted(base_set):
            moves.append(
                (
                    frozenset(set(base_set) - {removed}),
                    {"type": "delete", "removed": removed},
                )
            )
        if len(base_set) < config.max_skip_layers:
            for added in unused:
                moves.append(
                    (
                        frozenset(set(base_set) | {added}),
                        {"type": "add", "added": added},
                    )
                )
        for removed in sorted(base_set):
            for added in unused:
                moves.append(
                    (
                        frozenset((set(base_set) - {removed}) | {added}),
                        {
                            "type": "swap",
                            "removed": removed,
                            "added": added,
                        },
                    )
                )

        for candidate_set, move in moves:
            if refine_executed >= refine_budget:
                break
            if candidate_set in refine_seen or candidate_set == base_set:
                continue
            refine_seen.add(candidate_set)
            was_cached = candidate_set in memo
            metrics = evaluate(
                candidate_set,
                phase="local_refine",
                depth=len(candidate_set),
                parent=base_set,
                move=move,
            )
            if not was_cached:
                refine_executed += 1
                eval_stats["trials_refine"] += 1
            if metrics is None:
                continue

    strict_entries = [
        metrics
        for metrics in memo.values()
        if _strict_score_feasible(
            metrics, baseline_score=baseline_score, config=config
        )
    ]
    balanced, candidates_payload, frontier = pick_from_strict_pool(
        strict_entries,
        baseline_score=baseline_score,
        config=config,
        options=options,
    )
    final_skip = set(balanced.get("skip_layers") or [])

    try:
        final_metrics = (
            dict(eval_fn(final_skip)) if final_skip else dict(baseline)
        )
        final_metrics.setdefault("tok_per_s", _toks(final_metrics))
        final_metrics["skip_layers"] = sorted(final_skip)
        final_metrics["num_skip_layers"] = len(final_skip)
    except Exception as exc:
        print(
            f"  [{ALGORITHM_NAME}] final re-eval failed ({exc}); use cached metrics",
            flush=True,
        )
        final_metrics = dict(balanced)

    selected_key = tuple(sorted(final_skip))
    for row in history:
        row["pareto_member"] = tuple(row.get("skip_layers") or []) in {
            tuple(sorted(entry.get("skip_layers") or [])) for entry in frontier
        }
        row["selected"] = tuple(row.get("skip_layers") or []) == selected_key

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": {
            "name": ALGORITHM_NAME,
            "version": ALGORITHM_VERSION,
            "supersedes": "max_metrics_balanced",
        },
        "search_config": {
            "full_layer_scan": True,
            "allowed_layers": layers,
            "bridge_score_drop_tol": options.bridge_score_drop_tol,
            "bridge_score_tol_mode": options.bridge_score_tol_mode,
            "final_score_drop_tol": 0.0,
            "beam_width_per_objective": options.beam_width_per_objective,
            "refine_top_k": options.refine_top_k,
            "refine_max_evals": options.refine_max_evals,
            "score_weight": options.score_weight,
            "skip_weight": options.skip_weight,
            "accept_weight": options.accept_weight,
        },
        "beam_snapshots": beam_snapshots,
        "eval_stats": eval_stats,
        "coverage": {
            "eligible_layer_count": len(layers),
            "single_layers_evaluated": sum(1 for key in memo if len(key) == 1),
            "unique_skip_sets_evaluated": len(memo),
            "strict_candidate_count": len(strict_entries),
            "pareto_frontier_count": len(frontier),
        },
        "baseline_accept": baseline_accept,
    }
    print(
        f"  [{ALGORITHM_NAME}] SELECTED skip={sorted(final_skip)} "
        f"score={_metric(final_metrics, config.score_key):.4f} "
        f"accept={_metric(final_metrics, config.accept_metric):.3f}; "
        f"evals={eval_stats['trials_executed']} "
        f"cache_hits={eval_stats['trials_cache_hits']} "
        f"pareto={len(frontier)}",
        flush=True,
    )
    return final_skip, history, final_metrics, candidates_payload, metadata
