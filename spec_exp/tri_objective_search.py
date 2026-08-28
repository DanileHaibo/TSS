"""Tri-objective target-layer search with score-valley traversal.

Unlike pareto_bridge_v2, the accept beam is not pruned by task score. This is
important for synergistic skip sets whose proper subsets temporarily lose
quality. Final selection still requires score, accept, and throughput to each
match or beat the native baseline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from spec_exp.pareto_bridge_search import allowed_layers
from spec_exp.sleb_skip_search import SlebSearchConfig, _metric, _toks, score_within_tol

ALGORITHM_NAME = "tri_objective_v3"
ALGORITHM_VERSION = "3.0.0"
SCHEMA_VERSION = "search_result_v3"


@dataclass(frozen=True)
class TriObjectiveOptions:
    score_bridge_drop_tol: float = 0.05
    score_bridge_tol_mode: str = "relative"
    metrics_beam_width: int = 4
    accept_beam_width: int = 6
    speed_beam_width: int = 3
    refine_top_k: int = 8
    refine_max_evals: int = 180
    refine_unused_layers: int = 12
    candidate_top_k: int = 8
    accept_floor_ratio: float = 1.0
    speed_floor_ratio: float = 1.0
    seed_sets: tuple[tuple[int, ...], ...] = ()


def _finite(value: float) -> bool:
    return value == value and math.isfinite(value)


def three_win_feasible(
    entry: dict[str, Any],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    *,
    accept_floor_ratio: float = 1.0,
    speed_floor_ratio: float = 1.0,
) -> bool:
    score = _metric(entry, config.score_key)
    base_score = _metric(baseline, config.score_key)
    accept = _metric(entry, config.accept_metric)
    base_accept = _metric(baseline, config.accept_metric)
    speed, base_speed = _toks(entry), _toks(baseline)
    return (
        score_within_tol(score, base_score, 0.0, "absolute")
        and accept >= base_accept * accept_floor_ratio
        and speed >= base_speed * speed_floor_ratio
    )


def tri_geomean_ratio(
    entry: dict[str, Any],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
) -> float:
    pairs = (
        (_metric(entry, config.score_key), _metric(baseline, config.score_key)),
        (_metric(entry, config.accept_metric), _metric(baseline, config.accept_metric)),
        (_toks(entry), _toks(baseline)),
    )
    ratios: list[float] = []
    for value, base in pairs:
        if not (_finite(value) and _finite(base)) or base <= 0:
            return -math.inf
        ratios.append(max(value / base, 1e-12))
    return math.prod(ratios) ** (1.0 / 3.0)


def pareto_frontier_metrics(
    entries: Iterable[dict[str, Any]], config: SlebSearchConfig
) -> list[dict[str, Any]]:
    unique: dict[tuple[int, ...], dict[str, Any]] = {}
    for entry in entries:
        key = tuple(sorted(entry.get("skip_layers") or []))
        unique[key] = entry
    points = list(unique.values())
    frontier: list[dict[str, Any]] = []
    for candidate in points:
        vector = (
            _metric(candidate, config.score_key),
            _metric(candidate, config.accept_metric),
            _toks(candidate),
        )
        if not all(_finite(x) for x in vector):
            continue
        if any(
            all(ov >= cv for ov, cv in zip(
                (
                    _metric(other, config.score_key),
                    _metric(other, config.accept_metric),
                    _toks(other),
                ),
                vector,
            ))
            and any(ov > cv for ov, cv in zip(
                (
                    _metric(other, config.score_key),
                    _metric(other, config.accept_metric),
                    _toks(other),
                ),
                vector,
            ))
            for other in points
            if other is not candidate
        ):
            continue
        frontier.append(candidate)
    return frontier


def pick_three_win(
    entries: Iterable[dict[str, Any]],
    *,
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    options: TriObjectiveOptions,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feasible = [
        entry
        for entry in entries
        if three_win_feasible(
            entry,
            baseline,
            config,
            accept_floor_ratio=options.accept_floor_ratio,
            speed_floor_ratio=options.speed_floor_ratio,
        )
    ]
    if not feasible:
        feasible = [baseline]
    ranked = sorted(
        feasible,
        key=lambda entry: (
            tri_geomean_ratio(entry, baseline, config),
            _metric(entry, config.accept_metric),
            _toks(entry),
            _metric(entry, config.score_key),
        ),
        reverse=True,
    )
    return ranked[0], ranked


def _summary(
    entry: dict[str, Any],
    *,
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    role: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "skip_layers": sorted(entry.get("skip_layers") or []),
        "num_skip_layers": len(entry.get("skip_layers") or []),
        config.score_key: _metric(entry, config.score_key),
        config.accept_metric: _metric(entry, config.accept_metric),
        "mean_accepted_per_step": entry.get("mean_accepted_per_step"),
        "accept_rate": entry.get("accept_rate"),
        "tok_per_s": _toks(entry),
        "tri_geomean_ratio": tri_geomean_ratio(entry, baseline, config),
    }


def tri_objective_v3_search(
    *,
    eval_fn: Callable[[set[int]], dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    options: TriObjectiveOptions | None = None,
    on_trial_error: Callable[[int, Exception], None] | None = None,
) -> tuple[
    set[int],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    options = options or TriObjectiveOptions()
    baseline = dict(baseline)
    baseline["skip_layers"] = []
    baseline["num_skip_layers"] = 0
    baseline.setdefault("tok_per_s", _toks(baseline))
    base_score = _metric(baseline, config.score_key)
    base_accept = _metric(baseline, config.accept_metric)
    base_speed = _toks(baseline)
    layers = allowed_layers(config)
    allowed = set(layers)

    memo: dict[frozenset[int], dict[str, Any]] = {frozenset(): baseline}
    history: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    stats = {
        "trials_requested": 0,
        "trials_executed": 0,
        "trials_cache_hits": 0,
        "trials_failed": 0,
        "trials_refine": 0,
        "seed_sets_evaluated": 0,
    }

    def row(
        phase: str,
        skip_set: frozenset[int],
        metrics: dict[str, Any] | None,
        parent: frozenset[int] | None,
        error: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "phase": phase,
            "round": len(skip_set),
            "skip_layers": sorted(skip_set),
            "num_skip_layers": len(skip_set),
            "parent_skip_layers": sorted(parent) if parent is not None else None,
            "three_win_feasible": False,
        }
        if metrics is not None:
            result.update(
                {
                    config.score_key: _metric(metrics, config.score_key),
                    config.accept_metric: _metric(metrics, config.accept_metric),
                    "mean_accepted_per_step": metrics.get("mean_accepted_per_step"),
                    "accept_rate": metrics.get("accept_rate"),
                    "tok_per_s": _toks(metrics),
                    "three_win_feasible": three_win_feasible(
                        metrics,
                        baseline,
                        config,
                        accept_floor_ratio=options.accept_floor_ratio,
                        speed_floor_ratio=options.speed_floor_ratio,
                    ),
                    "tri_geomean_ratio": tri_geomean_ratio(metrics, baseline, config),
                }
            )
        if error:
            result["error"] = error
        return result

    history.append(row("baseline", frozenset(), baseline, None))

    def evaluate(
        skip_set: frozenset[int], phase: str, parent: frozenset[int] | None
    ) -> dict[str, Any] | None:
        stats["trials_requested"] += 1
        if skip_set in memo:
            stats["trials_cache_hits"] += 1
            return memo[skip_set]
        try:
            metrics = dict(eval_fn(set(skip_set)))
        except Exception as exc:
            stats["trials_failed"] += 1
            history.append(
                row(phase, skip_set, None, parent, f"{type(exc).__name__}: {exc}")
            )
            if on_trial_error:
                added = sorted(skip_set - (parent or frozenset()))
                on_trial_error(added[-1] if added else -1, exc)
            return None
        metrics["skip_layers"] = sorted(skip_set)
        metrics["num_skip_layers"] = len(skip_set)
        metrics.setdefault("tok_per_s", _toks(metrics))
        memo[skip_set] = metrics
        stats["trials_executed"] += 1
        history.append(row(phase, skip_set, metrics, parent))
        return metrics

    seeds: list[frozenset[int]] = []
    for raw_seed in options.seed_sets:
        seed = frozenset(int(x) for x in raw_seed if int(x) in allowed)
        if not seed or len(seed) > config.max_skip_layers or seed in seeds:
            continue
        seeds.append(seed)
        if evaluate(seed, "historical_seed", None) is not None:
            stats["seed_sets_evaluated"] += 1

    print(
        f"  [{ALGORITHM_NAME}] layers={len(layers)} depth={config.max_skip_layers} "
        f"beams metrics={options.metrics_beam_width}/accept={options.accept_beam_width}/"
        f"speed={options.speed_beam_width}; seeds={len(seeds)}",
        flush=True,
    )

    beam: list[frozenset[int]] = [frozenset()]
    for depth in range(1, config.max_skip_layers + 1):
        candidates: dict[frozenset[int], dict[str, Any]] = {}
        for parent in beam:
            for layer in layers:
                if layer in parent:
                    continue
                child = parent | {layer}
                metrics = evaluate(child, "multi_track_explore", parent)
                if metrics is not None:
                    candidates[child] = metrics
        for seed in seeds:
            if len(seed) == depth and seed in memo:
                candidates[seed] = memo[seed]
        if not candidates:
            break

        items = list(candidates.items())
        score_pool = [
            item
            for item in items
            if score_within_tol(
                _metric(item[1], config.score_key),
                base_score,
                options.score_bridge_drop_tol,
                options.score_bridge_tol_mode,
            )
        ]
        accept_pool = [
            item
            for item in items
            if _metric(item[1], config.accept_metric) >= base_accept
        ]
        speed_pool = [
            item
            for item in items
            if _toks(item[1]) >= base_speed
            and _metric(item[1], config.accept_metric) >= base_accept * 0.98
        ]
        score_pool.sort(
            key=lambda item: (
                _metric(item[1], config.score_key),
                _metric(item[1], config.accept_metric),
                _toks(item[1]),
            ),
            reverse=True,
        )
        accept_pool.sort(
            key=lambda item: (
                _metric(item[1], config.accept_metric),
                _toks(item[1]),
                _metric(item[1], config.score_key),
            ),
            reverse=True,
        )
        speed_pool.sort(
            key=lambda item: (
                _toks(item[1]),
                _metric(item[1], config.accept_metric),
                _metric(item[1], config.score_key),
            ),
            reverse=True,
        )
        selected = (
            score_pool[: options.metrics_beam_width]
            + accept_pool[: options.accept_beam_width]
            + speed_pool[: options.speed_beam_width]
        )
        beam = list(dict.fromkeys(key for key, _ in selected))
        snapshots.append(
            {
                "depth": depth,
                "candidate_count": len(candidates),
                "score_beam": [sorted(key) for key, _ in score_pool[: options.metrics_beam_width]],
                "accept_beam": [sorted(key) for key, _ in accept_pool[: options.accept_beam_width]],
                "speed_beam": [sorted(key) for key, _ in speed_pool[: options.speed_beam_width]],
                "merged_parent_count": len(beam),
            }
        )
        if not beam:
            break
        best = max(
            (memo[key] for key in beam),
            key=lambda entry: tri_geomean_ratio(entry, baseline, config),
        )
        print(
            f"  [{ALGORITHM_NAME} depth {depth}] candidates={len(candidates)} "
            f"beam={len(beam)} top={best['skip_layers']} "
            f"score={_metric(best, config.score_key):.4f} "
            f"accept={_metric(best, config.accept_metric):.3f} "
            f"tok/s={_toks(best):.1f}",
            flush=True,
        )

    selected, ranked = pick_three_win(
        memo.values(), baseline=baseline, config=config, options=options
    )
    frontier = pareto_frontier_metrics(memo.values(), config)
    extrema = [
        selected,
        max(memo.values(), key=lambda x: _metric(x, config.score_key)),
        max(memo.values(), key=lambda x: _metric(x, config.accept_metric)),
        max(memo.values(), key=_toks),
    ]
    extrema.extend(ranked[: options.refine_top_k])
    extrema.extend(memo[seed] for seed in seeds if seed in memo)
    bases: list[frozenset[int]] = []
    for entry in extrema:
        key = frozenset(entry.get("skip_layers") or [])
        if key and key not in bases:
            bases.append(key)
        if len(bases) >= options.refine_top_k + len(seeds):
            break

    singles = [
        (next(iter(key)), value) for key, value in memo.items() if len(key) == 1
    ]
    singles.sort(
        key=lambda item: (
            max(
                _metric(item[1], config.score_key) / max(base_score, 1e-12),
                _metric(item[1], config.accept_metric) / max(base_accept, 1e-12),
                _toks(item[1]) / max(base_speed, 1e-12),
            ),
            tri_geomean_ratio(item[1], baseline, config),
        ),
        reverse=True,
    )
    seed_layers = [layer for seed in seeds for layer in sorted(seed)]
    promising = list(
        dict.fromkeys(seed_layers + [layer for layer, _ in singles] + layers)
    )[: options.refine_unused_layers]
    refine_seen: set[frozenset[int]] = set()
    refine_count = 0
    for base in bases:
        moves: list[frozenset[int]] = []
        moves.extend(base - {removed} for removed in base)
        if len(base) < config.max_skip_layers:
            moves.extend(base | {added} for added in promising if added not in base)
        moves.extend(
            (base - {removed}) | {added}
            for removed in base
            for added in promising
            if added not in base
        )
        for candidate in moves:
            if refine_count >= options.refine_max_evals:
                break
            if not candidate or candidate in refine_seen or candidate == base:
                continue
            refine_seen.add(candidate)
            cached = candidate in memo
            evaluate(candidate, "synergy_refine", base)
            if not cached:
                refine_count += 1
                stats["trials_refine"] += 1
        if refine_count >= options.refine_max_evals:
            break

    selected, ranked = pick_three_win(
        memo.values(), baseline=baseline, config=config, options=options
    )
    final_skip = set(selected.get("skip_layers") or [])
    final_metrics = dict(eval_fn(final_skip)) if final_skip else dict(baseline)
    final_metrics["skip_layers"] = sorted(final_skip)
    final_metrics["num_skip_layers"] = len(final_skip)
    final_metrics.setdefault("tok_per_s", _toks(final_metrics))
    frontier = pareto_frontier_metrics(memo.values(), config)
    candidates = {
        "selected": _summary(
            final_metrics, baseline=baseline, config=config, role="selected"
        ),
        "top_three_win": [
            _summary(entry, baseline=baseline, config=config, role="three_win")
            for entry in ranked[: options.candidate_top_k]
        ],
        "max_metrics": _summary(
            max(memo.values(), key=lambda x: _metric(x, config.score_key)),
            baseline=baseline,
            config=config,
            role="max_metrics",
        ),
        "max_accept": _summary(
            max(memo.values(), key=lambda x: _metric(x, config.accept_metric)),
            baseline=baseline,
            config=config,
            role="max_accept",
        ),
        "max_speed": _summary(
            max(memo.values(), key=_toks),
            baseline=baseline,
            config=config,
            role="max_speed",
        ),
    }
    selected_key = tuple(sorted(final_skip))
    for item in history:
        item["selected"] = tuple(item["skip_layers"]) == selected_key
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": {
            "name": ALGORITHM_NAME,
            "version": ALGORITHM_VERSION,
            "supersedes": "pareto_bridge_v2",
        },
        "search_config": {
            **options.__dict__,
            "seed_sets": [list(seed) for seed in options.seed_sets],
            "allowed_layers": layers,
        },
        "beam_snapshots": snapshots,
        "eval_stats": stats,
        "coverage": {
            "eligible_layer_count": len(layers),
            "unique_skip_sets_evaluated": len(memo),
            "three_win_candidate_count": len(ranked),
            "pareto_frontier_count": len(frontier),
        },
    }
    print(
        f"  [{ALGORITHM_NAME}] SELECTED skip={sorted(final_skip)} "
        f"score={_metric(final_metrics, config.score_key):.4f} "
        f"accept={_metric(final_metrics, config.accept_metric):.3f} "
        f"tok/s={_toks(final_metrics):.1f}; evals={stats['trials_executed']}",
        flush=True,
    )
    return final_skip, history, final_metrics, candidates, metadata
