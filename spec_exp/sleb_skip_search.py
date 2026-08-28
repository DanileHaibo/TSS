"""SLEB-style skip layer search for speculative-decoding verify targets.

Reference: https://github.com/jiwonsong-dev/SLEB (ICML 2024)
  - Maintain an alive list of transformer block indices.
  - Each phase temporarily skip one candidate block, measure harm, restore.
  - Permanently skip the block with minimum combined harm (within tolerances).
  - Respect early/latter barriers on the *alive* list (not raw layer ids).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SlebSearchConfig:
    num_layers: int
    max_skip_layers: int
    early_barrier: int = 1
    latter_barrier: int = 1
    accept_drop_tol: float = 0.05
    score_drop_tol: float = 0.10
    score_tol_mode: str = "relative"
    accept_metric: str = "mean_accepted_per_step"
    score_key: str = "task_score"
    accept_weight: float = 1.0
    score_weight: float = 1.0
    layer_step: int = 1
    exhaustive_singles: bool = True


def _metric(metrics: dict[str, Any], key: str) -> float:
    v = metrics.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float("nan")
    return float(v)


def score_within_tol(
    score: float,
    init_score: float,
    tol: float,
    mode: str,
) -> bool:
    if math.isnan(score):
        return False
    if math.isnan(init_score):
        return True
    if mode == "relative":
        if abs(init_score) < 1e-9:
            return score >= init_score - tol
        return score >= init_score * (1.0 - tol)
    return score >= init_score - tol


def accept_within_tol(accept: float, init_accept: float, tol: float) -> bool:
    if math.isnan(accept) or math.isnan(init_accept):
        return False
    if tol <= 0.0:
        return accept >= init_accept
    return accept >= init_accept * (1.0 - tol)


def combined_harm(
    *,
    baseline: dict[str, Any],
    trial: dict[str, Any],
    config: SlebSearchConfig,
) -> float:
    a0 = _metric(baseline, config.accept_metric)
    a1 = _metric(trial, config.accept_metric)
    s0 = _metric(baseline, config.score_key)
    s1 = _metric(trial, config.score_key)
    a_drop = max(0.0, (a0 - a1) / max(abs(a0), 1e-9))
    if config.score_tol_mode == "relative":
        s_drop = max(0.0, (s0 - s1) / max(abs(s0), 1e-9))
    else:
        s_drop = max(0.0, s0 - s1)
    return config.accept_weight * a_drop + config.score_weight * s_drop


def trial_feasible(
    *,
    baseline: dict[str, Any],
    trial: dict[str, Any],
    config: SlebSearchConfig,
) -> bool:
    accept = _metric(trial, config.accept_metric)
    score = _metric(trial, config.score_key)
    init_accept = _metric(baseline, config.accept_metric)
    init_score = _metric(baseline, config.score_key)
    return accept_within_tol(accept, init_accept, config.accept_drop_tol) and score_within_tol(
        score, init_score, config.score_drop_tol, config.score_tol_mode
    )


def sleb_layer_search(
    *,
    eval_fn: Callable[[set[int]], dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    on_trial_error: Callable[[int, Exception], None] | None = None,
) -> tuple[set[int], list[dict[str, Any]], dict[str, Any]]:
    """Return (skip_layers, history, best_metrics)."""
    skip_set: set[int] = set()
    alive_list = list(range(config.num_layers))
    current = dict(baseline)
    init_accept = _metric(baseline, config.accept_metric)
    init_score = _metric(baseline, config.score_key)

    history: list[dict[str, Any]] = [
        {
            "round": 0,
            "layer": None,
            "skip_layers": [],
            "alive_count": len(alive_list),
            config.accept_metric: init_accept,
            config.score_key: init_score,
            "accept_rate": baseline.get("accept_rate"),
            "mean_accepted_per_step": baseline.get("mean_accepted_per_step"),
            "tok_per_s": baseline.get("tok_per_s"),
        }
    ]

    for rnd in range(1, config.max_skip_layers + 1):
        if len(skip_set) >= config.max_skip_layers:
            break
        if len(alive_list) <= config.early_barrier + config.latter_barrier:
            print(
                f"  [sleb round {rnd}] alive_list too short ({len(alive_list)}), stop",
                flush=True,
            )
            break

        best_j: int | None = None
        best_metrics: dict[str, Any] | None = None
        min_harm = float("inf")

        upper = len(alive_list) - config.latter_barrier
        for j in range(config.early_barrier, upper, max(1, config.layer_step)):
            layer_id = alive_list[j]
            trial_set = set(skip_set)
            trial_set.add(layer_id)
            try:
                m = eval_fn(trial_set)
            except Exception as exc:
                if on_trial_error:
                    on_trial_error(layer_id, exc)
                continue

            if not trial_feasible(baseline=baseline, trial=m, config=config):
                continue

            harm = combined_harm(baseline=baseline, trial=m, config=config)
            if harm < min_harm - 1e-12:
                min_harm = harm
                best_j = j
                best_metrics = m

        if best_j is None or best_metrics is None:
            print(f"  [sleb round {rnd}] no feasible layer within tolerances, stop", flush=True)
            break

        layer_id = alive_list[best_j]
        skip_set.add(layer_id)
        del alive_list[best_j]
        current = best_metrics
        cur_accept = _metric(current, config.accept_metric)
        cur_score = _metric(current, config.score_key)
        history.append(
            {
                "round": rnd,
                "layer": layer_id,
                "skip_layers": sorted(skip_set),
                "alive_count": len(alive_list),
                "harm": min_harm,
                config.accept_metric: cur_accept,
                config.score_key: cur_score,
                "accept_rate": current.get("accept_rate"),
                "mean_accepted_per_step": current.get("mean_accepted_per_step"),
                "tok_per_s": current.get("tok_per_s"),
                "delta_accept": cur_accept - init_accept,
                "delta_score": cur_score - init_score,
            }
        )
        print(
            f"  [sleb round {rnd}] +block {layer_id} harm={min_harm:.4f} "
            f"{config.accept_metric}={cur_accept:.3f} {config.score_key}={cur_score:.4f} "
            f"tok/s={current.get('tok_per_s', 0):.1f}",
            flush=True,
        )

    return skip_set, history, current


def pick_best_preserve_score(
    *,
    history: list[dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
) -> dict[str, Any]:
    """Among history entries, pick max accept_metric with task_score >= baseline."""
    init_score = _metric(baseline, config.score_key)
    best_entry = history[0]
    best_accept = _metric(baseline, config.accept_metric)
    for entry in history:
        score = _metric(entry, config.score_key)
        if not score_within_tol(score, init_score, config.score_drop_tol, config.score_tol_mode):
            continue
        accept = _metric(entry, config.accept_metric)
        if math.isnan(accept):
            continue
        if accept > best_accept + 1e-12:
            best_accept = accept
            best_entry = entry
    return best_entry


def _history_row(
    *,
    phase: str,
    rnd: int,
    layer: int | None,
    skip_layers: list[int],
    metrics: dict[str, Any],
    init_accept: float,
    init_score: float,
    config: SlebSearchConfig,
) -> dict[str, Any]:
    cur_accept = _metric(metrics, config.accept_metric)
    cur_score = _metric(metrics, config.score_key)
    return {
        "phase": phase,
        "round": rnd,
        "layer": layer,
        "skip_layers": skip_layers,
        config.accept_metric: cur_accept,
        config.score_key: cur_score,
        "accept_rate": metrics.get("accept_rate"),
        "mean_accepted_per_step": metrics.get("mean_accepted_per_step"),
        "tok_per_s": metrics.get("tok_per_s"),
        "delta_accept": cur_accept - init_accept,
        "delta_score": cur_score - init_score,
        "selected": False,
    }


def _global_best_accept(history: list[dict[str, Any]], config: SlebSearchConfig) -> float:
    vals = [_metric(h, config.accept_metric) for h in history]
    clean = [v for v in vals if not math.isnan(v)]
    return max(clean) if clean else float("-inf")


def max_accept_preserve_score_search(
    *,
    eval_fn: Callable[[set[int]], dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    on_trial_error: Callable[[int, Exception], None] | None = None,
) -> tuple[set[int], list[dict[str, Any]], dict[str, Any]]:
    """Maximize accept_metric while keeping task_score >= baseline.

    Phase 1 (optional): exhaustive single-layer sweep over every layer.
    Phase 2: greedy multi-layer addition (layer_step=1 by default, no early stop on
    non-monotonic accept); stop only when no feasible one-layer addition beats history best.
    Final pick: history entry with highest accept_metric among score-feasible configs.
    """
    skip_set: set[int] = set()
    init_accept = _metric(baseline, config.accept_metric)
    init_score = _metric(baseline, config.score_key)
    step = max(1, config.layer_step)

    history: list[dict[str, Any]] = [
        _history_row(
            phase="baseline",
            rnd=0,
            layer=None,
            skip_layers=[],
            metrics=baseline,
            init_accept=init_accept,
            init_score=init_score,
            config=config,
        )
    ]

    if config.exhaustive_singles:
        print(f"  [max_accept] exhaustive single-layer sweep ({config.num_layers} layers)...", flush=True)
        for layer in range(0, config.num_layers, step):
            try:
                m = eval_fn({layer})
            except Exception as exc:
                if on_trial_error:
                    on_trial_error(layer, exc)
                continue
            score = _metric(m, config.score_key)
            if not score_within_tol(score, init_score, config.score_drop_tol, config.score_tol_mode):
                continue
            cur_accept = _metric(m, config.accept_metric)
            history.append(
                _history_row(
                    phase="single",
                    rnd=layer,
                    layer=layer,
                    skip_layers=[layer],
                    metrics=m,
                    init_accept=init_accept,
                    init_score=init_score,
                    config=config,
                )
            )
            print(
                f"    [single L{layer}] {config.accept_metric}={cur_accept:.3f} "
                f"{config.score_key}={_metric(m, config.score_key):.4f}",
                flush=True,
            )

    # Seed multi-layer greedy from best single so we actually try {best}+L combos.
    # Without this, greedy round 1 duplicates singles and ties history_best → immediate stop.
    if config.exhaustive_singles and len(history) > 1:
        seed_cfg = SlebSearchConfig(
            num_layers=config.num_layers,
            max_skip_layers=config.max_skip_layers,
            score_drop_tol=config.score_drop_tol,
            score_tol_mode=config.score_tol_mode,
            accept_metric=config.accept_metric,
            score_key=config.score_key,
        )
        seed_pick = pick_best_preserve_score(history=history, baseline=baseline, config=seed_cfg)
        seed_layers = list(seed_pick.get("skip_layers") or [])
        if seed_layers:
            skip_set = set(seed_layers)
            print(
                f"  [max_accept] seed multi-layer search from best single skip={sorted(skip_set)} "
                f"({config.accept_metric}={_metric(seed_pick, config.accept_metric):.3f})",
                flush=True,
            )

    for rnd in range(1, config.max_skip_layers + 1):
        if len(skip_set) >= config.max_skip_layers:
            break

        history_best = _global_best_accept(history, config)
        best_layer: int | None = None
        best_metrics: dict[str, Any] | None = None
        best_accept = -1.0

        for layer in range(0, config.num_layers, step):
            if layer in skip_set:
                continue
            trial_set = set(skip_set)
            trial_set.add(layer)
            try:
                m = eval_fn(trial_set)
            except Exception as exc:
                if on_trial_error:
                    on_trial_error(layer, exc)
                continue

            score = _metric(m, config.score_key)
            if not score_within_tol(score, init_score, config.score_drop_tol, config.score_tol_mode):
                continue

            accept = _metric(m, config.accept_metric)
            if math.isnan(accept) or accept <= best_accept + 1e-12:
                continue
            best_accept = accept
            best_layer = layer
            best_metrics = m

        if best_layer is None or best_metrics is None:
            print(f"  [max_accept round {rnd}] no score-feasible candidate, stop", flush=True)
            break

        if best_accept <= history_best + 1e-12:
            print(
                f"  [max_accept round {rnd}] best {config.accept_metric}={best_accept:.3f} "
                f"does not beat history best {history_best:.3f}, stop",
                flush=True,
            )
            break

        skip_set.add(best_layer)
        cur_score = _metric(best_metrics, config.score_key)
        history.append(
            _history_row(
                phase="greedy",
                rnd=rnd,
                layer=best_layer,
                skip_layers=sorted(skip_set),
                metrics=best_metrics,
                init_accept=init_accept,
                init_score=init_score,
                config=config,
            )
        )
        print(
            f"  [max_accept round {rnd}] +block {best_layer} "
            f"{config.accept_metric}={best_accept:.3f} {config.score_key}={cur_score:.4f} "
            f"tok/s={best_metrics.get('tok_per_s', 0):.1f}",
            flush=True,
        )

    pick_cfg = SlebSearchConfig(
        num_layers=config.num_layers,
        max_skip_layers=config.max_skip_layers,
        score_drop_tol=config.score_drop_tol,
        score_tol_mode=config.score_tol_mode,
        accept_metric=config.accept_metric,
        score_key=config.score_key,
    )
    picked = pick_best_preserve_score(history=history, baseline=baseline, config=pick_cfg)
    for h in history:
        h["selected"] = h.get("round") == picked.get("round")

    final_skip = set(picked.get("skip_layers") or [])

    def _metrics_from_pick(entry: dict[str, Any]) -> dict[str, Any]:
        layers = list(entry.get("skip_layers") or [])
        return {
            "skip_layers": layers,
            "num_skip_layers": len(layers),
            config.accept_metric: _metric(entry, config.accept_metric),
            config.score_key: _metric(entry, config.score_key),
            "accept_rate": entry.get("accept_rate"),
            "mean_accepted_per_step": entry.get("mean_accepted_per_step"),
            "tok_per_s": entry.get("tok_per_s"),
        }

    if picked.get("round", 0) > 0 and final_skip:
        try:
            final_metrics = eval_fn(final_skip)
        except Exception as exc:
            print(
                f"  [max_accept] final re-eval failed ({exc}), use history metrics",
                flush=True,
            )
            final_metrics = _metrics_from_pick(picked)
    elif picked.get("round", 0) > 0:
        final_metrics = _metrics_from_pick(picked)
    else:
        final_metrics = dict(baseline)

    return final_skip, history, final_metrics


def quality_hard_feasible(
    *,
    baseline: dict[str, Any],
    trial: dict[str, Any],
    config: SlebSearchConfig,
) -> bool:
    """Hard constraint: accept >= baseline AND task_score >= baseline (zero tolerance)."""
    accept = _metric(trial, config.accept_metric)
    score = _metric(trial, config.score_key)
    init_accept = _metric(baseline, config.accept_metric)
    init_score = _metric(baseline, config.score_key)
    if math.isnan(accept) or math.isnan(init_accept):
        return False
    if accept + 1e-12 < init_accept:
        return False
    if math.isnan(score):
        return False
    if math.isnan(init_score):
        return True
    return score + 1e-12 >= init_score


def _toks(metrics: dict[str, Any]) -> float:
    v = metrics.get("tok_per_s")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        wall = metrics.get("wall_s")
        toks = metrics.get("total_output_tokens")
        if wall and toks and float(wall) > 0:
            return float(toks) / float(wall)
        return float("nan")
    return float(v)


def _better_toks_tuple(trial: dict[str, Any], best: dict[str, Any], config: SlebSearchConfig) -> bool:
    """Lexicographic: tok/s, then accept, then score."""
    t_tps, b_tps = _toks(trial), _toks(best)
    if math.isnan(t_tps):
        return False
    if math.isnan(b_tps) or t_tps > b_tps + 1e-9:
        return True
    if abs(t_tps - b_tps) > 1e-9:
        return False
    t_a, b_a = _metric(trial, config.accept_metric), _metric(best, config.accept_metric)
    if t_a > b_a + 1e-12:
        return True
    if abs(t_a - b_a) > 1e-12:
        return False
    t_s, b_s = _metric(trial, config.score_key), _metric(best, config.score_key)
    return (not math.isnan(t_s)) and (math.isnan(b_s) or t_s > b_s + 1e-12)


def pick_best_max_toks(
    *,
    history: list[dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
) -> dict[str, Any]:
    """Among hard-feasible history entries, pick max tok/s (then accept, then score)."""
    best_entry = history[0]
    best_metrics = {
        config.accept_metric: _metric(baseline, config.accept_metric),
        config.score_key: _metric(baseline, config.score_key),
        "tok_per_s": _toks(baseline),
        "skip_layers": [],
    }
    for entry in history:
        trial = {
            config.accept_metric: _metric(entry, config.accept_metric),
            config.score_key: _metric(entry, config.score_key),
            "tok_per_s": _toks(entry),
            "skip_layers": entry.get("skip_layers") or [],
        }
        if not quality_hard_feasible(baseline=baseline, trial=trial, config=config):
            continue
        if _better_toks_tuple(trial, best_metrics, config):
            best_entry = entry
            best_metrics = trial
    return best_entry


def max_toks_preserve_quality_search(
    *,
    eval_fn: Callable[[set[int]], dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    on_trial_error: Callable[[int, Exception], None] | None = None,
) -> tuple[set[int], list[dict[str, Any]], dict[str, Any]]:
    """Maximize tok/s under hard constraints accept>=baseline and score>=baseline.

    Phase 1: exhaustive single-layer sweep (keep only hard-feasible).
    Phase 2: greedy multi-layer from best single; add a layer only if still feasible
             and tok/s (then accept/score) strictly improves.
    Final: best hard-feasible history entry; empty skip if baseline wins.
    """
    skip_set: set[int] = set()
    init_accept = _metric(baseline, config.accept_metric)
    init_score = _metric(baseline, config.score_key)
    step = max(1, config.layer_step)

    # Ensure baseline has tok_per_s for comparisons.
    if baseline.get("tok_per_s") is None or (
        isinstance(baseline.get("tok_per_s"), float) and math.isnan(baseline["tok_per_s"])
    ):
        baseline = dict(baseline)
        baseline["tok_per_s"] = _toks(baseline)

    history: list[dict[str, Any]] = [
        _history_row(
            phase="baseline",
            rnd=0,
            layer=None,
            skip_layers=[],
            metrics=baseline,
            init_accept=init_accept,
            init_score=init_score,
            config=config,
        )
    ]

    if config.exhaustive_singles:
        print(
            f"  [max_toks] exhaustive single-layer sweep ({config.num_layers} layers); "
            f"hard constraint accept>={init_accept:.3f} score>={init_score:.4f}",
            flush=True,
        )
        for layer in range(0, config.num_layers, step):
            try:
                m = eval_fn({layer})
            except Exception as exc:
                if on_trial_error:
                    on_trial_error(layer, exc)
                continue
            if m.get("tok_per_s") is None:
                m = dict(m)
                m["tok_per_s"] = _toks(m)
            if not quality_hard_feasible(baseline=baseline, trial=m, config=config):
                continue
            history.append(
                _history_row(
                    phase="single",
                    rnd=layer,
                    layer=layer,
                    skip_layers=[layer],
                    metrics=m,
                    init_accept=init_accept,
                    init_score=init_score,
                    config=config,
                )
            )
            print(
                f"    [single L{layer}] tok/s={_toks(m):.1f} "
                f"{config.accept_metric}={_metric(m, config.accept_metric):.3f} "
                f"{config.score_key}={_metric(m, config.score_key):.4f}",
                flush=True,
            )

    if config.exhaustive_singles and len(history) > 1:
        seed_pick = pick_best_max_toks(history=history, baseline=baseline, config=config)
        seed_layers = list(seed_pick.get("skip_layers") or [])
        if seed_layers:
            skip_set = set(seed_layers)
            print(
                f"  [max_toks] seed multi-layer from best single skip={sorted(skip_set)} "
                f"tok/s={_toks(seed_pick):.1f}",
                flush=True,
            )

    for rnd in range(1, config.max_skip_layers + 1):
        if len(skip_set) >= config.max_skip_layers:
            break

        best_layer: int | None = None
        best_metrics: dict[str, Any] | None = None
        # Compare against current skip_set metrics from history
        current_ref = {
            config.accept_metric: -1.0,
            config.score_key: -1.0,
            "tok_per_s": float("-inf"),
        }
        for h in history:
            if set(h.get("skip_layers") or []) == skip_set:
                current_ref = {
                    config.accept_metric: _metric(h, config.accept_metric),
                    config.score_key: _metric(h, config.score_key),
                    "tok_per_s": _toks(h),
                }
                break

        for layer in range(0, config.num_layers, step):
            if layer in skip_set:
                continue
            trial_set = set(skip_set)
            trial_set.add(layer)
            try:
                m = eval_fn(trial_set)
            except Exception as exc:
                if on_trial_error:
                    on_trial_error(layer, exc)
                continue
            if m.get("tok_per_s") is None:
                m = dict(m)
                m["tok_per_s"] = _toks(m)
            if not quality_hard_feasible(baseline=baseline, trial=m, config=config):
                continue
            cand_best = best_metrics if best_metrics is not None else current_ref
            if not _better_toks_tuple(m, cand_best, config):
                continue
            best_layer = layer
            best_metrics = m

        if best_layer is None or best_metrics is None:
            print(f"  [max_toks round {rnd}] no improving feasible candidate, stop", flush=True)
            break

        if not _better_toks_tuple(best_metrics, current_ref, config):
            print(
                f"  [max_toks round {rnd}] no tok/s improvement over current "
                f"{_toks(current_ref):.1f}, stop",
                flush=True,
            )
            break

        skip_set.add(best_layer)
        history.append(
            _history_row(
                phase="greedy",
                rnd=rnd,
                layer=best_layer,
                skip_layers=sorted(skip_set),
                metrics=best_metrics,
                init_accept=init_accept,
                init_score=init_score,
                config=config,
            )
        )
        print(
            f"  [max_toks round {rnd}] +block {best_layer} "
            f"tok/s={_toks(best_metrics):.1f} "
            f"{config.accept_metric}={_metric(best_metrics, config.accept_metric):.3f} "
            f"{config.score_key}={_metric(best_metrics, config.score_key):.4f}",
            flush=True,
        )

    picked = pick_best_max_toks(history=history, baseline=baseline, config=config)
    for h in history:
        h["selected"] = (
            h.get("round") == picked.get("round")
            and list(h.get("skip_layers") or []) == list(picked.get("skip_layers") or [])
            and h.get("phase") == picked.get("phase")
        )

    final_skip = set(picked.get("skip_layers") or [])

    def _metrics_from_pick(entry: dict[str, Any]) -> dict[str, Any]:
        layers = list(entry.get("skip_layers") or [])
        return {
            "skip_layers": layers,
            "num_skip_layers": len(layers),
            config.accept_metric: _metric(entry, config.accept_metric),
            config.score_key: _metric(entry, config.score_key),
            "accept_rate": entry.get("accept_rate"),
            "mean_accepted_per_step": entry.get("mean_accepted_per_step"),
            "tok_per_s": _toks(entry),
            "wall_s": entry.get("wall_s"),
            "total_output_tokens": entry.get("total_output_tokens"),
        }

    if final_skip:
        try:
            final_metrics = eval_fn(final_skip)
            if final_metrics.get("tok_per_s") is None:
                final_metrics = dict(final_metrics)
                final_metrics["tok_per_s"] = _toks(final_metrics)
        except Exception as exc:
            print(f"  [max_toks] final re-eval failed ({exc}), use history metrics", flush=True)
            final_metrics = _metrics_from_pick(picked)
    else:
        final_metrics = dict(baseline)
        final_metrics["skip_layers"] = []
        final_metrics["num_skip_layers"] = 0
        final_metrics["tok_per_s"] = _toks(baseline)

    return final_skip, history, final_metrics


def soft_quality_feasible(
    *,
    baseline: dict[str, Any],
    trial: dict[str, Any],
    config: SlebSearchConfig,
) -> bool:
    """Soft constraint: accept/score may drop within configured tolerances."""
    accept = _metric(trial, config.accept_metric)
    score = _metric(trial, config.score_key)
    init_accept = _metric(baseline, config.accept_metric)
    init_score = _metric(baseline, config.score_key)
    if not accept_within_tol(accept, init_accept, config.accept_drop_tol):
        return False
    return score_within_tol(score, init_score, config.score_drop_tol, config.score_tol_mode)


def pick_best_quality_first(
    *,
    history: list[dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    min_skips: int = 0,
) -> dict[str, Any]:
    """Final pick: accept hard (no drop), score within soft tol; prefer accept, then |S|>=min_skips, then tok/s."""
    best_entry = history[0]
    init_a = _metric(baseline, config.accept_metric)
    init_s = _metric(baseline, config.score_key)
    best_key = (init_a, 0, _toks(baseline), init_s if init_s == init_s else -1.0)

    def _ok(trial: dict[str, Any]) -> bool:
        # accept: no drop (hard)
        if not accept_within_tol(
            trial[config.accept_metric], init_a, 0.0
        ):
            return False
        # score: soft (config.score_drop_tol / mode)
        return score_within_tol(
            trial[config.score_key], init_s, config.score_drop_tol, config.score_tol_mode
        )

    # Prefer configs with |S| >= min_skips if any exist
    feasible = []
    for entry in history:
        trial = {
            config.accept_metric: _metric(entry, config.accept_metric),
            config.score_key: _metric(entry, config.score_key),
            "tok_per_s": _toks(entry),
            "skip_layers": list(entry.get("skip_layers") or []),
        }
        if _ok(trial):
            feasible.append((entry, trial))

    pool = feasible
    if min_skips > 0:
        deep = [(e, t) for e, t in feasible if len(t["skip_layers"]) >= min_skips]
        if deep:
            pool = deep

    for entry, trial in pool:
        n = len(trial["skip_layers"])
        key = (
            trial[config.accept_metric],
            n,
            trial["tok_per_s"] if trial["tok_per_s"] == trial["tok_per_s"] else -1.0,
            trial[config.score_key] if trial[config.score_key] == trial[config.score_key] else -1.0,
        )
        if key > best_key:
            best_key = key
            best_entry = entry
    return best_entry


def max_skip_latter_search(
    *,
    eval_fn: Callable[[set[int]], dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    on_trial_error: Callable[[int, Exception], None] | None = None,
    beam_width: int = 4,
    min_skips: int = 3,
    score_first: bool = False,
    enforce_score_tol_on_explore: bool = False,
) -> tuple[set[int], list[dict[str, Any]], dict[str, Any]]:
    """Explore to target depth; final keeps accept & soft score.

    Expansion (continue searching):
      - require accept >= baseline (hard, no drop)
      - by default score may temporarily drop (do NOT stop just because score dipped)
      - if enforce_score_tol_on_explore: also require score within score_drop_tol vs baseline
        at every expansion step (so tol changes the search path, not only final pick)
      - prefer latter layers; keep a beam of promising partial skip-sets

    Final selection:
      - accept >= baseline (hard)
      - score within score_drop_tol (e.g. 5% relative)
      - prefer higher accept, then more skips (>= min_skips if possible), then tok/s
    """
    init_accept = _metric(baseline, config.accept_metric)
    init_score = _metric(baseline, config.score_key)
    if baseline.get("tok_per_s") is None or (
        isinstance(baseline.get("tok_per_s"), float) and math.isnan(baseline["tok_per_s"])
    ):
        baseline = dict(baseline)
        baseline["tok_per_s"] = _toks(baseline)

    history: list[dict[str, Any]] = [
        _history_row(
            phase="baseline",
            rnd=0,
            layer=None,
            skip_layers=[],
            metrics=baseline,
            init_accept=init_accept,
            init_score=init_score,
            config=config,
        )
    ]

    lo = max(0, config.early_barrier)
    hi = config.num_layers - max(0, config.latter_barrier)
    label = "score_first" if score_first else "max_skip_latter"
    print(
        f"  [{label}] EXPLORE-to-depth={config.max_skip_layers} "
        f"min_skips={min_skips} beam={beam_width}; "
        + (
            "rank and select strictly by score; accept/tok/s only break ties; "
            if score_first
            else (
                (
                    f"expand if accept>={init_accept:.3f} and score_tol="
                    f"{config.score_drop_tol}({config.score_tol_mode}); "
                    if enforce_score_tol_on_explore
                    else f"expand if accept>={init_accept:.3f} (score may dip); "
                )
                + f"final: accept hard + score_tol={config.score_drop_tol}"
                f"({config.score_tol_mode}); "
            )
        )
        +
        f"prefer layers [{hi-1}..{lo}]",
        flush=True,
    )

    # beam items: (skip_set_frozenset, metrics_dict)
    beam: list[tuple[frozenset[int], dict[str, Any]]] = [(frozenset(), baseline)]

    for depth in range(1, config.max_skip_layers + 1):
        cand_map: dict[frozenset[int], dict[str, Any]] = {}
        for skip_f, _parent_m in beam:
            skip_set = set(skip_f)
            n_try = 0
            n_keep = 0
            for layer in range(hi - 1, lo - 1, -1):
                if layer in skip_set:
                    continue
                # Cap evaluations per parent, but always scan full latter→early range
                # until we have enough accept-preserving children or exhaust layers.
                if n_try >= 28 and n_keep >= 3:
                    break
                if n_try >= 40:
                    break
                n_try += 1
                trial_set = set(skip_set)
                trial_set.add(layer)
                key = frozenset(trial_set)
                if key in cand_map:
                    continue
                try:
                    m = eval_fn(trial_set)
                except Exception as exc:
                    if on_trial_error:
                        on_trial_error(layer, exc)
                    continue
                if m.get("tok_per_s") is None:
                    m = dict(m)
                    m["tok_per_s"] = _toks(m)
                # Default mode protects acceptance while score-first deliberately
                # keeps the highest-scoring path regardless of acceptance.
                if not score_first and not accept_within_tol(
                    _metric(m, config.accept_metric), init_accept, 0.0
                ):
                    continue
                if (
                    enforce_score_tol_on_explore
                    and not score_first
                    and not score_within_tol(
                        _metric(m, config.score_key),
                        init_score,
                        config.score_drop_tol,
                        config.score_tol_mode,
                    )
                ):
                    continue
                cand_map[key] = m
                n_keep += 1
                history.append(
                    _history_row(
                        phase="explore",
                        rnd=depth,
                        layer=layer,
                        skip_layers=sorted(trial_set),
                        metrics=m,
                        init_accept=init_accept,
                        init_score=init_score,
                        config=config,
                    )
                )

        if not cand_map:
            print(
                f"  [{label} depth {depth}] no viable child, stop explore",
                flush=True,
            )
            break

        # In score-first mode score is the only optimization objective;
        # acceptance and throughput are deterministic tie-breakers.
        ranked = sorted(
            cand_map.items(),
            key=lambda kv: (
                _metric(kv[1], config.score_key)
                if _metric(kv[1], config.score_key) == _metric(kv[1], config.score_key)
                else -1e9,
                _metric(kv[1], config.accept_metric)
                if score_first
                else 0.0,
                _toks(kv[1]) if _toks(kv[1]) == _toks(kv[1]) else -1.0,
                sum(kv[0]) / max(len(kv[0]), 1),
            )
            if score_first
            else (
                _metric(kv[1], config.accept_metric),
                _metric(kv[1], config.score_key)
                if _metric(kv[1], config.score_key) == _metric(kv[1], config.score_key)
                else -1e9,
                _toks(kv[1]) if _toks(kv[1]) == _toks(kv[1]) else -1.0,
                sum(kv[0]) / max(len(kv[0]), 1),
            ),
            reverse=True,
        )
        beam = [(k, v) for k, v in ranked[: max(1, beam_width)]]
        top = beam[0]
        print(
            f"  [{label} depth {depth}] beam={len(beam)}/{len(cand_map)} "
            f"top skip={sorted(top[0])} accept={_metric(top[1], config.accept_metric):.3f} "
            f"score={_metric(top[1], config.score_key):.4f} tok/s={_toks(top[1]):.1f}",
            flush=True,
        )

    if score_first:
        eligible = [
            entry
            for entry in history
            if len(entry.get("skip_layers") or []) >= min_skips
        ]
        pool = eligible or history
        picked = max(
            pool,
            key=lambda entry: (
                _metric(entry, config.score_key)
                if _metric(entry, config.score_key) == _metric(entry, config.score_key)
                else -1e18,
                _metric(entry, config.accept_metric)
                if _metric(entry, config.accept_metric)
                == _metric(entry, config.accept_metric)
                else -1e18,
                _toks(entry) if _toks(entry) == _toks(entry) else -1e18,
                sum(entry.get("skip_layers") or [])
                / max(len(entry.get("skip_layers") or []), 1),
            ),
        )
    else:
        picked = pick_best_quality_first(
            history=history, baseline=baseline, config=config, min_skips=min_skips
        )
    for h in history:
        h["selected"] = list(h.get("skip_layers") or []) == list(picked.get("skip_layers") or [])

    final_skip = set(picked.get("skip_layers") or [])

    def _metrics_from_pick(entry: dict[str, Any]) -> dict[str, Any]:
        layers = list(entry.get("skip_layers") or [])
        return {
            "skip_layers": layers,
            "num_skip_layers": len(layers),
            config.accept_metric: _metric(entry, config.accept_metric),
            config.score_key: _metric(entry, config.score_key),
            "accept_rate": entry.get("accept_rate"),
            "mean_accepted_per_step": entry.get("mean_accepted_per_step"),
            "tok_per_s": _toks(entry),
            "wall_s": entry.get("wall_s"),
            "total_output_tokens": entry.get("total_output_tokens"),
        }

    if final_skip:
        try:
            final_metrics = eval_fn(final_skip)
            if final_metrics.get("tok_per_s") is None:
                final_metrics = dict(final_metrics)
                final_metrics["tok_per_s"] = _toks(final_metrics)
        except Exception as exc:
            print(f"  [{label}] final re-eval failed ({exc})", flush=True)
            final_metrics = _metrics_from_pick(picked)
    else:
        final_metrics = dict(baseline)
        final_metrics["skip_layers"] = []
        final_metrics["num_skip_layers"] = 0
        final_metrics["tok_per_s"] = _toks(baseline)

    print(
        f"  [{label}] SELECTED skip={sorted(final_skip)} (|S|={len(final_skip)}) "
        f"accept={_metric(final_metrics, config.accept_metric):.3f} "
        f"score={_metric(final_metrics, config.score_key):.4f} "
        f"tok/s={_toks(final_metrics):.1f}",
        flush=True,
    )
    return final_skip, history, final_metrics


def _entry_triple(
    entry: dict[str, Any], config: SlebSearchConfig
) -> tuple[float, int, float]:
    score = _metric(entry, config.score_key)
    accept = _metric(entry, config.accept_metric)
    n_skip = len(entry.get("skip_layers") or [])
    return (
        score if score == score else -1e18,
        n_skip,
        accept if accept == accept else -1e18,
    )


def _summarize_candidate(
    entry: dict[str, Any], config: SlebSearchConfig, *, role: str
) -> dict[str, Any]:
    layers = list(entry.get("skip_layers") or [])
    return {
        "role": role,
        "skip_layers": layers,
        "num_skip_layers": len(layers),
        config.score_key: _metric(entry, config.score_key),
        config.accept_metric: _metric(entry, config.accept_metric),
        "mean_accepted_per_step": entry.get("mean_accepted_per_step"),
        "accept_rate": entry.get("accept_rate"),
        "tok_per_s": _toks(entry),
    }


def pick_metrics_extremes_and_balanced(
    *,
    history: list[dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Keep three extremes among score-preserving entries; pick the most balanced.

    Pool: task_score >= baseline (hard; only preserve metrics).
    Extremes:
      - max_metrics: highest score (then |S|, accept)
      - max_skip: highest |S| (then score, accept)
      - max_accept: highest accept (then score, |S|)
    Balanced: among the three unique configs, maximize geometric mean of
      score/max_score, |S|/max_|S|, accept/max_accept (dims with max=0 → 1).
    """
    init_score = _metric(baseline, config.score_key)
    pool: list[dict[str, Any]] = []
    for entry in history:
        if not score_within_tol(
            _metric(entry, config.score_key), init_score, 0.0, "absolute"
        ):
            continue
        pool.append(entry)
    if not pool:
        pool = [history[0]]

    def _best(key_fn):
        return max(pool, key=key_fn)

    max_metrics = _best(
        lambda e: (
            _entry_triple(e, config)[0],
            _entry_triple(e, config)[1],
            _entry_triple(e, config)[2],
        )
    )
    max_skip = _best(
        lambda e: (
            _entry_triple(e, config)[1],
            _entry_triple(e, config)[0],
            _entry_triple(e, config)[2],
        )
    )
    max_accept = _best(
        lambda e: (
            _entry_triple(e, config)[2],
            _entry_triple(e, config)[0],
            _entry_triple(e, config)[1],
        )
    )

    extremes = {
        "max_metrics": max_metrics,
        "max_skip": max_skip,
        "max_accept": max_accept,
    }
    # Deduplicate by skip-set while keeping role labels for summaries.
    unique: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[int, ...]] = set()
    for role, entry in extremes.items():
        key = tuple(entry.get("skip_layers") or [])
        if key in seen:
            continue
        seen.add(key)
        unique.append((role, entry))
    # Always evaluate balance over the three extreme *entries* (may share sets).
    cand_entries = [max_metrics, max_skip, max_accept]
    scores = [_entry_triple(e, config)[0] for e in cand_entries]
    skips = [float(_entry_triple(e, config)[1]) for e in cand_entries]
    accepts = [_entry_triple(e, config)[2] for e in cand_entries]
    max_s = max(scores) if scores else 0.0
    max_n = max(skips) if skips else 0.0
    max_a = max(accepts) if accepts else 0.0

    def _balance(entry: dict[str, Any]) -> float:
        s, n, a = _entry_triple(entry, config)
        ns = (s / max_s) if max_s > 1e-12 else 1.0
        nn = (n / max_n) if max_n > 1e-12 else 1.0
        na = (a / max_a) if max_a > 1e-12 else 1.0
        # Geometric mean; clamp tiny floors so a zero dim does not wipe the pick.
        ns = max(ns, 1e-6)
        nn = max(nn, 1e-6)
        na = max(na, 1e-6)
        return (ns * nn * na) ** (1.0 / 3.0)

    balanced = max(
        cand_entries,
        key=lambda e: (
            _balance(e),
            _entry_triple(e, config)[0],
            _entry_triple(e, config)[1],
            _entry_triple(e, config)[2],
        ),
    )

    candidates = {
        "max_metrics": _summarize_candidate(max_metrics, config, role="max_metrics"),
        "max_skip": _summarize_candidate(max_skip, config, role="max_skip"),
        "max_accept": _summarize_candidate(max_accept, config, role="max_accept"),
        "balanced": _summarize_candidate(balanced, config, role="balanced"),
    }
    candidates["balanced"]["balance_score"] = _balance(balanced)
    for role, entry in (("max_metrics", max_metrics), ("max_skip", max_skip), ("max_accept", max_accept)):
        candidates[role]["balance_score"] = _balance(entry)

    # Mark history roles
    skip_to_roles: dict[tuple[int, ...], list[str]] = {}
    for role, entry in (
        ("max_metrics", max_metrics),
        ("max_skip", max_skip),
        ("max_accept", max_accept),
        ("balanced", balanced),
    ):
        key = tuple(entry.get("skip_layers") or [])
        skip_to_roles.setdefault(key, []).append(role)
    for h in history:
        key = tuple(h.get("skip_layers") or [])
        roles = skip_to_roles.get(key, [])
        h["candidate_roles"] = roles
        h["selected"] = "balanced" in roles

    _ = unique  # retained for possible debug; roles already recorded
    return balanced, candidates


def max_metrics_balanced_search(
    *,
    eval_fn: Callable[[set[int]], dict[str, Any]],
    baseline: dict[str, Any],
    config: SlebSearchConfig,
    on_trial_error: Callable[[int, Exception], None] | None = None,
    beam_width: int = 4,
) -> tuple[set[int], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Metrics-centric beam search; keep max-metrics / max-skip / max-accept; pick balanced.

    Expansion:
      - hard gate: task_score >= baseline (only preserve metrics)
      - rank beam by score desc, then |S|, then accept; prefer latter layers
    Final:
      - retain three extremes among score-preserving history
      - select the most balanced of the three (geometric-mean of normalized axes)
    """
    init_accept = _metric(baseline, config.accept_metric)
    init_score = _metric(baseline, config.score_key)
    if baseline.get("tok_per_s") is None or (
        isinstance(baseline.get("tok_per_s"), float) and math.isnan(baseline["tok_per_s"])
    ):
        baseline = dict(baseline)
        baseline["tok_per_s"] = _toks(baseline)

    history: list[dict[str, Any]] = [
        _history_row(
            phase="baseline",
            rnd=0,
            layer=None,
            skip_layers=[],
            metrics=baseline,
            init_accept=init_accept,
            init_score=init_score,
            config=config,
        )
    ]

    lo = max(0, config.early_barrier)
    hi = config.num_layers - max(0, config.latter_barrier)
    print(
        f"  [max_metrics_balanced] EXPLORE-to-depth={config.max_skip_layers} beam={beam_width}; "
        f"expand if score>={init_score:.4f} (metrics hard); "
        f"final: keep max_metrics / max_skip / max_accept, pick balanced; "
        f"prefer layers [{hi - 1}..{lo}]",
        flush=True,
    )

    beam: list[tuple[frozenset[int], dict[str, Any]]] = [(frozenset(), baseline)]

    for depth in range(1, config.max_skip_layers + 1):
        cand_map: dict[frozenset[int], dict[str, Any]] = {}
        for skip_f, _parent_m in beam:
            skip_set = set(skip_f)
            n_try = 0
            n_keep = 0
            for layer in range(hi - 1, lo - 1, -1):
                if layer in skip_set:
                    continue
                if n_try >= 28 and n_keep >= 3:
                    break
                if n_try >= 40:
                    break
                n_try += 1
                trial_set = set(skip_set)
                trial_set.add(layer)
                key = frozenset(trial_set)
                if key in cand_map:
                    continue
                try:
                    m = eval_fn(trial_set)
                except Exception as exc:
                    if on_trial_error:
                        on_trial_error(layer, exc)
                    continue
                if m.get("tok_per_s") is None:
                    m = dict(m)
                    m["tok_per_s"] = _toks(m)
                # Expansion gate: metrics must not drop below baseline
                if not score_within_tol(
                    _metric(m, config.score_key), init_score, 0.0, "absolute"
                ):
                    continue
                cand_map[key] = m
                n_keep += 1
                history.append(
                    _history_row(
                        phase="explore",
                        rnd=depth,
                        layer=layer,
                        skip_layers=sorted(trial_set),
                        metrics=m,
                        init_accept=init_accept,
                        init_score=init_score,
                        config=config,
                    )
                )

        if not cand_map:
            print(
                f"  [max_metrics_balanced depth {depth}] no metrics-preserving child, stop",
                flush=True,
            )
            break

        ranked = sorted(
            cand_map.items(),
            key=lambda kv: (
                _metric(kv[1], config.score_key)
                if _metric(kv[1], config.score_key) == _metric(kv[1], config.score_key)
                else -1e9,
                len(kv[0]),
                _metric(kv[1], config.accept_metric)
                if _metric(kv[1], config.accept_metric)
                == _metric(kv[1], config.accept_metric)
                else -1e9,
                sum(kv[0]) / max(len(kv[0]), 1),
            ),
            reverse=True,
        )
        beam = [(k, v) for k, v in ranked[: max(1, beam_width)]]
        top = beam[0]
        print(
            f"  [max_metrics_balanced depth {depth}] beam={len(beam)}/{len(cand_map)} "
            f"top skip={sorted(top[0])} score={_metric(top[1], config.score_key):.4f} "
            f"accept={_metric(top[1], config.accept_metric):.3f} tok/s={_toks(top[1]):.1f}",
            flush=True,
        )

    picked, candidates = pick_metrics_extremes_and_balanced(
        history=history, baseline=baseline, config=config
    )
    final_skip = set(picked.get("skip_layers") or [])

    def _metrics_from_pick(entry: dict[str, Any]) -> dict[str, Any]:
        layers = list(entry.get("skip_layers") or [])
        return {
            "skip_layers": layers,
            "num_skip_layers": len(layers),
            config.accept_metric: _metric(entry, config.accept_metric),
            config.score_key: _metric(entry, config.score_key),
            "accept_rate": entry.get("accept_rate"),
            "mean_accepted_per_step": entry.get("mean_accepted_per_step"),
            "tok_per_s": _toks(entry),
            "wall_s": entry.get("wall_s"),
            "total_output_tokens": entry.get("total_output_tokens"),
        }

    if final_skip:
        try:
            final_metrics = eval_fn(final_skip)
            if final_metrics.get("tok_per_s") is None:
                final_metrics = dict(final_metrics)
                final_metrics["tok_per_s"] = _toks(final_metrics)
        except Exception as exc:
            print(f"  [max_metrics_balanced] final re-eval failed ({exc})", flush=True)
            final_metrics = _metrics_from_pick(picked)
    else:
        final_metrics = dict(baseline)
        final_metrics["skip_layers"] = []
        final_metrics["num_skip_layers"] = 0
        final_metrics["tok_per_s"] = _toks(baseline)

    print(
        f"  [max_metrics_balanced] CANDIDATES "
        f"metrics={candidates['max_metrics']['skip_layers']} "
        f"skip={candidates['max_skip']['skip_layers']} "
        f"accept={candidates['max_accept']['skip_layers']}",
        flush=True,
    )
    print(
        f"  [max_metrics_balanced] SELECTED(balanced) skip={sorted(final_skip)} "
        f"(|S|={len(final_skip)}) "
        f"score={_metric(final_metrics, config.score_key):.4f} "
        f"accept={_metric(final_metrics, config.accept_metric):.3f} "
        f"tok/s={_toks(final_metrics):.1f} "
        f"balance={candidates['balanced'].get('balance_score', float('nan')):.4f}",
        flush=True,
    )
    return final_skip, history, final_metrics, candidates
