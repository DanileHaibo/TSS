from __future__ import annotations

from collections import Counter

from spec_exp.pareto_bridge_search import (
    ParetoBridgeOptions,
    allowed_layers,
    pareto_bridge_v2_search,
    pareto_frontier_3d,
)
from spec_exp.sleb_skip_search import SlebSearchConfig, max_metrics_balanced_search


def _metrics(skip: set[int], *, score: float, accept: float) -> dict:
    return {
        "skip_layers": sorted(skip),
        "task_score": score,
        "mean_accepted_per_step": accept,
        "accept_rate": accept / 10.0,
        "tok_per_s": 50.0 + 5.0 * len(skip),
    }


def test_allowed_layers_is_full_and_respects_barriers() -> None:
    cfg = SlebSearchConfig(
        num_layers=10,
        max_skip_layers=3,
        early_barrier=2,
        latter_barrier=1,
        layer_step=1,
    )
    assert allowed_layers(cfg) == [8, 7, 6, 5, 4, 3, 2]


def test_pareto_frontier_removes_dominated_points() -> None:
    cfg = SlebSearchConfig(
        num_layers=8,
        max_skip_layers=3,
        score_key="task_score",
        accept_metric="mean_accepted_per_step",
    )
    entries = [
        _metrics({1}, score=1.0, accept=2.0),
        _metrics({1, 2}, score=1.0, accept=2.0),
        _metrics({3}, score=1.2, accept=1.5),
        _metrics({4}, score=0.9, accept=1.0),
    ]
    frontier = pareto_frontier_3d(entries, cfg)
    keys = {tuple(x["skip_layers"]) for x in frontier}
    assert keys == {(1, 2), (3,)}


def test_bridge_allows_non_monotonic_pair_and_final_is_strict() -> None:
    cfg = SlebSearchConfig(
        num_layers=6,
        max_skip_layers=2,
        early_barrier=1,
        latter_barrier=1,
        score_key="task_score",
        accept_metric="mean_accepted_per_step",
    )
    baseline = _metrics(set(), score=1.0, accept=2.0)
    calls: Counter[tuple[int, ...]] = Counter()

    def evaluate(skip: set[int]) -> dict:
        key = tuple(sorted(skip))
        calls[key] += 1
        if len(skip) == 1:
            return _metrics(skip, score=0.96, accept=2.0 + 0.01 * sum(skip))
        if skip == {3, 4}:
            return _metrics(skip, score=1.20, accept=2.3)
        return _metrics(skip, score=1.01, accept=2.1)

    selected, history, final, candidates, metadata = pareto_bridge_v2_search(
        eval_fn=evaluate,
        baseline=baseline,
        config=cfg,
        options=ParetoBridgeOptions(
            bridge_score_drop_tol=0.05,
            bridge_score_tol_mode="relative",
            beam_width_per_objective=2,
            refine_top_k=3,
            refine_max_evals=8,
        ),
    )

    assert metadata["coverage"]["single_layers_evaluated"] == 4
    assert any(
        row["bridge_feasible"] and not row["final_feasible"] for row in history
    )
    assert final["task_score"] >= baseline["task_score"]
    assert selected
    assert candidates["balanced"]["task_score"] >= baseline["task_score"]
    assert max(calls.values()) <= 2
    assert metadata["eval_stats"]["trials_cache_hits"] > 0


def test_v1_search_remains_callable() -> None:
    cfg = SlebSearchConfig(
        num_layers=6,
        max_skip_layers=1,
        early_barrier=1,
        latter_barrier=1,
        score_key="task_score",
        accept_metric="mean_accepted_per_step",
    )
    baseline = _metrics(set(), score=1.0, accept=2.0)

    def evaluate(skip: set[int]) -> dict:
        return _metrics(skip, score=1.0, accept=2.0 + 0.1 * sum(skip))

    selected, history, final, candidates = max_metrics_balanced_search(
        eval_fn=evaluate,
        baseline=baseline,
        config=cfg,
        beam_width=2,
    )
    assert len(selected) == 1
    assert history
    assert final["task_score"] == 1.0
    assert candidates["balanced"]["skip_layers"] == sorted(selected)
