from __future__ import annotations

from collections import Counter

from spec_exp.sleb_skip_search import SlebSearchConfig
from spec_exp.tri_objective_search import (
    TriObjectiveOptions,
    pick_three_win,
    three_win_feasible,
    tri_objective_v3_search,
)


def metrics(skip: set[int], score: float, accept: float, speed: float) -> dict:
    return {
        "skip_layers": sorted(skip),
        "task_score": score,
        "mean_accepted_per_step": accept,
        "accept_rate": accept / 10,
        "tok_per_s": speed,
    }


def config() -> SlebSearchConfig:
    return SlebSearchConfig(
        num_layers=8,
        max_skip_layers=3,
        early_barrier=1,
        latter_barrier=1,
        score_key="task_score",
        accept_metric="mean_accepted_per_step",
    )


def test_three_win_gate_and_native_fallback() -> None:
    cfg = config()
    baseline = metrics(set(), 1.0, 2.0, 50.0)
    score_only = metrics({2}, 1.2, 1.9, 60.0)
    assert not three_win_feasible(score_only, baseline, cfg)
    picked, ranked = pick_three_win(
        [score_only, baseline],
        baseline=baseline,
        config=cfg,
        options=TriObjectiveOptions(),
    )
    assert picked["skip_layers"] == []
    assert ranked == [baseline]


def test_accept_beam_crosses_score_valley_to_synergy_set() -> None:
    cfg = config()
    baseline = metrics(set(), 1.0, 2.0, 50.0)
    calls: Counter[tuple[int, ...]] = Counter()

    def evaluate(skip: set[int]) -> dict:
        key = tuple(sorted(skip))
        calls[key] += 1
        if skip == {2}:
            return metrics(skip, 0.80, 2.8, 60.0)
        if skip == {2, 3}:
            return metrics(skip, 0.75, 3.0, 65.0)
        if skip == {2, 3, 4}:
            return metrics(skip, 1.30, 3.5, 80.0)
        return metrics(skip, 1.01, 2.05, 52.0)

    selected, history, final, candidates, metadata = tri_objective_v3_search(
        eval_fn=evaluate,
        baseline=baseline,
        config=cfg,
        options=TriObjectiveOptions(
            metrics_beam_width=1,
            accept_beam_width=2,
            speed_beam_width=1,
            refine_top_k=2,
            refine_max_evals=4,
            seed_sets=((2, 3, 4),),
        ),
    )
    assert selected == {2, 3, 4}
    assert final["task_score"] > baseline["task_score"]
    assert final["mean_accepted_per_step"] > baseline["mean_accepted_per_step"]
    assert final["tok_per_s"] > baseline["tok_per_s"]
    assert any(
        row["skip_layers"] == [2] and not row["three_win_feasible"]
        for row in history
    )
    assert candidates["selected"]["skip_layers"] == [2, 3, 4]
    assert metadata["eval_stats"]["seed_sets_evaluated"] == 1
    assert max(calls.values()) <= 2


def test_known_seed_is_evaluated_even_when_prefix_is_pruned() -> None:
    cfg = config()
    baseline = metrics(set(), 1.0, 2.0, 50.0)

    def evaluate(skip: set[int]) -> dict:
        if skip == {2, 4, 6}:
            return metrics(skip, 1.2, 3.0, 75.0)
        return metrics(skip, 0.2, 1.0, 40.0)

    selected, _, _, _, metadata = tri_objective_v3_search(
        eval_fn=evaluate,
        baseline=baseline,
        config=cfg,
        options=TriObjectiveOptions(
            metrics_beam_width=1,
            accept_beam_width=1,
            speed_beam_width=1,
            refine_max_evals=0,
            seed_sets=((2, 4, 6),),
        ),
    )
    assert selected == {2, 4, 6}
    assert metadata["eval_stats"]["seed_sets_evaluated"] == 1
