from spec_exp.sleb_skip_search import SlebSearchConfig, max_skip_latter_search


def test_score_first_ignores_acceptance_and_honors_minimum_depth() -> None:
    harm = {0: 0.5, 1: 0.4, 2: 0.1, 3: 0.2}

    def evaluate(skip_layers: set[int]) -> dict[str, float]:
        return {
            "task_score": 1.0 - sum(harm[layer] for layer in skip_layers),
            "mean_accepted_per_step": 5.0 - len(skip_layers) * 3.0,
            "tok_per_s": 10.0 + len(skip_layers),
        }

    baseline = evaluate(set())
    selected, _, metrics = max_skip_latter_search(
        eval_fn=evaluate,
        baseline=baseline,
        config=SlebSearchConfig(
            num_layers=4,
            max_skip_layers=3,
            early_barrier=0,
            latter_barrier=0,
        ),
        beam_width=3,
        min_skips=2,
        score_first=True,
    )

    assert selected == {2, 3}
    assert metrics["task_score"] == 0.7
