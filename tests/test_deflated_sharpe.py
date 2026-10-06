"""How much of a Sharpe ratio is luck: PSR, expected max Sharpe, DSR."""

from __future__ import annotations

import numpy as np
import pytest

from honest_backtest.metrics import (
    annualised_sharpe,
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe,
)


def test_psr_is_near_certain_for_a_strong_edge_and_a_coin_flip_for_none():
    rng = np.random.default_rng(0)
    strong = rng.normal(0.002, 0.01, 2000)
    none = rng.normal(0.0, 0.01, 2000)
    assert probabilistic_sharpe(strong) > 0.99
    assert 0.02 < probabilistic_sharpe(none) < 0.98
    assert probabilistic_sharpe(-strong) < 0.01


def test_negative_skew_makes_the_same_sharpe_less_convincing():
    rng = np.random.default_rng(1)
    crashy = rng.normal(0.0015, 0.01, 1000)
    crashy[::100] -= 0.06  # rare large losses: negative skew, fat tails
    symmetric = rng.normal(0.0, 1.0, 1000)
    # Rescale so both series have exactly the same mean and stdev.
    symmetric = (symmetric - symmetric.mean()) / symmetric.std() * crashy.std()
    symmetric += crashy.mean()
    assert annualised_sharpe(symmetric) == pytest.approx(annualised_sharpe(crashy))
    assert probabilistic_sharpe(crashy) < probabilistic_sharpe(symmetric)


def test_expected_max_sharpe_grows_with_the_number_of_trials():
    assert expected_max_sharpe(1, 1.0) == 0.0
    values = [expected_max_sharpe(n, 1.0) for n in (2, 10, 100, 1000)]
    assert values == sorted(values)
    assert expected_max_sharpe(10, 4.0) == pytest.approx(2 * expected_max_sharpe(10, 1.0))


def test_expected_max_sharpe_matches_a_simulation():
    """Best of N zero-edge strategies, simulated, against the closed form."""
    rng = np.random.default_rng(2)
    n_trials, n_obs, n_rep = 50, 500, 200
    maxima, variances = [], []
    for _ in range(n_rep):
        rets = rng.normal(0.0, 1.0, (n_trials, n_obs))
        srs = rets.mean(axis=1) / rets.std(axis=1, ddof=1)
        maxima.append(srs.max())
        variances.append(srs.var(ddof=1))
    predicted = expected_max_sharpe(n_trials, float(np.mean(variances)))
    assert np.mean(maxima) == pytest.approx(predicted, rel=0.1)


def test_dsr_penalises_selection_and_reduces_to_psr_for_one_trial():
    rng = np.random.default_rng(3)
    rets = rng.normal(0.0006, 0.01, 1500)
    sr = annualised_sharpe(rets)
    assert deflated_sharpe(rets, [sr]) == pytest.approx(probabilistic_sharpe(rets))
    many = list(rng.normal(0.0, 1.0, 99)) + [sr]
    assert deflated_sharpe(rets, many) < probabilistic_sharpe(rets)


def test_degenerate_inputs_are_uninformative_not_errors():
    assert probabilistic_sharpe(np.zeros(100)) == 0.5
    assert probabilistic_sharpe(np.array([0.01, 0.02])) == 0.5
    assert 0.0 <= deflated_sharpe(np.zeros(100), [0.0, 0.0]) <= 1.0


def test_every_rung_reports_a_probability():
    from honest_backtest.experiments import StudyConfig, run_study
    from honest_backtest.synthetic import SyntheticConfig

    study = run_study(
        StudyConfig(
            synthetic=SyntheticConfig(n_bars=1008),
            seed=7,
            lookback_grid=(5, 20, 60),
            train_size=252,
            test_size=252,
        )
    )
    for stage in study.stages:
        assert 0.0 <= stage.p_edge <= 1.0, stage.name
    # The look-ahead rung is near certain; the honest rung is far from it.
    assert study.stages[0].p_edge > study.stages[-1].p_edge
