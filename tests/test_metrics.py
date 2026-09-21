"""Performance statistics, checked against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from honest_backtest.metrics import (
    annualised_return,
    annualised_sharpe,
    compute_metrics,
    max_drawdown,
    simple_returns,
    total_return,
)


def test_simple_returns_are_period_over_period():
    assert simple_returns([100.0, 110.0, 99.0]) == pytest.approx([0.10, -0.10])


def test_simple_returns_of_a_degenerate_curve():
    assert simple_returns([100.0]).size == 0
    assert simple_returns([]).size == 0


def test_sharpe_of_a_constant_return_series_is_undefined_not_infinite():
    """Zero variance must give 0.0, not a division by zero."""
    assert annualised_sharpe(np.full(100, 0.001)) == 0.0


def test_sharpe_matches_the_definition():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 1000)
    expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert annualised_sharpe(r) == pytest.approx(expected)


def test_sharpe_sign_follows_the_mean():
    rng = np.random.default_rng(1)
    r = rng.normal(-0.001, 0.01, 500)
    assert annualised_sharpe(r) < 0


def test_max_drawdown_on_a_hand_checked_curve():
    # Peak 120, trough 60 -> -50%.
    assert max_drawdown([100.0, 120.0, 60.0, 110.0]) == pytest.approx(-0.5)


def test_max_drawdown_of_a_monotone_curve_is_zero():
    assert max_drawdown([1.0, 2.0, 3.0]) == 0.0


def test_total_and_annualised_return():
    equity = [100.0] + [100.0 * 1.5]
    assert total_return(equity) == pytest.approx(0.5)
    # One bar of 252 -> annualising a 50% move gives a very large number.
    assert annualised_return(equity, bars_per_year=252) > 1.0
    # Exactly one year of bars: annualised equals total.
    year = list(np.linspace(100.0, 150.0, 253))
    assert annualised_return(year, 252) == pytest.approx(0.5, rel=1e-6)


def test_annualised_return_of_a_wiped_out_account_is_defined():
    assert annualised_return([100.0, 0.0]) == 0.0


def test_compute_metrics_bundles_everything():
    equity = [100.0, 110.0, 105.0, 130.0]
    m = compute_metrics(equity, bars_per_year=252, n_trades=3, commission_cost=7.0)
    assert m.n_periods == 3
    assert m.n_trades == 3
    assert m.commission_cost == 7.0
    assert m.total_return == pytest.approx(0.3)
    assert m.max_drawdown < 0


def test_stitched_returns_do_not_invent_a_jump():
    """Walk-forward metrics must be built from returns, not glued equity curves.

    Two folds that each gain 10% give a 21% compounded out-of-sample return.
    Naively concatenating their equity curves would imply something else.
    """
    fold_a = np.array([0.10])
    fold_b = np.array([0.10])
    stitched = np.concatenate([fold_a, fold_b])
    curve = np.concatenate([[1.0], np.cumprod(1.0 + stitched)])
    m = compute_metrics(curve, returns=stitched)
    assert m.total_return == pytest.approx(0.21)
    assert m.n_periods == 2


def test_turnover_is_reported_per_year():
    equity = [1_000_000.0] * 253  # one year of flat equity
    m = compute_metrics(equity, bars_per_year=252, traded_notional=2_000_000.0)
    assert m.annual_turnover == pytest.approx(2.0, rel=1e-6)
