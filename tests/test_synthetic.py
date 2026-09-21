"""The synthetic generator must be reproducible, and its edge must be real.

If the seed does not reproduce, none of the headline numbers can be checked.
If the injected edge is not actually there, the study is measuring noise and
the "degradation" would be meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from honest_backtest.synthetic import (
    SyntheticConfig,
    generate_price_series,
    oracle_sharpe,
)


def test_the_same_seed_reproduces_exactly():
    a = generate_price_series(SyntheticConfig(n_bars=500), seed=7)
    b = generate_price_series(SyntheticConfig(n_bars=500), seed=7)
    assert np.array_equal(a.closes, b.closes)
    assert np.array_equal(a.volumes, b.volumes)
    assert np.array_equal(a.latent_state, b.latent_state)


def test_different_seeds_give_different_paths():
    a = generate_price_series(SyntheticConfig(n_bars=500), seed=7)
    b = generate_price_series(SyntheticConfig(n_bars=500), seed=8)
    assert not np.array_equal(a.closes, b.closes)


def test_prices_stay_positive():
    s = generate_price_series(SyntheticConfig(n_bars=2000, annual_vol=0.6), seed=3)
    assert (s.closes > 0).all()
    assert (s.volumes > 0).all()


def test_realised_volatility_is_close_to_the_target():
    cfg = SyntheticConfig(n_bars=20_000, annual_vol=0.25, signal_alpha=0.0)
    s = generate_price_series(cfg, seed=11)
    realised = s.log_returns[1:].std(ddof=1) * np.sqrt(cfg.bars_per_year)
    assert realised == pytest.approx(0.25, rel=0.05)


def test_the_injected_edge_is_actually_present():
    """The latent state must predict the NEXT bar's return, not the current one."""
    cfg = SyntheticConfig(n_bars=20_000, signal_alpha=0.2)
    s = generate_price_series(cfg, seed=5)
    forward = np.corrcoef(s.latent_state[:-1], s.log_returns[1:])[0, 1]
    contemporaneous = np.corrcoef(s.latent_state[1:], s.log_returns[1:])[0, 1]
    assert forward > 0.1, "the injected edge should be forward-looking"
    assert forward > contemporaneous


def test_zero_alpha_means_no_edge():
    cfg = SyntheticConfig(n_bars=20_000, signal_alpha=0.0)
    s = generate_price_series(cfg, seed=5)
    assert abs(oracle_sharpe(s)) < 0.3


def test_the_edge_is_weak_by_design():
    """The oracle ceiling must be modest; a huge edge would prove nothing."""
    s = generate_price_series(SyntheticConfig(n_bars=20_000), seed=5)
    assert 0.5 < oracle_sharpe(s) < 2.5


def test_oracle_beats_a_strategy_that_cannot_see_the_state():
    s = generate_price_series(SyntheticConfig(n_bars=20_000), seed=5)
    rng = np.random.default_rng(0)
    blind = np.sign(rng.standard_normal(len(s) - 1)) * s.log_returns[1:]
    blind_sharpe = blind.mean() / blind.std(ddof=1) * np.sqrt(252)
    assert oracle_sharpe(s) > blind_sharpe


def test_bars_round_trip_the_close_series():
    s = generate_price_series(SyntheticConfig(n_bars=50), seed=2)
    bars = s.to_bars()
    assert len(bars) == 50
    assert [b.close for b in bars] == pytest.approx(s.closes.tolist())
    assert [b.timestamp for b in bars] == list(range(50))
    for bar in bars:
        assert bar.low <= bar.close <= bar.high
        assert bar.low <= bar.open <= bar.high


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        SyntheticConfig(signal_persistence=1.0)
    with pytest.raises(ValueError):
        SyntheticConfig(n_bars=1)
    with pytest.raises(ValueError):
        SyntheticConfig(annual_vol=0.0)
