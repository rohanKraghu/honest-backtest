"""The ladder must run on any strategy, not only the built-in one."""

from __future__ import annotations

import pytest

from honest_backtest.commission import PercentOfNotionalCommission
from honest_backtest.experiments import (
    LadderSettings,
    StudyConfig,
    momentum_spec,
    run_ladder,
    run_study,
)
from honest_backtest.slippage import FixedBpsSlippage
from honest_backtest.spec import StrategySpec, format_params, param_grid
from honest_backtest.strategy import BuyAndHoldStrategy
from honest_backtest.synthetic import SyntheticConfig, generate_price_series

SETTINGS = LadderSettings(train_size=252, test_size=252)


@pytest.fixture(scope="module")
def bars():
    """A short synthetic path, shared across tests."""
    return generate_price_series(SyntheticConfig(n_bars=1008), seed=7).to_bars()


def test_param_grid_is_the_ordered_cartesian_product():
    grid = param_grid(fast=(5, 10), slow=(50, 100))
    assert grid == [
        {"fast": 5, "slow": 50},
        {"fast": 5, "slow": 100},
        {"fast": 10, "slow": 50},
        {"fast": 10, "slow": 100},
    ]
    assert param_grid() == [{}]


def test_format_params_is_bare_for_one_key():
    assert format_params({"lookback": 20}) == "20"
    assert format_params({"fast": 5, "slow": 50}) == "fast=5,slow=50"


def test_an_empty_grid_is_rejected():
    with pytest.raises(ValueError, match="empty parameter grid"):
        StrategySpec(name="x", build=lambda *a, **k: None, grid=[])


def test_a_spec_without_a_leaky_twin_cannot_be_asked_to_leak():
    spec = StrategySpec(name="bh", build=lambda *a, **k: None, grid=[{}])
    with pytest.raises(ValueError, match="no look-ahead variant"):
        spec.factory({}, look_ahead=True)


def test_a_strategy_without_a_leaky_twin_gets_a_four_rung_ladder(bars):
    """No look-ahead rung is invented for a strategy that cannot leak."""
    honest_only = momentum_spec((5, 20))
    spec = StrategySpec(
        name="momentum, honest only",
        build=honest_only.build,
        grid=honest_only.grid,
        warmup=honest_only.warmup,
    )
    ladder = run_ladder(
        bars,
        spec,
        SETTINGS,
        slippage=FixedBpsSlippage(2.0),
        commission=PercentOfNotionalCommission(1.0),
    )
    assert [s.name for s in ladder.stages] == [
        "In-sample, no costs",
        "+ slippage",
        "+ commissions",
        "+ walk-forward OOS",
    ]
    assert [s.honest for s in ladder.stages] == [False, False, False, True]
    assert len(ladder.stages[-1].chosen_params) == len(ladder.folds)
    assert ladder.stages[1].metrics.sharpe < ladder.stages[0].metrics.sharpe


def test_a_parameterless_strategy_runs_through_the_ladder(bars):
    spec = StrategySpec(
        name="buy and hold",
        build=lambda events, data, symbol: BuyAndHoldStrategy(
            events, data, symbol=symbol
        ),
        grid=param_grid(),
    )
    ladder = run_ladder(
        bars,
        spec,
        SETTINGS,
        slippage=FixedBpsSlippage(2.0),
        commission=PercentOfNotionalCommission(1.0),
    )
    # Buy and hold has nothing to fit, so walk-forward cannot change it much,
    # and the ladder's own benchmark is the same strategy.
    assert ladder.stages[-1].metrics.sharpe == pytest.approx(
        ladder.buy_hold_sharpe, abs=0.05
    )


def test_the_built_in_study_is_the_generic_ladder_on_the_momentum_spec():
    """run_study is a thin wrapper: same stages, same numbers."""
    cfg = StudyConfig(
        synthetic=SyntheticConfig(n_bars=1008),
        seed=7,
        lookback_grid=(5, 20),
        train_size=252,
        test_size=252,
    )
    study = run_study(cfg)
    ladder = run_ladder(
        generate_price_series(cfg.synthetic, seed=cfg.seed).to_bars(),
        momentum_spec(cfg.lookback_grid),
        cfg.settings(),
        slippage=cfg.slippage(),
        commission=cfg.commission(),
    )
    assert [s.metrics for s in study.stages] == [s.metrics for s in ladder.stages]
    assert study.buy_hold_sharpe == ladder.buy_hold_sharpe
