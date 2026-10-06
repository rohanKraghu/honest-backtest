"""The fast path must give the event engine's answer exactly, not approximately."""

from __future__ import annotations

import pytest
from helpers import FixedWeightStrategy

from honest_backtest.commission import PercentOfNotionalCommission, PerShareCommission
from honest_backtest.data import Bar
from honest_backtest.engine import run_backtest
from honest_backtest.experiments import (
    LadderSettings,
    StudyConfig,
    momentum_spec,
    run_study,
)
from honest_backtest.fast import SignalCache, replay_signals, simulate, supports
from honest_backtest.slippage import (
    FixedBpsSlippage,
    PermanentImpactSlippage,
    SpreadPlusImpactSlippage,
)
from honest_backtest.spec import StrategySpec, param_grid
from honest_backtest.synthetic import SyntheticConfig, generate_price_series


@pytest.fixture(scope="module")
def bars():
    """Synthetic bars with opens that differ from the previous close."""
    raw = generate_price_series(SyntheticConfig(n_bars=700), seed=21).to_bars()
    out = []
    for i, b in enumerate(raw):
        o = b.open * (1.0 + 0.002 * ((i % 7) - 3) / 3) if i else b.open
        out.append(
            Bar(
                b.timestamp,
                b.symbol,
                o,
                max(o, b.close),
                min(o, b.close),
                b.close,
                b.volume,
            )
        )
    return out


def _same(fast, slow):
    assert fast.equity == slow.equity
    assert fast.positions == slow.positions
    assert fast.final_cash == slow.final_cash
    assert fast.final_position == slow.final_position
    assert fast.total_commission == slow.total_commission
    assert fast.total_slippage == slow.total_slippage
    assert fast.traded_notional == slow.traded_notional
    assert fast.n_trades == slow.n_trades
    assert [
        (f.timestamp, f.quantity, f.direction, f.fill_price, f.commission)
        for f in fast.fills
    ] == [
        (f.timestamp, f.quantity, f.direction, f.fill_price, f.commission)
        for f in slow.fills
    ]
    assert fast.metrics().sharpe == slow.metrics().sharpe
    assert fast.used_look_ahead == slow.used_look_ahead


COSTS = [
    (None, None),
    (FixedBpsSlippage(3.0), PerShareCommission()),
    (SpreadPlusImpactSlippage(), PercentOfNotionalCommission(1.0)),
    (PermanentImpactSlippage(permanent_coefficient=0.5), PerShareCommission()),
]


@pytest.mark.parametrize("timing", ["close", "next_open", "next_close"])
@pytest.mark.parametrize("costs", COSTS, ids=["free", "fixed", "impact", "permanent"])
@pytest.mark.parametrize("look_ahead", [False, True], ids=["honest", "leaky"])
def test_the_fast_path_equals_the_engine(bars, timing, costs, look_ahead):
    spec = momentum_spec((10,))
    params = spec.grid[0]
    slippage, commission = costs
    kwargs = dict(
        slippage=slippage,
        commission=commission,
        initial_capital=500_000.0,
        rebalance_threshold=0.05,
        warmup=100,
        fill_timing=timing,
    )
    slow = run_backtest(
        bars,
        spec.factory(params, look_ahead=look_ahead),
        allow_look_ahead=look_ahead,
        symbol=bars[0].symbol,
        **kwargs,
    )
    signals, leaked = replay_signals(bars, spec, params, look_ahead=look_ahead)
    fast = simulate(bars, signals, used_look_ahead=leaked, **kwargs)
    _same(fast, slow)


def test_several_signals_on_one_bar_are_handled_like_the_engine(bars):
    class TwoPerBar(FixedWeightStrategy):
        def calculate_signals(self, event):
            super().calculate_signals(event)
            super().calculate_signals(event)

    spec = StrategySpec(
        name="two",
        build=lambda e, d, s: TwoPerBar(e, d, [0.5, -0.5, 1.0, 0.2] * 50, symbol=s),
        grid=param_grid(),
    )
    for timing in ("close", "next_close"):
        slow = run_backtest(
            bars, spec.factory({}), symbol=bars[0].symbol, fill_timing=timing
        )
        signals, _ = replay_signals(bars, spec, {})
        fast = simulate(bars, signals, fill_timing=timing)
        _same(fast, slow)
        assert fast.unfilled_quantity == slow.unfilled_quantity


def test_the_cache_replays_each_setting_once(bars):
    spec = momentum_spec((5, 10))
    cache = SignalCache()
    for _ in range(3):
        for params in spec.grid:
            cache.get(bars, spec, params, False)
    assert cache.misses == 2
    assert cache.hits == 4


def test_frictions_the_fast_path_does_not_model_are_declined():
    assert supports(fill_timing="next_open")
    assert not supports(max_participation=0.1)
    assert not supports(limit_offset_bps=5.0)
    assert not supports(financing=object())


def test_a_fast_study_is_identical_to_the_engine_study():
    cfg = StudyConfig(
        synthetic=SyntheticConfig(n_bars=1008),
        seed=7,
        lookback_grid=(5, 20),
        train_size=252,
        test_size=252,
    )
    slow = run_study(cfg)
    fast = run_study(cfg, fast=True)
    assert [s.metrics for s in fast.stages] == [s.metrics for s in slow.stages]
    assert [s.chosen_params for s in fast.stages] == [
        s.chosen_params for s in slow.stages
    ]
    assert [s.p_edge for s in fast.stages] == [s.p_edge for s in slow.stages]
    assert fast.buy_hold_sharpe == slow.buy_hold_sharpe
    assert LadderSettings().fast is False


def test_a_parallel_sweep_equals_a_serial_one():
    from honest_backtest.experiments import run_seed_sweep

    cfg = StudyConfig(
        synthetic=SyntheticConfig(n_bars=1008),
        seed=3,
        lookback_grid=(5, 20),
        train_size=252,
        test_size=252,
    )
    serial = run_seed_sweep(3, cfg)
    parallel = run_seed_sweep(3, cfg, fast=True, workers=3)
    assert parallel.seeds == serial.seeds
    assert parallel.sharpes == serial.sharpes
    assert parallel.monotone_flags == serial.monotone_flags
