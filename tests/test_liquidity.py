"""Partial fills under a volume cap, and passive limit orders."""

from __future__ import annotations

import pytest
from helpers import FixedWeightStrategy

from honest_backtest.commission import PerShareCommission
from honest_backtest.data import Bar, bars_from_series
from honest_backtest.engine import run_backtest
from honest_backtest.slippage import FixedBpsSlippage
from honest_backtest.strategy import TimeSeriesMomentumStrategy
from honest_backtest.synthetic import SyntheticConfig, generate_price_series


def _flat_bars(n=6, price=100.0, volume=1_000.0, low=None, high=None):
    return [
        Bar(
            i,
            "SYNTH",
            price,
            price if high is None else high,
            price if low is None else low,
            price,
            volume,
        )
        for i in range(n)
    ]


def _run(bars, weights, **kwargs):
    return run_backtest(
        bars,
        lambda events, data: FixedWeightStrategy(events, data, list(weights)),
        rebalance_threshold=0.0,
        initial_capital=1_000_000.0,
        **kwargs,
    )


def test_a_volume_cap_splits_an_order_across_bars():
    # Target 10,000 units at 100; each bar trades 1,000, of which 25% (250) is
    # ours, so the order takes 40 bars.
    result = _run(_flat_bars(n=50), [1.0], max_participation=0.25)
    quantities = [f.quantity for f in result.fills]
    assert all(q <= 250.0 + 1e-9 for q in quantities)
    assert sum(quantities) == pytest.approx(10_000.0)
    assert [f.timestamp for f in result.fills] == list(range(len(quantities)))


def test_without_a_cap_the_order_fills_at_once():
    result = _run(_flat_bars(), [1.0])
    assert len(result.fills) == 1


def test_a_new_order_cancels_the_leftover_instead_of_trading_it_twice():
    # Bar 0 asks for +1.0, bar 1 changes its mind to flat.
    result = _run(_flat_bars(n=8), [1.0, 0.0], max_participation=0.25)
    assert result.final_position == pytest.approx(0.0, abs=1e-9)
    assert result.unfilled_quantity > 0


def test_capped_runs_still_reconcile_against_the_blotter():
    bars = generate_price_series(SyntheticConfig(n_bars=500), seed=2).to_bars()
    result = run_backtest(
        bars,
        lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=10),
        slippage=FixedBpsSlippage(2.0),
        commission=PerShareCommission(),
        max_participation=0.001,
    )
    cash, position = result.initial_capital, 0.0
    for fill in result.fills:
        cash -= fill.signed_quantity * fill.fill_price + fill.commission
        position += fill.signed_quantity
    assert position == pytest.approx(result.final_position, abs=1e-9)
    assert cash == pytest.approx(result.final_cash, abs=1e-6)


def test_limit_orders_fill_only_when_price_trades_through():
    # The limit sits 10bp below a flat 100 close. Bars that only touch it
    # never fill; a bar whose low goes through does, at the limit.
    touching = _flat_bars(n=4, low=99.9)
    result = _run(touching, [1.0], limit_offset_bps=10.0, limit_expiry_bars=3)
    assert result.fills == []
    assert result.unfilled_quantity == pytest.approx(10_000.0)

    through = _flat_bars(n=4, low=99.0)
    result = _run(through, [1.0], limit_offset_bps=10.0)
    (fill,) = result.fills
    assert fill.timestamp == 1  # rests from the next bar, never the same one
    assert fill.fill_price == pytest.approx(99.9)
    assert fill.slippage_cost == 0.0


def test_sell_limits_need_a_high_through_the_limit():
    bars = _flat_bars(n=4, high=101.0)
    result = _run(bars, [-1.0], limit_offset_bps=10.0)
    (fill,) = result.fills
    assert fill.direction == "SELL"
    assert fill.fill_price == pytest.approx(100.1)


def test_limit_orders_expire():
    # Price only trades through on bar 3; an order resting for one bar
    # (bar 1) misses it, one resting for three bars (bars 1 to 3) gets it.
    bars = _flat_bars(n=5)
    bars[3] = Bar(3, "SYNTH", 100.0, 100.0, 98.0, 100.0, 1_000.0)
    short = _run(bars, [1.0], limit_offset_bps=10.0, limit_expiry_bars=1)
    long = _run(bars, [1.0], limit_offset_bps=10.0, limit_expiry_bars=3)
    assert short.fills == []
    assert [f.timestamp for f in long.fills] == [3]


def test_passive_execution_saves_spread_but_misses_trades():
    bars = bars_from_series(
        generate_price_series(SyntheticConfig(n_bars=500), seed=4).closes
    )

    def factory(e, d):
        return TimeSeriesMomentumStrategy(e, d, lookback=10)

    market = run_backtest(bars, factory, slippage=FixedBpsSlippage(5.0))
    passive = run_backtest(
        bars, factory, slippage=FixedBpsSlippage(5.0), limit_offset_bps=2.0
    )
    assert passive.total_slippage == 0.0
    assert market.total_slippage > 0.0
    assert passive.unfilled_quantity > 0.0
