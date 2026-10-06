"""Permanent impact: trading the same way repeatedly walks the price against you."""

from __future__ import annotations

from dataclasses import replace

import pytest

from honest_backtest.data import Bar
from honest_backtest.engine import run_backtest
from honest_backtest.events import OrderEvent
from honest_backtest.slippage import PermanentImpactSlippage, SpreadPlusImpactSlippage
from honest_backtest.strategy import TimeSeriesMomentumStrategy
from honest_backtest.synthetic import SyntheticConfig, generate_price_series

BAR = Bar(
    timestamp=0,
    symbol="SYNTH",
    open=100.0,
    high=101.0,
    low=99.0,
    close=100.0,
    volume=1_000_000.0,
)


def _order(direction: str, qty: float = 100_000.0) -> OrderEvent:
    return OrderEvent(symbol="SYNTH", timestamp=0, quantity=qty, direction=direction)


def _at(t: int) -> Bar:
    return replace(BAR, timestamp=t)


def test_first_fill_matches_the_temporary_model():
    perm = PermanentImpactSlippage(permanent_coefficient=0.5)
    temp = SpreadPlusImpactSlippage()
    assert perm.fill_price(_order("BUY"), BAR) == pytest.approx(
        temp.fill_price(_order("BUY"), BAR)
    )


def test_repeated_buys_pay_more_each_time():
    model = PermanentImpactSlippage(permanent_coefficient=0.5, half_life_bars=5.0)
    prices = [model.fill_price(_order("BUY"), _at(t)) for t in range(4)]
    assert prices == sorted(prices)
    assert prices[-1] > prices[0]


def test_the_push_decays_with_the_half_life():
    model = PermanentImpactSlippage(permanent_coefficient=0.5, half_life_bars=4.0)
    model.fill_price(_order("BUY"), _at(0))
    push0 = model.outstanding_push(0)
    assert model.outstanding_push(4) == pytest.approx(push0 / 2)
    assert model.outstanding_push(8) == pytest.approx(push0 / 4)


def test_reversing_is_never_paid_back():
    model = PermanentImpactSlippage(permanent_coefficient=2.0)
    for t in range(3):
        model.fill_price(_order("BUY"), _at(t))
    # Selling into your own push may not fill above the reference close.
    assert model.fill_price(_order("SELL"), _at(3)) <= BAR.close


def test_reset_forgets_previous_trades():
    model = PermanentImpactSlippage(permanent_coefficient=0.5)
    first = model.fill_price(_order("BUY"), _at(0))
    model.fill_price(_order("BUY"), _at(1))
    model.reset()
    assert model.fill_price(_order("BUY"), _at(0)) == pytest.approx(first)


def test_bad_parameters_are_rejected():
    with pytest.raises(ValueError):
        PermanentImpactSlippage(permanent_coefficient=-1.0)
    with pytest.raises(ValueError):
        PermanentImpactSlippage(half_life_bars=0.0)


def test_permanent_impact_costs_more_and_runs_are_independent():
    bars = generate_price_series(SyntheticConfig(n_bars=600), seed=1).to_bars()

    def factory(events, data):
        return TimeSeriesMomentumStrategy(
            events, data, symbol=data.symbols[0], lookback=5
        )

    temp = run_backtest(bars, factory, slippage=SpreadPlusImpactSlippage())
    model = PermanentImpactSlippage(permanent_coefficient=0.5)
    perm = run_backtest(bars, factory, slippage=model)
    again = run_backtest(bars, factory, slippage=model)
    assert perm.total_slippage > temp.total_slippage
    assert perm.equity[-1] < temp.equity[-1]
    # The shared instance is reset between runs, so a rerun is identical.
    assert again.equity == perm.equity
