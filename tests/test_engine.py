"""The event loop: causal ordering, determinism, and no dropped events."""

from __future__ import annotations

from queue import Queue

import pytest

from honest_backtest.data import HistoricBarDataHandler
from honest_backtest.engine import Backtest, run_backtest
from honest_backtest.events import (
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from honest_backtest.execution import SimulatedExecutionHandler
from honest_backtest.portfolio import Portfolio
from honest_backtest.slippage import FixedBpsSlippage
from honest_backtest.strategy import TimeSeriesMomentumStrategy


def test_every_bar_produces_exactly_one_market_event(bars):
    result = run_backtest(bars, lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=10))
    assert result.event_counts["MARKET"] == len(bars)


def test_the_causal_chain_is_respected(bars):
    """Signals <= markets, orders <= signals, fills == orders."""
    result = run_backtest(bars, lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=10))
    c = result.event_counts
    assert c["SIGNAL"] <= c["MARKET"]
    assert c["ORDER"] <= c["SIGNAL"]
    assert c["FILL"] == c["ORDER"]
    assert c["ORDER"] > 0, "the test is vacuous if nothing traded"


def test_the_queue_is_drained_before_time_advances(bars):
    """No event may survive to the end of the run."""
    events: Queue = Queue()
    data = HistoricBarDataHandler(events, bars)
    strategy = TimeSeriesMomentumStrategy(events, data, lookback=10)
    portfolio = Portfolio(events=events, data=data)
    execution = SimulatedExecutionHandler(events, data, FixedBpsSlippage(2.0))
    Backtest(data, strategy, portfolio, execution, events).run()
    assert events.empty()


def test_runs_are_deterministic(bars):
    a = run_backtest(bars, lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=12))
    b = run_backtest(bars, lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=12))
    assert a.equity == b.equity
    assert a.n_trades == b.n_trades


def test_warmup_excludes_leading_bars_from_scoring(bars):
    result = run_backtest(
        bars, lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=10), warmup=100
    )
    assert len(result.equity) == len(bars)
    # One extra leading point is kept as the base for the first scored return.
    assert len(result.scored_equity) == len(bars) - 99
    assert result.metrics().n_periods == len(bars) - 100


def test_a_limit_order_without_a_limit_price_is_refused_not_treated_as_market():
    with pytest.raises(ValueError):
        OrderEvent("SYNTH", 0, 100.0, "BUY", order_type="LMT")
    with pytest.raises(ValueError):
        OrderEvent("SYNTH", 0, 100.0, "BUY", limit_price=99.0)
    with pytest.raises(ValueError):
        OrderEvent("SYNTH", 0, 100.0, "BUY", order_type="STOP")


def test_event_constructors_validate_their_inputs():
    with pytest.raises(ValueError):
        OrderEvent("SYNTH", 0, -5.0, "BUY")
    with pytest.raises(ValueError):
        OrderEvent("SYNTH", 0, 5.0, "SIDEWAYS")


def test_events_carry_the_right_discriminator():
    assert MarketEvent(0).type is EventType.MARKET
    assert SignalEvent("S", 0, 0.5).type is EventType.SIGNAL
    assert OrderEvent("S", 0, 1.0, "BUY").type is EventType.ORDER
    assert FillEvent("S", 0, 1.0, "BUY", 101.0, 0.5, 100.0).type is EventType.FILL


def test_signed_quantity_and_slippage_cost_are_consistent():
    buy = FillEvent("S", 0, 10.0, "BUY", 101.0, 1.0, 100.0)
    sell = FillEvent("S", 0, 10.0, "SELL", 99.0, 1.0, 100.0)
    assert buy.signed_quantity == 10.0
    assert sell.signed_quantity == -10.0
    assert buy.slippage_cost == pytest.approx(10.0)
    assert sell.slippage_cost == pytest.approx(10.0)


def test_events_are_immutable():
    """Frozen events mean no consumer can rewrite history."""
    event = SignalEvent("S", 0, 0.5)
    with pytest.raises(Exception):
        event.target_weight = 1.0  # type: ignore[misc]
