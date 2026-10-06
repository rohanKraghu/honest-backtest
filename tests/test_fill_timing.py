"""Filling on the next bar removes the "trade at the close you just saw" assumption."""

from __future__ import annotations

import pytest
from helpers import FixedWeightStrategy

from honest_backtest.data import Bar, bars_from_series
from honest_backtest.engine import run_backtest
from honest_backtest.execution import SimulatedExecutionHandler
from honest_backtest.slippage import FixedBpsSlippage

CLOSES = [100.0, 110.0, 99.0, 120.0, 90.0, 105.0]


def _gapped_bars():
    """Bars whose opens differ from the previous close, like real daily data."""
    bars = []
    for i, close in enumerate(CLOSES):
        open_ = close - 3.0 if i else close
        bars.append(
            Bar(i, "SYNTH", open_, max(open_, close), min(open_, close), close, 1e6)
        )
    return bars


def _run(bars, timing, weights=(1.0,), **kwargs):
    return run_backtest(
        bars,
        lambda events, data: FixedWeightStrategy(events, data, list(weights)),
        rebalance_threshold=0.0,
        fill_timing=timing,
        **kwargs,
    )


def test_next_close_fills_at_the_following_bar_close():
    result = _run(bars_from_series(CLOSES), "next_close")
    (fill,) = result.fills
    assert fill.timestamp == 1
    assert fill.fill_price == 110.0
    # Sized at bar 0's close, filled at bar 1's.
    assert fill.quantity == pytest.approx(1_000_000.0 / 100.0)


def test_next_open_uses_the_real_open():
    result = _run(_gapped_bars(), "next_open")
    (fill,) = result.fills
    assert fill.timestamp == 1
    assert fill.fill_price == 107.0
    assert fill.reference_price == 107.0


def test_next_open_on_derived_opens_matches_the_close_fill():
    """Synthetic bars take their open from the previous close, so nothing changes."""
    bars = bars_from_series(CLOSES)
    same = _run(bars, "close", weights=(1.0, -0.5, 0.25, 1.0))
    nxt = _run(bars, "next_open", weights=(1.0, -0.5, 0.25, 1.0))
    assert [f.fill_price for f in nxt.fills] == [f.fill_price for f in same.fills]
    assert nxt.final_position == pytest.approx(same.final_position)


def test_delayed_fills_keep_the_books_balanced():
    bars = _gapped_bars()
    result = _run(
        bars,
        "next_open",
        weights=(1.0, -1.0, 0.5, 0.0, 1.0),
        slippage=FixedBpsSlippage(5.0),
    )
    for snap in result.snapshots:
        assert snap.equity == pytest.approx(snap.cash + snap.position * snap.price)
    cash = 1_000_000.0
    position = 0.0
    for fill in result.fills:
        cash -= fill.signed_quantity * fill.fill_price + fill.commission
        position += fill.signed_quantity
    assert result.final_cash == pytest.approx(cash)
    assert result.final_position == pytest.approx(position)


def test_snapshot_of_the_fill_bar_includes_the_fill():
    result = _run(bars_from_series(CLOSES), "next_close")
    assert result.snapshots[0].position == 0.0
    assert result.snapshots[1].position == pytest.approx(1_000_000.0 / 100.0)


def test_an_order_on_the_last_bar_is_never_filled():
    bars = bars_from_series(CLOSES[:2])
    result = _run(bars, "next_close", weights=(0.0, 1.0))
    assert result.fills == []


def test_unknown_timing_is_rejected():
    with pytest.raises(ValueError, match="fill_timing"):
        SimulatedExecutionHandler(None, None, fill_timing="tomorrow")
