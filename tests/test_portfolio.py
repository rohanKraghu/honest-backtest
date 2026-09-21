"""Portfolio accounting: every unit of cash must be accounted for.

A backtester that loses track of cash will report whatever its author hoped
for, and nobody will notice. These tests assert the accounting identity bar
by bar and reconcile the full run against the trade blotter.
"""

from __future__ import annotations

import math
from queue import Queue

import pytest

from honest_backtest.commission import PerShareCommission, ZeroCommission
from honest_backtest.data import bars_from_series
from honest_backtest.engine import run_backtest
from honest_backtest.events import FillEvent
from honest_backtest.portfolio import Portfolio
from honest_backtest.slippage import FixedBpsSlippage, ZeroSlippage
from honest_backtest.strategy import BuyAndHoldStrategy, TimeSeriesMomentumStrategy

from helpers import FixedWeightStrategy, make_handler

TOL = 1e-6


def _run_scripted(weights, closes, slippage=None, commission=None, capital=1_000_000.0):
    """Run a backtest with a scripted weight schedule."""
    return run_backtest(
        bars_from_series(closes),
        lambda e, d: FixedWeightStrategy(e, d, weights),
        slippage=slippage,
        commission=commission,
        initial_capital=capital,
        rebalance_threshold=0.0,
    )


def test_mark_to_market_identity_holds_at_every_bar(bars):
    """equity == cash + position * price, at every single snapshot."""
    result = run_backtest(
        bars,
        lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=10),
        slippage=FixedBpsSlippage(3.0),
        commission=PerShareCommission(),
    )
    assert result.snapshots
    for snap in result.snapshots:
        assert snap.equity == pytest.approx(
            snap.cash + snap.position * snap.price, rel=0, abs=TOL
        )


def test_bar_to_bar_accounting_identity(bars):
    """The change in equity must equal PnL minus costs, exactly.

    equity[t+1] - equity[t]
        = position_t * (price[t+1] - price[t])          # mark-to-market PnL
        + dq_t * (price[t+1] - fill_price_t)            # PnL on what was traded
        - commission_t                                  # explicit cost

    If any cash leaks, this fails.
    """
    result = run_backtest(
        bars,
        lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=10),
        slippage=FixedBpsSlippage(3.0),
        commission=PerShareCommission(),
    )
    fills_by_ts: dict[int, list[FillEvent]] = {}
    for fill in result.fills:
        fills_by_ts.setdefault(fill.timestamp, []).append(fill)

    snaps = result.snapshots
    assert len(snaps) > 100
    for now, nxt in zip(snaps, snaps[1:]):
        traded_pnl = 0.0
        fees = 0.0
        for fill in fills_by_ts.get(now.timestamp, []):
            traded_pnl += fill.signed_quantity * (nxt.price - fill.fill_price)
            fees += fill.commission
        expected = (
            now.position * (nxt.price - now.price) + traded_pnl - fees
        )
        assert nxt.equity - now.equity == pytest.approx(expected, rel=0, abs=1e-6)


def test_full_run_reconciles_against_the_blotter(bars):
    """Final cash must equal initial capital minus every signed cash flow."""
    result = run_backtest(
        bars,
        lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=15),
        slippage=FixedBpsSlippage(2.0),
        commission=PerShareCommission(),
    )
    cash = result.initial_capital
    position = 0.0
    for fill in result.fills:
        cash -= fill.signed_quantity * fill.fill_price
        cash -= fill.commission
        position += fill.signed_quantity
    assert position == pytest.approx(result.final_position, rel=0, abs=1e-9)
    assert cash == pytest.approx(result.final_cash, rel=0, abs=1e-6)


def test_last_snapshot_precedes_the_last_fill(bars):
    """Snapshots are taken on bar arrival, before that bar's own fills.

    This ordering is deliberate -- it is what makes a bar's trading costs
    land in the *next* period's return rather than the one that generated
    them -- so the final snapshot lags the blotter by one bar. Pinning it
    down here stops anyone "fixing" it later and silently shifting every
    cost by one period.
    """
    result = run_backtest(
        bars,
        lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=15),
        slippage=FixedBpsSlippage(2.0),
        commission=PerShareCommission(),
    )
    last_ts = result.snapshots[-1].timestamp
    cash = result.initial_capital
    position = 0.0
    for fill in result.fills:
        if fill.timestamp >= last_ts:
            continue
        cash -= fill.signed_quantity * fill.fill_price
        cash -= fill.commission
        position += fill.signed_quantity
    final = result.snapshots[-1]
    assert position == pytest.approx(final.position, rel=0, abs=1e-9)
    assert cash == pytest.approx(final.cash, rel=0, abs=1e-6)


def test_no_zero_quantity_fills(bars):
    """A zero-size order must never become a fill and inflate the trade count."""
    result = run_backtest(
        bars,
        lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=10),
        rebalance_threshold=0.0,
    )
    assert result.fills
    assert all(f.quantity > 0 for f in result.fills)


def test_a_flat_book_is_exactly_flat():
    """No signal, no trades, no drift in equity."""
    from honest_backtest.strategy import Strategy

    class Silent(Strategy):
        def __init__(self, events, data):
            pass

        def calculate_signals(self, event):
            return None

    result = run_backtest(bars_from_series([100.0, 110.0, 90.0, 105.0]), lambda e, d: Silent(e, d))
    assert result.n_trades == 0
    assert all(v == 1_000_000.0 for v in result.equity)


def test_buy_and_hold_tracks_the_instrument():
    """A fully invested book must earn exactly the instrument's return."""
    closes = [100.0, 104.0, 98.0, 101.0, 120.0]
    result = run_backtest(
        bars_from_series(closes),
        lambda e, d: BuyAndHoldStrategy(e, d),
        slippage=ZeroSlippage(),
        commission=ZeroCommission(),
        rebalance_threshold=0.0,
    )
    # Entered at close of bar 0, so the book earns closes[-1] / closes[0] - 1.
    assert result.equity[-1] / result.initial_capital == pytest.approx(
        closes[-1] / closes[0], rel=1e-12
    )


def test_short_positions_are_accounted_correctly():
    """Going short a falling instrument must make money; the identity still holds."""
    closes = [100.0, 95.0, 90.0, 85.0]
    result = _run_scripted([-1.0, -1.0, -1.0, -1.0], closes)
    assert result.snapshots[1].position < 0
    assert result.equity[-1] > result.initial_capital
    for snap in result.snapshots:
        assert snap.equity == pytest.approx(snap.cash + snap.position * snap.price, abs=TOL)


def test_target_weight_is_respected():
    """A 50% target weight must produce a position worth half of equity."""
    result = _run_scripted([0.5, 0.5, 0.5], [100.0, 100.0, 100.0])
    snap = result.snapshots[1]
    assert snap.position * snap.price == pytest.approx(0.5 * snap.equity, rel=1e-9)


def test_rebalance_band_suppresses_dust_trades():
    """Below the band, no order is generated at all."""
    bars_ = bars_from_series([100.0] * 10)
    tight = run_backtest(
        bars_, lambda e, d: FixedWeightStrategy(e, d, [0.50, 0.51] + [0.51] * 8),
        rebalance_threshold=0.0,
    )
    banded = run_backtest(
        bars_, lambda e, d: FixedWeightStrategy(e, d, [0.50, 0.51] + [0.51] * 8),
        rebalance_threshold=0.05,
    )
    assert tight.n_trades == 2, "a 1% move should trade when there is no band"
    assert banded.n_trades == 1, "a 1% move should be suppressed by a 5% band"


def test_orders_and_fills_are_one_to_one(bars):
    """Every order produced exactly one fill; nothing was dropped."""
    result = run_backtest(bars, lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=10))
    assert result.event_counts["ORDER"] == result.event_counts["FILL"]
    assert result.event_counts["FILL"] == result.n_trades
    assert result.event_counts["MARKET"] == len(bars)


def test_invalid_capital_is_rejected():
    q: Queue = Queue()
    _, handler = make_handler(bars_from_series([1.0, 2.0]))
    with pytest.raises(ValueError):
        Portfolio(events=q, data=handler, initial_capital=0.0)
    with pytest.raises(ValueError):
        Portfolio(events=q, data=handler, rebalance_threshold=-0.1)


def test_equity_curve_has_one_point_per_bar(bars):
    result = run_backtest(bars, lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=5))
    assert len(result.equity) == len(bars)
    assert result.timestamps == [b.timestamp for b in bars]
    assert all(math.isfinite(v) for v in result.equity)
