"""Holding a position costs money: interest, borrow fees, and a leverage cap."""

from __future__ import annotations

import pytest
from helpers import FixedWeightStrategy

from honest_backtest.commission import PerShareCommission
from honest_backtest.data import bars_from_series
from honest_backtest.engine import run_backtest
from honest_backtest.events import FillEvent
from honest_backtest.financing import Financing
from honest_backtest.slippage import FixedBpsSlippage
from honest_backtest.strategy import TimeSeriesMomentumStrategy
from honest_backtest.synthetic import SyntheticConfig, generate_price_series

FLAT = [100.0] * 253  # one year of flat prices: only financing moves equity


def _run(weights, closes=FLAT, financing=None):
    return run_backtest(
        bars_from_series(closes),
        lambda e, d: FixedWeightStrategy(e, d, list(weights)),
        rebalance_threshold=0.0,
        financing=financing,
    )


def test_without_financing_holding_is_free():
    result = _run([-1.0], financing=None)
    assert result.equity[-1] == pytest.approx(result.initial_capital)
    assert result.total_financing == 0.0


def test_idle_cash_earns_the_cash_rate():
    result = _run([0.0], financing=Financing(cash_rate=0.05))
    # 252 accruals of 5%/252 on a growing balance: compound interest.
    assert result.equity[-1] / result.initial_capital == pytest.approx(
        (1 + 0.05 / 252) ** 252, rel=1e-9
    )
    assert result.total_financing < 0  # negative cost: income


def test_leverage_pays_the_borrow_rate():
    result = _run([2.0], financing=Financing(borrow_rate=0.08))
    # The loan is one unit of equity and compounds at 8%/252 per bar.
    loss = 1 - result.equity[-1] / result.initial_capital
    assert loss == pytest.approx((1 + 0.08 / 252) ** 252 - 1, rel=1e-9)
    assert result.total_financing > 0


def test_a_short_pays_the_lending_fee_and_earns_interest_on_proceeds():
    fee_only = _run([-1.0], financing=Financing(short_fee=0.03))
    both = _run([-1.0], financing=Financing(cash_rate=0.04, short_fee=0.03))
    assert 1 - fee_only.equity[-1] / fee_only.initial_capital == pytest.approx(
        0.03, rel=0.02
    )
    # Proceeds double the cash balance, so 4% on twice the capital minus 3%
    # on the short: about +5%.
    assert both.equity[-1] / both.initial_capital - 1 == pytest.approx(0.05, rel=0.05)


def test_the_leverage_limit_clips_target_weights():
    result = _run([3.0, 3.0, 3.0], financing=Financing(max_leverage=1.5))
    snap = result.snapshots[1]
    assert snap.position * snap.price == pytest.approx(1.5 * snap.equity, rel=1e-9)


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        Financing(borrow_rate=-0.01)
    with pytest.raises(ValueError):
        Financing(max_leverage=0.0)


def test_the_accounting_identity_holds_with_financing():
    """Delta equity = mark-to-market + traded PnL - fees + financing accrued."""
    financing = Financing(cash_rate=0.03, borrow_rate=0.07, short_fee=0.02)
    bars = generate_price_series(SyntheticConfig(n_bars=400), seed=5).to_bars()
    result = run_backtest(
        bars,
        lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=10),
        slippage=FixedBpsSlippage(3.0),
        commission=PerShareCommission(),
        financing=financing,
    )
    fills_by_ts: dict[int, list[FillEvent]] = {}
    for fill in result.fills:
        fills_by_ts.setdefault(fill.timestamp, []).append(fill)
    snaps = result.snapshots
    total = 0.0
    for now, nxt in zip(snaps, snaps[1:], strict=False):
        traded, fees, cash, position = 0.0, 0.0, now.cash, now.position
        for fill in fills_by_ts.get(now.timestamp, []):
            traded += fill.signed_quantity * (nxt.price - fill.fill_price)
            fees += fill.commission
            cash -= fill.signed_quantity * fill.fill_price + fill.commission
            position += fill.signed_quantity
        accrued = financing.accrual(cash, position, now.price)
        total += accrued
        expected = now.position * (nxt.price - now.price) + traded - fees + accrued
        assert nxt.equity - now.equity == pytest.approx(expected, abs=1e-6)
    assert result.total_financing == pytest.approx(-total, abs=1e-6)
