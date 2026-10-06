"""Cost models must always cost you something.

Two classes of test here. The unit tests pin down the shape of each model --
in particular that slippage is *adverse by construction*, because a slippage
model that can accidentally improve a fill flatters the backtest. The
integration tests assert the property the whole project rests on: layering a
friction on top of an otherwise identical run can only reduce returns.
"""

from __future__ import annotations

import pytest

from honest_backtest.commission import (
    PercentOfNotionalCommission,
    PerShareCommission,
    ZeroCommission,
)
from honest_backtest.data import Bar, bars_from_series
from honest_backtest.engine import run_backtest
from honest_backtest.events import OrderEvent
from honest_backtest.slippage import (
    FixedBpsSlippage,
    PermanentImpactSlippage,
    SpreadPlusImpactSlippage,
    ZeroSlippage,
)
from honest_backtest.strategy import TimeSeriesMomentumStrategy

BAR = Bar(
    timestamp=0, symbol="SYNTH", open=100.0, high=101.0, low=99.0, close=100.0,
    volume=1_000_000.0,
)


def _order(direction: str, qty: float = 1_000.0) -> OrderEvent:
    return OrderEvent(symbol="SYNTH", timestamp=0, quantity=qty, direction=direction)


ALL_SLIPPAGE = [
    ZeroSlippage(),
    FixedBpsSlippage(2.0),
    FixedBpsSlippage(25.0),
    SpreadPlusImpactSlippage(),
    SpreadPlusImpactSlippage(half_spread_bps=5.0, impact_coefficient=1.2),
    PermanentImpactSlippage(permanent_coefficient=1.0),
]


@pytest.mark.parametrize("model", ALL_SLIPPAGE, ids=lambda m: type(m).__name__ + str(id(m))[-3:])
def test_slippage_is_never_favourable(model):
    """A buy never fills below the reference, a sell never above it."""
    assert model.fill_price(_order("BUY"), BAR) >= BAR.close
    assert model.fill_price(_order("SELL"), BAR) <= BAR.close


def test_fixed_bps_slippage_is_exact():
    model = FixedBpsSlippage(10.0)
    assert model.fill_price(_order("BUY"), BAR) == pytest.approx(100.10)
    assert model.fill_price(_order("SELL"), BAR) == pytest.approx(99.90)


def test_impact_grows_with_order_size():
    """Bigger orders must pay more, and the growth must be concave (sqrt law)."""
    model = SpreadPlusImpactSlippage()
    prices = [model.fill_price(_order("BUY", q), BAR) for q in (1e3, 1e4, 1e5)]
    assert prices[0] < prices[1] < prices[2]
    # Concavity: a 10x bigger order costs less than 10x more impact.
    impact = [p - BAR.close - BAR.close * 2e-4 for p in prices]
    assert impact[1] < 10 * impact[0]


def test_zero_slippage_is_the_close():
    assert ZeroSlippage().fill_price(_order("BUY"), BAR) == BAR.close


def test_negative_slippage_parameters_are_rejected():
    with pytest.raises(ValueError):
        FixedBpsSlippage(-1.0)
    with pytest.raises(ValueError):
        SpreadPlusImpactSlippage(impact_coefficient=-1.0)


def test_commission_is_never_negative():
    for model in (ZeroCommission(), PerShareCommission(), PercentOfNotionalCommission(5.0)):
        assert model.calculate(_order("BUY"), 100.0) >= 0.0
        assert model.calculate(_order("SELL"), 100.0) >= 0.0


def test_per_share_commission_floor_and_cap():
    model = PerShareCommission(per_share=0.005, minimum=1.0, max_fraction_of_notional=0.01)
    # Tiny order: the floor bites.
    assert model.calculate(_order("BUY", 10), 100.0) == pytest.approx(1.0)
    # Normal order: per-share rate applies.
    assert model.calculate(_order("BUY", 1_000), 100.0) == pytest.approx(5.0)
    # Penny stock: the notional cap bites.
    assert model.calculate(_order("BUY", 10_000), 0.01) == pytest.approx(0.01 * 100.0)


def test_percent_of_notional_commission_is_exact():
    model = PercentOfNotionalCommission(bps=10.0)
    assert model.calculate(_order("BUY", 1_000), 100.0) == pytest.approx(100.0)


# --- The property the whole project rests on ---------------------------------------


def _final_equity(bars, slippage, commission) -> float:
    """Run one identical backtest under the given frictions."""
    return run_backtest(
        bars,
        lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=10),
        slippage=slippage,
        commission=commission,
    ).equity[-1]


def test_slippage_strictly_reduces_returns(bars):
    frictionless = _final_equity(bars, ZeroSlippage(), ZeroCommission())
    with_slippage = _final_equity(bars, FixedBpsSlippage(5.0), ZeroCommission())
    assert with_slippage < frictionless


def test_commission_strictly_reduces_returns(bars):
    frictionless = _final_equity(bars, ZeroSlippage(), ZeroCommission())
    with_fees = _final_equity(bars, ZeroSlippage(), PerShareCommission())
    assert with_fees < frictionless


def test_frictions_compound(bars):
    """Each layer must make things worse, not merely different."""
    none = _final_equity(bars, ZeroSlippage(), ZeroCommission())
    slip = _final_equity(bars, FixedBpsSlippage(5.0), ZeroCommission())
    both = _final_equity(bars, FixedBpsSlippage(5.0), PerShareCommission())
    assert both < slip < none


def test_more_slippage_is_monotonically_worse(bars):
    equities = [
        _final_equity(bars, FixedBpsSlippage(bps), ZeroCommission())
        for bps in (0.0, 2.0, 10.0, 50.0)
    ]
    assert equities == sorted(equities, reverse=True)


def test_costs_are_reported_and_add_up(bars):
    """Reported commission must equal the sum of the blotter's commissions."""
    result = run_backtest(
        bars,
        lambda e, d: TimeSeriesMomentumStrategy(e, d, lookback=10),
        slippage=FixedBpsSlippage(5.0),
        commission=PerShareCommission(),
    )
    assert result.total_commission == pytest.approx(sum(f.commission for f in result.fills))
    assert result.total_slippage == pytest.approx(sum(f.slippage_cost for f in result.fills))
    assert result.total_commission > 0
    assert result.total_slippage > 0


def test_a_strategy_that_never_trades_pays_nothing():
    """No trades, no costs -- the frictions must not be a fixed drag."""
    from honest_backtest.strategy import Strategy

    class Silent(Strategy):
        def __init__(self, events, data):
            pass

        def calculate_signals(self, event):
            return None

    result = run_backtest(
        bars_from_series([100.0, 101.0, 102.0]),
        lambda e, d: Silent(e, d),
        slippage=FixedBpsSlippage(50.0),
        commission=PerShareCommission(per_share=1.0),
    )
    assert result.total_commission == 0.0
    assert result.total_slippage == 0.0
