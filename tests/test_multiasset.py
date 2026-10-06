"""Several instruments on one cursor: alignment, point-in-time, accounting."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from queue import Queue

import pytest
from helpers import FixedWeightStrategy

from honest_backtest.commission import PerShareCommission
from honest_backtest.data import Bar, LookAheadError, bars_from_series
from honest_backtest.engine import run_backtest
from honest_backtest.events import SignalEvent
from honest_backtest.financing import Financing
from honest_backtest.multiasset import (
    CrossSectionalMomentumStrategy,
    MultiAssetDataHandler,
    align_panel,
    load_csv_panel,
)
from honest_backtest.slippage import FixedBpsSlippage, PermanentImpactSlippage
from honest_backtest.strategy import Strategy, TimeSeriesMomentumStrategy
from honest_backtest.synthetic import SyntheticConfig, generate_price_series


def _panel(n=400, seeds=(1, 2, 3)):
    return {
        f"S{seed}": bars_from_series(
            generate_price_series(SyntheticConfig(n_bars=n), seed=seed).closes,
            symbol=f"S{seed}",
        )
        for seed in seeds
    }


class Weights(Strategy):
    """Holds fixed weights in every instrument from the first bar."""

    def __init__(self, events, data, weights):
        self.events, self.data, self.weights = events, data, weights

    def calculate_signals(self, event):
        for symbol, w in self.weights.items():
            self.events.put(SignalEvent(symbol, event.timestamp, w))


class Recorder(Strategy):
    """Records, per bar, the latest timestamp it can see for every instrument."""

    def __init__(self, events, data):
        self.events, self.data, self.seen = events, data, []

    def calculate_signals(self, event):
        for symbol in self.data.symbols:
            for bar in self.data.get_latest_bars(symbol, 1000):
                self.seen.append((event.timestamp, bar.timestamp))


def test_no_instrument_is_ever_read_ahead_of_the_cursor():
    holder = {}

    def factory(e, d):
        holder["s"] = Recorder(e, d)
        return holder["s"]

    run_backtest(_panel(n=120), factory)
    assert holder["s"].seen
    assert all(seen <= now for now, seen in holder["s"].seen)


def test_the_handler_refuses_reads_before_the_first_bar_and_unknown_symbols():
    data = MultiAssetDataHandler(Queue(), _panel(n=10))
    with pytest.raises(LookAheadError):
        data.current_bar("S1")
    data.update_bars()
    with pytest.raises(KeyError):
        data.current_bar("NOPE")
    with pytest.raises(LookAheadError):
        data.peek_ahead("S1")


def test_misaligned_panels_are_refused():
    panel = _panel(n=10)
    panel["S1"] = panel["S1"][1:]
    with pytest.raises(ValueError):
        MultiAssetDataHandler(Queue(), panel)
    with pytest.raises(ValueError):
        align_panel(panel)  # undated and different lengths


def test_alignment_keeps_only_the_dates_every_instrument_traded():
    start = date(2024, 1, 1)

    def dated(symbol, days):
        return [
            Bar(
                i, symbol, 10.0, 10.0, 10.0, 10.0 + i, 1e6, time=start + timedelta(days=d)
            )
            for i, d in enumerate(days)
        ]

    panel = align_panel({"A": dated("A", [0, 1, 2, 3]), "B": dated("B", [0, 2, 3, 4])})
    assert [b.time.day for b in panel["A"]] == [1, 3, 4]
    assert [b.time.day for b in panel["B"]] == [1, 3, 4]
    assert [b.timestamp for b in panel["B"]] == [0, 1, 2]


def test_a_one_instrument_panel_reproduces_the_single_asset_engine_exactly():
    bars = generate_price_series(SyntheticConfig(n_bars=500), seed=9).to_bars()

    def factory(e, d):
        return TimeSeriesMomentumStrategy(e, d, symbol="SYNTH", lookback=10)

    kwargs = dict(slippage=FixedBpsSlippage(3.0), commission=PerShareCommission())
    single = run_backtest(bars, factory, **kwargs)
    panel = run_backtest({"SYNTH": bars}, factory, **kwargs)
    assert panel.equity == pytest.approx(single.equity, rel=0, abs=1e-6)
    assert panel.n_trades == single.n_trades


def test_equity_is_cash_plus_every_position_at_every_bar():
    result = run_backtest(
        _panel(),
        lambda e, d: CrossSectionalMomentumStrategy(e, d, lookback=20, rebalance_every=5),
        slippage=FixedBpsSlippage(3.0),
        commission=PerShareCommission(),
    )
    assert result.n_trades > 10
    for snap in result.snapshots:
        marked = snap.cash + sum(q * snap.prices[s] for s, q in snap.positions.items())
        assert snap.equity == pytest.approx(marked, abs=1e-6)


def test_the_run_reconciles_against_the_blotter_per_instrument():
    result = run_backtest(
        _panel(),
        lambda e, d: CrossSectionalMomentumStrategy(
            e, d, lookback=20, rebalance_every=5, long_short=True
        ),
        slippage=FixedBpsSlippage(3.0),
        commission=PerShareCommission(),
    )
    cash = result.initial_capital
    held = {"S1": 0.0, "S2": 0.0, "S3": 0.0}
    for fill in result.fills:
        cash -= fill.signed_quantity * fill.fill_price + fill.commission
        held[fill.symbol] += fill.signed_quantity
    assert cash == pytest.approx(result.final_cash, abs=1e-6)
    for symbol, q in held.items():
        assert q == pytest.approx(result.final_position[symbol], abs=1e-9)


def test_momentum_holds_the_instrument_that_rose_most():
    up = bars_from_series([100.0 * 1.01**i for i in range(80)], symbol="UP")
    flat = bars_from_series([100.0] * 80, symbol="FLAT")
    down = bars_from_series([100.0 * 0.99**i for i in range(80)], symbol="DOWN")
    result = run_backtest(
        {"UP": up, "FLAT": flat, "DOWN": down},
        lambda e, d: CrossSectionalMomentumStrategy(
            e, d, lookback=10, top_n=1, long_short=True, rebalance_every=5
        ),
        rebalance_threshold=0.0,
    )
    final = result.final_position
    assert final["UP"] > 0 and final["DOWN"] < 0 and final["FLAT"] == 0


def test_gross_leverage_is_capped_across_the_book():
    result = run_backtest(
        _panel(n=30),
        lambda e, d: Weights(e, d, {"S1": 1.0, "S2": 1.0, "S3": -1.0}),
        rebalance_threshold=0.0,
        financing=Financing(max_leverage=1.5),
    )
    # Orders are sized at the cap; the marks then drift a little with price.
    leverage = [s.gross_exposure / s.equity for s in result.snapshots[1:]]
    assert max(leverage) < 1.5 * 1.02
    assert min(leverage) > 1.5 * 0.98


def test_shorts_across_the_book_pay_the_lending_fee():
    flat = {s: bars_from_series([100.0] * 253, symbol=s) for s in ("A", "B")}
    result = run_backtest(
        flat,
        lambda e, d: Weights(e, d, {"A": -0.5, "B": -0.5}),
        rebalance_threshold=0.0,
        financing=Financing(short_fee=0.04),
    )
    assert 1 - result.equity[-1] / result.initial_capital == pytest.approx(0.04, rel=0.03)


def test_permanent_impact_is_tracked_per_instrument():
    model = PermanentImpactSlippage(permanent_coefficient=1.0)
    from honest_backtest.events import OrderEvent

    bar_a = Bar(0, "A", 100.0, 100.0, 100.0, 100.0, 1e5)
    bar_b = Bar(0, "B", 100.0, 100.0, 100.0, 100.0, 1e5)
    first_b = PermanentImpactSlippage(permanent_coefficient=1.0).fill_price(
        OrderEvent("B", 0, 5e4, "BUY"), bar_b
    )
    model.fill_price(OrderEvent("A", 0, 5e4, "BUY"), bar_a)
    # Buying A pushed A's price, not B's.
    assert model.fill_price(OrderEvent("B", 0, 5e4, "BUY"), bar_b) == pytest.approx(
        first_b
    )


def test_csv_files_load_into_an_aligned_panel(tmp_path):
    paths = []
    for name, skip in (("aaa", 2), ("bbb", None)):
        path = tmp_path / f"{name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Date", "Close", "Volume"])
            for i in range(6):
                if i == skip:
                    continue
                writer.writerow(
                    [(date(2024, 1, 1) + timedelta(days=i)).isoformat(), 10 + i, 1e6]
                )
        paths.append(path)
    panel = load_csv_panel(paths)
    assert list(panel) == ["AAA", "BBB"]
    assert len(panel["AAA"]) == len(panel["BBB"]) == 5


def test_the_single_asset_scripted_strategy_still_works_on_a_panel():
    bars = bars_from_series([100.0, 101.0, 102.0])
    result = run_backtest(
        {"SYNTH": bars},
        lambda e, d: FixedWeightStrategy(e, d, [1.0, 1.0, 1.0]),
        rebalance_threshold=0.0,
    )
    assert result.final_position["SYNTH"] > 0
