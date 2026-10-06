"""The look-ahead detector: change the future, and the past must not move."""

from __future__ import annotations

from pathlib import Path

import pytest

from honest_backtest import SignalEvent, Strategy
from honest_backtest.audit import AuditConfig, render_audit_report, run_audit
from honest_backtest.experiments import LadderSettings, momentum_spec
from honest_backtest.leaks import (
    default_cuts,
    detect_look_ahead,
    perturb_future,
    record_signals,
)
from honest_backtest.spec import StrategySpec, param_grid
from honest_backtest.synthetic import SyntheticConfig, generate_price_series


@pytest.fixture(scope="module")
def bars():
    """A short synthetic path, shared across tests."""
    return generate_price_series(SyntheticConfig(n_bars=600), seed=3).to_bars()


class _Base(Strategy):
    def __init__(self, events, data, symbol):
        self.events, self.data, self.symbol = events, data, symbol

    def emit(self, event, weight):
        self.events.put(SignalEvent(self.symbol, event.timestamp, weight))


class PrivateListPeek(_Base):
    """Goes around the handler's accessors to read tomorrow's close."""

    def calculate_signals(self, event):
        i = self.data.cursor
        if i + 1 < len(self.data._bars):
            up = self.data._bars[i + 1].close > self.data._bars[i].close
            self.emit(event, 1.0 if up else -1.0)


class SharedCache(_Base):
    """Remembers the last close of the whole run in a class attribute.

    A leak across runs rather than within one: the second backtest starts
    already knowing where the first one ended.
    """

    final_close: float | None = None

    def calculate_signals(self, event):
        bar = self.data.current_bar(self.symbol)
        if SharedCache.final_close is not None:
            self.emit(event, 1.0 if SharedCache.final_close > bar.close else -1.0)
        if not self.data.continue_backtest or self.data.cursor == len(self.data) - 1:
            SharedCache.final_close = bar.close


class AsksForTheFuture(_Base):
    """Calls the accessor that the point-in-time handler refuses."""

    def calculate_signals(self, event):
        self.data.peek_ahead(self.symbol)


class Coin(_Base):
    """Random signals: nothing can be concluded from a divergence."""

    def __init__(self, events, data, symbol):
        super().__init__(events, data, symbol)
        import random

        self.rng = random.Random()

    def calculate_signals(self, event):
        self.emit(event, self.rng.choice((-1.0, 1.0)))


def _spec(cls, name=None):
    return StrategySpec(name=name or cls.__name__, build=cls, grid=param_grid())


def test_honest_strategies_pass(bars):
    report = detect_look_ahead(bars, momentum_spec((5, 20)))
    assert report.passed
    assert len(report.cuts) == 5
    assert "passed" in report.summary()


def test_the_deliberately_leaky_twin_is_caught(bars):
    report = detect_look_ahead(bars, momentum_spec((5, 20)), look_ahead=True)
    assert not report.passed
    assert len(report.findings) == 2  # one per setting
    assert "LEAK FOUND" in report.summary()


def test_reading_the_handlers_private_list_is_caught(bars):
    report = detect_look_ahead(bars, _spec(PrivateListPeek))
    assert not report.passed
    finding = report.findings[0]
    # It moves the very first signal at or before the first cut.
    assert finding.timestamp is not None
    assert finding.timestamp <= bars[finding.cut].timestamp


def test_state_shared_between_runs_fails_the_check(bars):
    # Two runs on identical data disagree, so the check refuses to vouch.
    SharedCache.final_close = None
    report = detect_look_ahead(bars, _spec(SharedCache))
    assert not report.passed
    assert report.nondeterministic


def test_asking_the_handler_for_the_future_is_reported_not_raised(bars):
    report = detect_look_ahead(bars, _spec(AsksForTheFuture))
    assert not report.passed
    assert "refused" in report.findings[0].detail


def test_nondeterministic_strategies_are_flagged_as_uncheckable(bars):
    report = detect_look_ahead(bars, _spec(Coin))
    assert not report.passed
    assert not report.findings
    assert "not checkable" in report.summary()


def test_perturbation_keeps_the_past_and_replaces_the_future(bars):
    cut = 300
    changed = perturb_future(bars, cut, seed=1)
    assert changed[: cut + 1] == bars[: cut + 1]
    assert len(changed) == len(bars)
    assert all(
        a.close != b.close
        for a, b in zip(bars[cut + 1 :], changed[cut + 1 :], strict=True)
    )
    assert [b.timestamp for b in changed] == [b.timestamp for b in bars]
    assert all(
        b.low <= min(b.open, b.close) and b.high >= max(b.open, b.close) for b in changed
    )


def test_cut_points_respect_the_warmup():
    cuts = default_cuts(1000, warmup=200, n_cuts=4)
    assert len(cuts) == 4
    assert all(200 < c < 998 for c in cuts)
    assert default_cuts(3, warmup=0, n_cuts=5) == (1,)


def test_record_signals_matches_what_the_engine_sees(bars):
    signals = record_signals(bars, momentum_spec((20,)), {"lookback": 20})
    assert signals
    assert [t for t, _ in signals] == sorted(t for t, _ in signals)


def test_an_audit_of_a_leaky_strategy_leads_with_a_warning(bars):
    result = run_audit(
        bars,
        _spec(PrivateListPeek),
        AuditConfig(settings=LadderSettings(train_size=150, test_size=150)),
    )
    text = render_audit_report(result)
    assert text.startswith("WARNING")
    assert "LEAK FOUND" in text


def test_the_check_can_be_turned_off(bars):
    result = run_audit(
        bars,
        _spec(PrivateListPeek),
        AuditConfig(
            settings=LadderSettings(train_size=150, test_size=150), check_leaks=False
        ),
    )
    assert result.leaks is None
    assert "look-ahead check    skipped" in render_audit_report(result)


LEAKY_FILE = """
from honest_backtest import SignalEvent, Strategy, StrategySpec, param_grid

class Peek(Strategy):
    def __init__(self, events, data, symbol):
        self.events, self.data, self.symbol = events, data, symbol

    def calculate_signals(self, event):
        i = self.data.cursor
        if i + 1 < len(self.data._bars):
            up = self.data._bars[i + 1].close > self.data._bars[i].close
            w = 1.0 if up else -1.0
            self.events.put(SignalEvent(self.symbol, event.timestamp, w))

SPEC = StrategySpec(name="peek", build=Peek, grid=param_grid())
"""


def test_the_audit_command_exits_nonzero_on_a_leak(tmp_path, capsys):
    from honest_backtest.cli import main

    strategy = tmp_path / "peek.py"
    strategy.write_text(LEAKY_FILE)
    data = Path(__file__).resolve().parents[1] / "examples" / "sample_prices.csv"
    args = ["audit", "--data", str(data), "--strategy", str(strategy)]
    assert main(args) == 1
    assert "WARNING" in capsys.readouterr().out
    assert main([*args, "--no-leak-check"]) == 0
