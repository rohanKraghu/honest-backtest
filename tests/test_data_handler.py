"""The point-in-time guarantee: the data handler cannot serve the future.

These are the most important tests in the project. If they pass, look-ahead
bias is structurally impossible for any component that gets its data through
the handler -- which is every component.
"""

from __future__ import annotations

import pytest

from honest_backtest.data import (
    HistoricBarDataHandler,
    LookAheadDataHandler,
    LookAheadError,
    bars_from_series,
)
from honest_backtest.engine import run_backtest
from honest_backtest.strategy import LookAheadMomentumStrategy

from helpers import RecordingStrategy, make_handler


def test_cursor_starts_before_the_first_bar(short_bars):
    _, handler = make_handler(short_bars)
    assert handler.cursor == -1


def test_reading_before_any_bar_is_delivered_raises(short_bars):
    _, handler = make_handler(short_bars)
    with pytest.raises(LookAheadError):
        handler.get_latest_bars("SYNTH", 1)
    with pytest.raises(LookAheadError):
        handler.current_bar("SYNTH")


def test_update_bars_advances_exactly_one_bar(short_bars):
    _, handler = make_handler(short_bars)
    for expected in range(len(short_bars)):
        handler.update_bars()
        assert handler.cursor == expected
        assert handler.current_bar("SYNTH").timestamp == expected


def test_get_latest_bars_never_returns_a_future_bar(short_bars):
    """Exhaustive over the whole series and over every window length."""
    _, handler = make_handler(short_bars)
    for _ in range(len(short_bars)):
        handler.update_bars()
        now = handler.current_bar("SYNTH").timestamp
        for n in range(1, len(short_bars) + 5):
            window = handler.get_latest_bars("SYNTH", n)
            assert window, "a started handler must return at least one bar"
            assert max(b.timestamp for b in window) <= now
            assert len(window) <= n
            # Bars come back oldest-first and contiguous.
            stamps = [b.timestamp for b in window]
            assert stamps == sorted(stamps)
            assert stamps[-1] == now


def test_latest_closes_agrees_with_latest_bars(short_bars):
    _, handler = make_handler(short_bars)
    for _ in range(len(short_bars)):
        handler.update_bars()
        for n in (1, 3, 10):
            assert handler.latest_closes("SYNTH", n) == [
                b.close for b in handler.get_latest_bars("SYNTH", n)
            ]


def test_peek_ahead_is_refused_by_the_point_in_time_handler(short_bars):
    _, handler = make_handler(short_bars)
    handler.update_bars()
    with pytest.raises(LookAheadError):
        handler.peek_ahead("SYNTH", 1)


def test_unknown_symbol_is_rejected(short_bars):
    _, handler = make_handler(short_bars)
    handler.update_bars()
    with pytest.raises(KeyError):
        handler.get_latest_bars("NOPE", 1)


def test_non_positive_window_is_rejected(short_bars):
    _, handler = make_handler(short_bars)
    handler.update_bars()
    with pytest.raises(ValueError):
        handler.get_latest_bars("SYNTH", 0)


def test_backtest_never_shows_a_strategy_a_future_bar(bars):
    """The guarantee, asserted over a full backtest rather than in isolation."""
    recorder: list[RecordingStrategy] = []

    def factory(events, data):
        strategy = RecordingStrategy(events, data, window=200)
        recorder.append(strategy)
        return strategy

    run_backtest(bars, factory)
    observations = recorder[0].observations
    assert observations, "the strategy must have seen something"
    assert all(seen <= now for now, seen in observations)
    assert max(seen for _, seen in observations) == len(bars) - 1


def test_handler_stops_when_the_data_runs_out(short_bars):
    _, handler = make_handler(short_bars)
    for _ in range(len(short_bars)):
        handler.update_bars()
    assert handler.continue_backtest is True
    handler.update_bars()
    assert handler.continue_backtest is False
    # The cursor does not run past the end.
    assert handler.cursor == len(short_bars) - 1


def test_empty_series_terminates_immediately():
    from queue import Queue

    handler = HistoricBarDataHandler(Queue(), [])
    assert handler.continue_backtest is False


# --- The deliberate counterexample -------------------------------------------------


def test_look_ahead_handler_really_does_leak(short_bars):
    """If this fails, stage 1 of the study is not reproducing the bug."""
    from queue import Queue

    handler = LookAheadDataHandler(Queue(), short_bars)
    handler.update_bars()
    future = handler.peek_ahead("SYNTH", 2)
    assert [b.timestamp for b in future] == [1, 2]


def test_look_ahead_handler_peek_is_empty_at_the_end(short_bars):
    from queue import Queue

    handler = LookAheadDataHandler(Queue(), short_bars)
    for _ in range(len(short_bars)):
        handler.update_bars()
    assert handler.peek_ahead("SYNTH", 3) == []


def test_all_closes_exposes_the_whole_sample(short_bars):
    from queue import Queue

    handler = LookAheadDataHandler(Queue(), short_bars)
    handler.update_bars()
    assert len(handler.all_closes("SYNTH")) == len(short_bars)


def test_leaky_strategy_refuses_a_point_in_time_handler(short_bars):
    """The leak must be opt-in. A PIT handler cannot be tricked into it."""
    events, handler = make_handler(bars_from_series([1.0, 2.0, 3.0]))
    with pytest.raises(TypeError):
        LookAheadMomentumStrategy(events, handler, lookback=2)
