"""Paper trading runs the backtested strategy unchanged, on bars as they arrive."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest
from helpers import RecordingStrategy

from honest_backtest.commission import PerShareCommission
from honest_backtest.data import LookAheadError
from honest_backtest.engine import run_backtest
from honest_backtest.experiments import momentum_spec
from honest_backtest.fast import replay_signals
from honest_backtest.financing import Financing
from honest_backtest.live import (
    CsvTailFeed,
    FeedError,
    LiveDataHandler,
    PaperTrader,
    PollingFeed,
    ReplayFeed,
    read_journal,
)
from honest_backtest.slippage import SpreadPlusImpactSlippage
from honest_backtest.synthetic import SyntheticConfig, generate_price_series

START = datetime(2024, 1, 1)


def _bars(n=600, seed=4):
    bars = generate_price_series(SyntheticConfig(n_bars=n), seed=seed).to_bars()
    return [replace(b, time=START + timedelta(days=i)) for i, b in enumerate(bars)]


def _costs():
    return dict(
        slippage=SpreadPlusImpactSlippage(2.0, 0.6, 0.01),
        commission=PerShareCommission(),
    )


def _factory(lookback=10):
    return momentum_spec().factory({"lookback": lookback})


@pytest.mark.parametrize("timing", ["close", "next_open", "next_close"])
def test_a_replayed_feed_reproduces_the_backtest_exactly(timing):
    bars = _bars()
    backtest = run_backtest(bars, _factory(), fill_timing=timing, **_costs())
    paper = PaperTrader(
        ReplayFeed(bars), _factory(), fill_timing=timing, **_costs()
    ).run()
    assert paper.equity == backtest.equity
    assert paper.fills == backtest.fills
    assert paper.final_cash == backtest.final_cash
    assert paper.final_position == backtest.final_position


def test_history_warms_the_strategy_to_exactly_its_backtest_state():
    """The momentum strategy keeps a trailing window of its own past values."""
    bars, k = _bars(), 300
    expected, _ = replay_signals(bars, momentum_spec(), {"lookback": 10})
    trader = PaperTrader(ReplayFeed(bars[k:]), _factory(), history=bars[:k])
    seen = []
    on_signal = trader.engine.portfolio.on_signal
    trader.engine.portfolio.on_signal = lambda e: (seen.append(e), on_signal(e))
    trader.run()
    assert trader.discarded_signals > 0
    assert {e.timestamp: [e.target_weight] for e in seen} == {
        t: w for t, w in expected.items() if t >= k
    }


def test_nothing_trades_on_the_history():
    bars, k = _bars(), 300
    trader = PaperTrader(ReplayFeed(bars[k:]), _factory(), history=bars[:k])
    result = trader.run()
    assert result.timestamps[0] == k
    assert result.equity[0] == result.initial_capital
    assert all(fill.timestamp > k for fill in result.fills)


def test_a_live_strategy_is_never_shown_a_bar_before_it_arrives():
    holder = {}

    def factory(events, data):
        holder["s"] = RecordingStrategy(events, data)
        return holder["s"]

    trader = PaperTrader(ReplayFeed(_bars(80)[40:]), factory, history=_bars(80)[:40])
    trader.run()
    assert holder["s"].observations
    assert all(seen <= now for now, seen in holder["s"].observations)
    with pytest.raises(LookAheadError):
        trader.data.peek_ahead("SYNTH")


def test_bars_out_of_time_order_are_refused():
    bars = _bars(10)
    feed = ReplayFeed([bars[5], bars[3]])
    with pytest.raises(FeedError):
        PaperTrader(feed, _factory()).run()


def test_polling_emits_each_new_bar_once_and_catches_up():
    bars = _bars(6)
    replies = [bars[0], bars[0], None, bars[:3], [bars[2], bars[3]], bars[1], None]
    calls = []

    def fetch():
        calls.append(1)
        return replies[len(calls) - 1] if len(calls) <= len(replies) else None

    clock = iter(range(1000))
    feed = PollingFeed(
        fetch, interval=5, idle_timeout=3, sleep=lambda s: None, clock=lambda: next(clock)
    )
    got = []
    while (bar := feed.next_bar()) is not None:
        got.append(bar.timestamp)
    assert got == [0, 1, 2, 3]


def test_polling_gives_up_after_the_idle_timeout():
    now = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    feed = PollingFeed(
        lambda: None, interval=10, idle_timeout=35, sleep=sleep, clock=lambda: now[0]
    )
    assert feed.next_bar() is None
    assert slept == [10, 10, 10, 10]


def test_polling_needs_dated_bars():
    bar = replace(_bars(2)[0], time=None)
    with pytest.raises(FeedError):
        PollingFeed(lambda: bar, idle_timeout=0).next_bar()


def _write_rows(path, rows, mode="a"):
    with path.open(mode) as handle:
        handle.write(rows)


def test_a_followed_csv_delivers_rows_as_they_are_appended(tmp_path):
    path = tmp_path / "live.csv"
    _write_rows(path, "Date,Close,Volume\n2024-01-01,10,100\n", "w")
    feed = CsvTailFeed(path, idle_timeout=0)
    assert feed.next_bar().close == 10
    assert feed.next_bar() is None
    _write_rows(path, "2024-01-02,11,100\n2024-01-03,1")  # last line half-written
    assert feed.next_bar().close == 11
    assert feed.next_bar() is None
    _write_rows(path, "2,100\n")
    assert feed.next_bar().close == 12


def test_a_followed_csv_that_rewrites_a_traded_row_is_refused(tmp_path):
    path = tmp_path / "live.csv"
    _write_rows(path, "Date,Close,Volume\n2024-01-01,10,100\n", "w")
    feed = CsvTailFeed(path, idle_timeout=0)
    feed.next_bar()
    _write_rows(path, "Date,Close,Volume\n2024-01-01,10.5,100\n2024-01-02,11,100\n", "w")
    with pytest.raises(FeedError):
        feed.next_bar()


def test_a_missing_or_empty_file_is_waited_for(tmp_path):
    path = tmp_path / "later.csv"
    feed = CsvTailFeed(path, idle_timeout=0)
    assert feed.next_bar() is None
    path.write_text("")
    assert feed.next_bar() is None


def test_the_journal_has_a_line_per_bar_matching_the_book(tmp_path):
    bars, journal = _bars(400), tmp_path / "paper.jsonl"
    result = PaperTrader(
        ReplayFeed(bars[200:]),
        _factory(),
        history=bars[:200],
        journal=journal,
        **_costs(),
    ).run()
    rows = read_journal(journal)
    assert rows[0]["type"] == "start" and rows[0]["history_bars"] == 200
    assert [r["index"] for r in rows[1:]] == list(range(200, 400))
    assert rows[-1]["cash"] == result.final_cash
    assert rows[-1]["position"] == result.final_position
    assert sum(len(r["fills"]) for r in rows[1:]) == result.n_trades


@pytest.mark.parametrize("timing", ["next_open", "close"])
def test_resuming_from_the_journal_continues_exactly(tmp_path, timing):
    bars, k = _bars(), 200
    settings = dict(fill_timing=timing, financing=Financing(cash_rate=0.03), **_costs())
    whole_journal = tmp_path / "whole.jsonl"
    whole = PaperTrader(
        ReplayFeed(bars[k:]),
        _factory(),
        history=bars[:k],
        journal=whole_journal,
        **settings,
    ).run()
    # Stop just after a bar that left an order working, when there is one.
    rows = read_journal(whole_journal)[1:]
    pending = [r["index"] for r in rows if r["working"] and r["index"] >= 350]
    cut = (pending[0] if pending else 399) + 1
    assert pending or timing == "close"

    journal = tmp_path / "paper.jsonl"
    first = PaperTrader(
        ReplayFeed(bars[k:]), _factory(), history=bars[:k], journal=journal, **settings
    )
    first.run(max_bars=cut - k)
    second = PaperTrader(
        ReplayFeed(bars[cut:]),
        _factory(),
        history=bars[:cut],
        journal=journal,
        resume=True,
        **settings,
    ).run()
    assert second.final_cash == whole.final_cash
    assert second.final_position == whole.final_position
    assert second.fills == [f for f in whole.fills if f.timestamp >= cut]
    assert second.equity[1:] == whole.equity[cut - k :]
    assert read_journal(journal) == read_journal(whole_journal)


def test_resuming_needs_history_up_to_the_last_journaled_bar(tmp_path):
    bars, journal = _bars(300), tmp_path / "paper.jsonl"
    PaperTrader(
        ReplayFeed(bars[100:200]), _factory(), history=bars[:100], journal=journal
    ).run()
    with pytest.raises(ValueError, match="history must run up to"):
        PaperTrader(
            ReplayFeed(bars[250:]),
            _factory(),
            history=bars[:250],
            journal=journal,
            resume=True,
        )


def test_the_live_handler_numbers_bars_on_from_the_history():
    from queue import Queue

    bars = _bars(5)
    data = LiveDataHandler(Queue(), ReplayFeed(bars[2:]), history=bars[:2], symbol="X")
    stamps = []
    while True:
        data.update_bars()
        if not data.continue_backtest:
            break
        stamps.append((data.current_bar("X").timestamp, data.in_history))
    assert stamps == [(0, True), (1, False), (2, False), (3, False), (4, False)]
