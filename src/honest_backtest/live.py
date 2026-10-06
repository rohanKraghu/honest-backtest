"""Paper trading: the backtested strategy, unchanged, on bars as they arrive.

The engine's docstring claims that the same strategy object could be driven
by a live feed. This module is that claim made concrete and tested. A
:class:`LiveDataHandler` serves bars from a :class:`Feed` through exactly the
accessors the historic handler uses, and :class:`PaperTrader` runs the
engine's own :class:`~honest_backtest.engine.Backtest` loop over it one bar at
a time, filling orders with the same simulated execution, slippage and
commission models a backtest uses. Nothing is re-implemented, so a strategy
cannot behave differently on paper than it did in the audit;
``tests/test_live.py`` checks that a replayed feed gives the backtest's
equity curve and blotter exactly.

Three feeds are provided, none needing a network library:

* :class:`ReplayFeed` replays a list of bars, optionally at a fixed pace.
* :class:`PollingFeed` calls a function you supply (a broker or data
  vendor's API) on an interval and emits each bar it has not seen before.
* :class:`CsvTailFeed` follows a CSV file that something else appends to,
  such as a scheduled download, parsing it with the audit's own checks.

History before the first live bar warms the strategy up: the trader replays
it through the strategy with every signal discarded, so indicators, trailing
statistics and any other state are exactly what they would be at that point
in a backtest, and nothing trades on bars that are already in the past.

Every bar is appended to a journal (JSON lines, flushed per bar), and a
trader started with ``resume=True`` restores cash, position, running costs
and working orders from it, so a restart picks up where the last bar left
off.
"""

from __future__ import annotations

import io
import json
import time as _time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, TextIO

from .commission import CommissionModel, ZeroCommission
from .csvdata import read_csv_rows
from .data import Bar, HistoricBarDataHandler
from .engine import Backtest, BacktestResult, StrategyFactory
from .events import FillEvent, MarketEvent, OrderEvent, SignalEvent
from .execution import SimulatedExecutionHandler
from .financing import Financing
from .portfolio import Portfolio, PortfolioSnapshot
from .slippage import SlippageModel, ZeroSlippage

#: Version of the journal's line format.
JOURNAL_VERSION = 1


class FeedError(RuntimeError):
    """Raised when a feed delivers bars that cannot be traded on honestly."""


class Feed(ABC):
    """A source of completed bars, oldest first."""

    @abstractmethod
    def next_bar(self) -> Bar | None:
        """Wait for the next completed bar.

        Returns:
            The bar, or ``None`` once the feed has closed.
        """


class ReplayFeed(Feed):
    """Replays a list of bars, one per call, for demonstrations and tests.

    Args:
        bars: Bars to deliver, oldest first.
        pace: Seconds to wait before delivering each bar.
        sleep: The waiting function; tests pass a fake one.
    """

    def __init__(
        self,
        bars: Iterable[Bar],
        *,
        pace: float = 0.0,
        sleep: Callable[[float], None] = _time.sleep,
    ) -> None:
        """Queue up ``bars``."""
        self._bars = list(bars)
        self._next = 0
        self.pace = float(pace)
        self._sleep = sleep

    def next_bar(self) -> Bar | None:
        """Return the next bar, or ``None`` after the last one."""
        if self._next >= len(self._bars):
            return None
        if self.pace > 0:
            self._sleep(self.pace)
        bar = self._bars[self._next]
        self._next += 1
        return bar


#: What a polling function may return: nothing new, one bar, or several.
FetchResult = Bar | Sequence[Bar] | None


class PollingFeed(Feed):
    """Asks a function for bars on an interval and emits each new one once.

    ``fetch`` is where a broker or data vendor plugs in. It should return
    completed bars only (never the bar still forming), each with ``time``
    set; returning the same bar again, or the last few bars every time, is
    fine, because only bars newer than the last one emitted are passed on.
    Returning several bars lets the feed catch up after a pause without
    skipping any.

    Args:
        fetch: Called with no arguments; returns ``None``, a bar or bars.
        interval: Seconds between calls while nothing new has arrived.
        idle_timeout: Close the feed after this many seconds with nothing
            new. ``None`` waits forever; ``0`` returns as soon as a call
            brings nothing new, which drains what is available and stops.
        after: Ignore bars at or before this time (when resuming).
        sleep: The waiting function; tests pass a fake one.
        clock: Monotonic clock for the idle timeout; tests pass a fake one.
    """

    def __init__(
        self,
        fetch: Callable[[], FetchResult],
        *,
        interval: float = 60.0,
        idle_timeout: float | None = None,
        after: datetime | None = None,
        sleep: Callable[[float], None] = _time.sleep,
        clock: Callable[[], float] = _time.monotonic,
    ) -> None:
        """Set up polling; nothing is fetched until :meth:`next_bar`."""
        if interval < 0:
            raise ValueError("interval must be non-negative")
        if idle_timeout is not None and idle_timeout < 0:
            raise ValueError("idle_timeout must be non-negative")
        self._fetch = fetch
        self.interval = float(interval)
        self.idle_timeout = idle_timeout
        self.last_time = after
        self._sleep = sleep
        self._clock = clock
        self._buffer: list[Bar] = []

    def _new_bars(self) -> list[Bar]:
        got = self._fetch()
        if got is None:
            return []
        bars = [got] if isinstance(got, Bar) else list(got)
        fresh = []
        last = self.last_time
        for bar in bars:
            if bar.time is None:
                raise FeedError("a polled bar needs its time set to tell new from old")
            if last is None or bar.time > last:
                fresh.append(bar)
                last = bar.time
        return fresh

    def next_bar(self) -> Bar | None:
        """Return the next unseen bar, polling until one arrives or time runs out."""
        started = self._clock()
        while not self._buffer:
            self._buffer = self._new_bars()
            if self._buffer:
                break
            if (
                self.idle_timeout is not None
                and self._clock() - started >= self.idle_timeout
            ):
                return None
            self._sleep(self.interval)
        bar = self._buffer.pop(0)
        self.last_time = bar.time
        return bar


class CsvTailFeed(PollingFeed):
    """Follows a CSV file that another process appends rows to.

    Each poll re-reads the file's complete lines with the same parser and
    checks as :func:`~honest_backtest.csvdata.load_csv_bars` (a line still
    being written is left for the next poll) and emits rows newer than the
    last one emitted. A row that changes after it was delivered raises
    :class:`FeedError`: the strategy has already traded on it, so silently
    accepting a revised past would make the paper record unreproducible.
    For the same reason adjusted closes are off by default here; an
    adjustment factor rescales history every time a dividend is paid.

    Args:
        path: The CSV file.
        symbol: Name for the bars. Defaults to the file's stem.
        use_adjusted: Scale prices by an adjusted-close column.
        default_volume: Volume to assume when the file has none.
        **polling: Passed to :class:`PollingFeed` (``interval``,
            ``idle_timeout``, ``after``, ``sleep``, ``clock``).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        symbol: str | None = None,
        use_adjusted: bool = False,
        default_volume: float | None = None,
        **polling: Any,
    ) -> None:
        """Follow ``path``; it may not exist yet."""
        polling.setdefault("interval", 1.0)
        super().__init__(self._read, **polling)
        self.path = Path(path)
        self.symbol = symbol or self.path.stem.upper()
        self.use_adjusted = use_adjusted
        self.default_volume = default_volume
        self._delivered: dict[datetime, Bar] = {}

    def _read(self) -> list[Bar]:
        try:
            text = self.path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return []
        complete = text[: text.rfind("\n") + 1]
        if not complete.strip():
            return []
        bars = read_csv_rows(
            io.StringIO(complete, newline=""),
            self.symbol,
            source=str(self.path),
            use_adjusted=self.use_adjusted,
            default_volume=self.default_volume,
        )
        for bar in bars:
            seen = self._delivered.get(bar.time)  # type: ignore[arg-type]
            if seen is not None and seen != bar:
                raise FeedError(
                    f"{self.path}: the row for {bar.time} changed after it was traded on"
                )
        return bars

    def next_bar(self) -> Bar | None:
        """Return the next new row as a bar, polling the file until one appears."""
        bar = super().next_bar()
        if bar is not None:
            self._delivered[bar.time] = bar  # type: ignore[index]
        return bar


class LiveDataHandler(HistoricBarDataHandler):
    """The point-in-time handler, fed by a :class:`Feed` instead of a list.

    It inherits every accessor from
    :class:`~honest_backtest.data.HistoricBarDataHandler`, so a strategy sees
    exactly the interface and the guarantees it was backtested with. History
    is delivered first, then each bar the feed produces; bars are numbered
    on from the history, in arrival order, and must arrive in time order.

    Args:
        events: The engine's event queue.
        feed: Where new bars come from.
        history: Bars before the first live one, oldest first.
        symbol: The instrument; every bar is stamped with it.
    """

    def __init__(
        self,
        events: Queue,
        feed: Feed,
        history: Sequence[Bar] = (),
        symbol: str = "SYNTH",
    ) -> None:
        """Serve ``history`` first, then whatever ``feed`` produces."""
        bars = [replace(b, timestamp=i, symbol=symbol) for i, b in enumerate(history)]
        super().__init__(events, bars, symbol=symbol)
        self.feed = feed
        self.n_history = len(bars)
        self.continue_backtest = True

    @property
    def last_history_bar(self) -> Bar | None:
        """The final history bar, or ``None`` without history."""
        return self._bars[self.n_history - 1] if self.n_history else None

    @property
    def in_history(self) -> bool:
        """Whether the next bar delivered still comes from the history."""
        return self._cursor + 1 < self.n_history

    def update_bars(self) -> None:
        """Deliver the next history bar, or wait for the feed's next bar."""
        if self._cursor + 1 >= len(self._bars):
            bar = self.feed.next_bar()
            if bar is None:
                self.continue_backtest = False
                return
            last = self._bars[-1].time if self._bars else None
            if last is not None and bar.time is not None and bar.time <= last:
                raise FeedError(
                    f"bar for {bar.time} arrived after {last}; bars must arrive in "
                    "time order"
                )
            bar = replace(bar, timestamp=len(self._bars), symbol=self.symbol)
            self._bars.append(bar)
            self._closes.append(bar.close)
        self._cursor += 1
        self._events.put(MarketEvent(self._bars[self._cursor].timestamp))


@dataclass(frozen=True)
class PaperState:
    """The book after one live bar has been fully processed.

    Attributes:
        bar: The bar just processed.
        cash: Cash after every fill so far.
        position: Units held after every fill so far.
        equity: ``cash + position * bar.close``.
        fills: Fills that happened on this bar.
        working: Orders still working, as saved in the journal.
    """

    bar: Bar
    cash: float
    position: float
    equity: float
    fills: tuple[FillEvent, ...]
    working: tuple[dict, ...]


def _order_to_dict(item: tuple[OrderEvent, float, int | None, bool]) -> dict:
    order, remaining, expires, fresh = item
    return {
        "symbol": order.symbol,
        "timestamp": order.timestamp,
        "quantity": order.quantity,
        "direction": order.direction,
        "order_type": order.order_type,
        "limit_price": order.limit_price,
        "expiry_bars": order.expiry_bars,
        "remaining": remaining,
        "expires": expires,
        "fresh": fresh,
    }


def _order_from_dict(row: dict) -> tuple[OrderEvent, float, int | None, bool]:
    order = OrderEvent(
        symbol=row["symbol"],
        timestamp=row["timestamp"],
        quantity=row["quantity"],
        direction=row["direction"],
        order_type=row["order_type"],
        limit_price=row["limit_price"],
        expiry_bars=row["expiry_bars"],
    )
    return order, row["remaining"], row["expires"], row["fresh"]


def _fill_to_dict(fill: FillEvent) -> dict:
    return {
        "direction": fill.direction,
        "quantity": fill.quantity,
        "price": fill.fill_price,
        "reference_price": fill.reference_price,
        "commission": fill.commission,
        "slippage": fill.slippage_cost,
    }


def read_journal(path: str | Path) -> list[dict]:
    """Read a paper-trading journal: one dict per line, oldest first."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}, line {number}: not valid JSON") from exc
    return rows


class PaperTrader:
    """Run a strategy on a live feed with simulated fills.

    Built from the same parts as :func:`~honest_backtest.engine.run_backtest`
    (portfolio, simulated execution, slippage and commission models) and
    driven by the engine's own loop, so a replayed feed reproduces a
    backtest exactly. Orders fill at the next bar's open by default, because
    a bar's close has already printed by the time the bar is complete.

    Args:
        feed: Where live bars come from.
        strategy_factory: ``(queue, data_handler) -> Strategy``, as for a
            backtest; :meth:`~honest_backtest.spec.StrategySpec.factory`
            builds one from a spec and a parameter setting.
        history: Bars before the first live one, replayed through the
            strategy (signals discarded) to warm it up.
        symbol: Instrument name.
        slippage: Slippage model; ``None`` means zero slippage.
        commission: Commission model; ``None`` means zero commission.
        initial_capital: Starting cash.
        rebalance_threshold: No-trade band as a fraction of equity.
        fill_timing: When orders fill; see
            :class:`~honest_backtest.execution.SimulatedExecutionHandler`.
        max_participation: Cap on each fill as a fraction of bar volume.
        limit_offset_bps: Rebalance with passive limit orders this far
            inside the close.
        limit_expiry_bars: Bars a limit order rests before cancellation.
        financing: Interest, borrow fees and leverage limit.
        bars_per_year: Annualisation factor for the result's metrics.
        journal: File to append one JSON line per bar to.
        journal_meta: Anything else worth recording in the journal's first
            line, such as the strategy's name and setting.
        resume: Restore the book from the last bar in ``journal`` instead of
            starting fresh. ``history`` must end on that bar.
    """

    def __init__(
        self,
        feed: Feed,
        strategy_factory: StrategyFactory,
        *,
        history: Sequence[Bar] = (),
        symbol: str = "SYNTH",
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
        initial_capital: float = 1_000_000.0,
        rebalance_threshold: float = 0.05,
        fill_timing: str = "next_open",
        max_participation: float | None = None,
        limit_offset_bps: float | None = None,
        limit_expiry_bars: int = 1,
        financing: Financing | None = None,
        bars_per_year: int = 252,
        journal: str | Path | None = None,
        journal_meta: Mapping[str, Any] | None = None,
        resume: bool = False,
    ) -> None:
        """Wire the engine to the feed and replay ``history`` to warm up."""
        if resume and journal is None:
            raise ValueError("resume needs the journal to resume from")
        events: Queue = Queue()
        if slippage is not None:
            slippage.reset()
        self.data = LiveDataHandler(events, feed, history=history, symbol=symbol)
        strategy = strategy_factory(events, self.data)
        portfolio = Portfolio(
            events=events,
            data=self.data,
            symbol=symbol,
            initial_capital=initial_capital,
            rebalance_threshold=rebalance_threshold,
            limit_offset_bps=limit_offset_bps,
            limit_expiry_bars=limit_expiry_bars,
            financing=financing,
        )
        execution = SimulatedExecutionHandler(
            events=events,
            data=self.data,
            slippage=slippage or ZeroSlippage(),
            commission=commission or ZeroCommission(),
            fill_timing=fill_timing,
            max_participation=max_participation,
        )
        self.engine = Backtest(
            data=self.data,
            strategy=strategy,
            portfolio=portfolio,
            execution=execution,
            events=events,
            bars_per_year=bars_per_year,
        )
        self.symbol = symbol
        self.fill_timing = fill_timing
        self.discarded_signals = self._warm_up()
        self.bars_processed = 0
        self._journal_path = Path(journal) if journal is not None else None
        self._journal: TextIO | None = None
        if resume:
            self._restore(read_journal(self._journal_path))  # type: ignore[arg-type]
        if self._journal_path is not None:
            fresh = not resume
            self._journal = self._journal_path.open("a" if resume else "w")
            if fresh:
                self._write(
                    {
                        "type": "start",
                        "version": JOURNAL_VERSION,
                        "symbol": symbol,
                        "initial_capital": float(initial_capital),
                        "fill_timing": fill_timing,
                        "history_bars": self.data.n_history,
                        "meta": dict(journal_meta or {}),
                    }
                )

    def _warm_up(self) -> int:
        """Replay the history through the strategy alone, discarding signals."""
        discarded = 0
        events = self.engine.events
        while self.data.in_history:
            self.data.update_bars()
            while True:
                try:
                    event = events.get(block=False)
                except Empty:
                    break
                if isinstance(event, MarketEvent):
                    self.engine.strategy.calculate_signals(event)
                elif isinstance(event, SignalEvent):
                    discarded += 1
        return discarded

    def _restore(self, rows: list[dict]) -> None:
        start = next((r for r in rows if r.get("type") == "start"), None)
        bars = [r for r in rows if r.get("type") == "bar"]
        if start is None or not bars:
            raise ValueError(f"{self._journal_path} has no bars to resume from")
        if start.get("version") != JOURNAL_VERSION:
            raise ValueError("the journal was written by an incompatible version")
        if start["symbol"] != self.symbol or start["fill_timing"] != self.fill_timing:
            raise ValueError(
                "the journal was written for a different symbol or fill timing: "
                f"{start['symbol']} / {start['fill_timing']}"
            )
        last = bars[-1]
        end = self.data.last_history_bar
        if end is None or (last["index"], last["time"]) != (
            end.timestamp,
            _iso(end.time),
        ):
            raise ValueError(
                f"the journal ends on bar {last['index']} ({last['time']}) but the "
                f"history ends on bar {self.data.n_history - 1} "
                f"({_iso(end.time) if end else 'none'}); the history must run up "
                "to the last journaled bar"
            )
        portfolio = self.engine.portfolio
        portfolio.cash = last["cash"]
        portfolio.position = last["position"]
        totals = last["totals"]
        portfolio.total_commission = totals["commission"]
        portfolio.total_slippage = totals["slippage"]
        portfolio.traded_notional = totals["notional"]
        portfolio.n_trades = totals["trades"]
        portfolio.total_financing = totals["financing"]
        # The last snapshot carries the price financing accrues from.
        portfolio.history.append(
            PortfolioSnapshot(
                timestamp=last["index"],
                cash=last["cash"],
                position=last["position"],
                price=last["close"],
                equity=last["equity"],
            )
        )
        self.engine.execution.restore_working(
            [_order_from_dict(row) for row in last["working"]]
        )

    def _write(self, row: dict) -> None:
        if self._journal is None:
            return
        self._journal.write(json.dumps(row, allow_nan=False) + "\n")
        self._journal.flush()

    def step(self) -> PaperState | None:
        """Wait for the next live bar and process it.

        Returns:
            The book after the bar, or ``None`` once the feed has closed.
        """
        portfolio = self.engine.portfolio
        n_fills = len(portfolio.fills)
        if not self.engine.step():
            return None
        self.bars_processed += 1
        bar = self.data.current_bar(self.symbol)
        working = tuple(
            _order_to_dict(item) for item in self.engine.execution.working_orders()
        )
        state = PaperState(
            bar=bar,
            cash=portfolio.cash,
            position=portfolio.position,
            equity=portfolio.cash + portfolio.position * bar.close,
            fills=tuple(portfolio.fills[n_fills:]),
            working=working,
        )
        self._write(
            {
                "type": "bar",
                "index": bar.timestamp,
                "time": _iso(bar.time),
                "close": bar.close,
                "cash": state.cash,
                "position": state.position,
                "equity": state.equity,
                "fills": [_fill_to_dict(f) for f in state.fills],
                "working": list(working),
                "totals": {
                    "commission": portfolio.total_commission,
                    "slippage": portfolio.total_slippage,
                    "notional": portfolio.traded_notional,
                    "trades": portfolio.n_trades,
                    "financing": portfolio.total_financing,
                },
            }
        )
        return state

    def run(
        self,
        max_bars: int | None = None,
        on_bar: Callable[[PaperState], None] | None = None,
    ) -> BacktestResult:
        """Process bars until the feed closes or ``max_bars`` have been handled.

        Args:
            max_bars: Stop after this many live bars; ``None`` for no limit.
            on_bar: Called with the book after each bar, e.g. to print it.

        Returns:
            Everything traded so far, as a backtest result.
        """
        try:
            while max_bars is None or self.bars_processed < max_bars:
                state = self.step()
                if state is None:
                    break
                if on_bar is not None:
                    on_bar(state)
        finally:
            self.close()
        return self.result()

    def result(self) -> BacktestResult:
        """Everything traded so far, as a :class:`BacktestResult`."""
        return self.engine.result()

    def close(self) -> None:
        """Close the journal; the trader can still report its result."""
        if self._journal is not None:
            self._journal.close()
            self._journal = None


def _iso(when: datetime | None) -> str | None:
    return when.isoformat() if when is not None else None
