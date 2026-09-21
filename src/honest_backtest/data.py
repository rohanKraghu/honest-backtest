"""Market data handlers, and the point-in-time guarantee.

The single most common way a backtest lies is by letting the strategy see
data that did not exist yet.  In a vectorised pandas backtest this is one
mis-signed ``.shift()`` away, and nothing in the code will complain.

Here the guarantee is structural.  :class:`HistoricBarDataHandler` owns the
full price history but exposes it only through accessors that clamp to an
internal cursor.  The cursor advances exactly once per :meth:`update_bars`
call, which is the same call that emits the :class:`~honest_backtest.events.MarketEvent`.
There is no public method that returns a bar the engine has not yet reached.

:class:`LookAheadDataHandler` is the deliberate counterexample: it re-enables
peeking so that stage 1 of the degradation study can reproduce the bug a
naive backtest contains.  It is never used outside that demonstration, and
its name is intended to make its presence obvious in a diff.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import Queue
from typing import Iterable, Sequence

from .events import MarketEvent


class LookAheadError(RuntimeError):
    """Raised when a component tries to read data from the future.

    This is the error that a point-in-time data handler raises instead of
    silently returning tomorrow's price.
    """


@dataclass(frozen=True)
class Bar:
    """A single OHLCV observation.

    ``timestamp`` is an integer bar index for the synthetic generator.  A real
    adapter would carry a ``datetime``; nothing in the engine depends on the
    type beyond ordering and equality, so swapping it is a local change.
    """

    timestamp: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataHandler(ABC):
    """Interface every market data source must implement.

    The interface is deliberately narrow and *pull-based on the past only*.
    A CSV reader, a database cursor or a live websocket feed can all satisfy
    it, because none of the methods require random access to the future.
    Implementing a new source means implementing :meth:`update_bars` and the
    three accessors below; the engine, strategies, portfolio and execution
    handler need no changes.
    """

    symbols: Sequence[str]
    continue_backtest: bool

    @abstractmethod
    def update_bars(self) -> None:
        """Advance to the next bar and emit a :class:`MarketEvent`.

        Implementations must set ``continue_backtest`` to ``False`` once the
        stream is exhausted.
        """

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> list[Bar]:
        """Return up to the ``n`` most recent bars, oldest first.

        Must never return a bar later than the current cursor.  Returning
        fewer than ``n`` bars at the start of the stream is expected; callers
        are responsible for checking the length before computing a signal.
        """

    @abstractmethod
    def current_bar(self, symbol: str) -> Bar:
        """Return the bar the engine is currently processing."""

    @abstractmethod
    def latest_closes(self, symbol: str, n: int) -> list[float]:
        """Convenience accessor returning up to ``n`` recent closing prices."""

    def peek_ahead(self, symbol: str, n: int = 1) -> list[Bar]:
        """Look at future bars.

        The base implementation always raises.  Only the deliberately broken
        :class:`LookAheadDataHandler` overrides it.  Keeping the method on the
        base class (rather than omitting it) means the prohibition is written
        down and testable, instead of being an absence someone could fill in.

        Raises:
            LookAheadError: always, for any point-in-time handler.
        """
        raise LookAheadError(
            f"{type(self).__name__} is a point-in-time data handler and cannot "
            "serve future bars. If you are seeing this, a strategy tried to "
            "look ahead."
        )


class HistoricBarDataHandler(DataHandler):
    """Replays a fixed list of bars, one at a time, with a hard cursor.

    Args:
        events: The engine's event queue; a ``MarketEvent`` is pushed per bar.
        bars: Full history to replay, in ascending timestamp order.
        symbol: The single instrument this handler serves.

    The handler is single-symbol by construction.  Multi-asset support would
    mean replacing the flat list with a per-symbol dict and a merged
    timestamp axis; the interface above already anticipates it (every
    accessor takes a ``symbol``), but implementing it without a real
    multi-asset dataset to test against would be speculative, so it is left
    out. This is a stated limitation, not an oversight.
    """

    def __init__(self, events: Queue, bars: Sequence[Bar], symbol: str = "SYNTH") -> None:
        self._bars: list[Bar] = list(bars)
        self._closes: list[float] = [b.close for b in self._bars]
        self.symbol = symbol
        self.symbols = [symbol]
        self._events = events
        self._cursor = -1
        self.continue_backtest = len(self._bars) > 0

    def __len__(self) -> int:
        return len(self._bars)

    @property
    def cursor(self) -> int:
        """Index of the bar currently being processed; ``-1`` before the first."""
        return self._cursor

    def _check_symbol(self, symbol: str) -> None:
        if symbol != self.symbol:
            raise KeyError(f"unknown symbol {symbol!r}; this handler serves {self.symbol!r}")

    def _check_started(self) -> None:
        if self._cursor < 0:
            raise LookAheadError(
                "no bar has been delivered yet; call update_bars() before "
                "requesting market data"
            )

    def update_bars(self) -> None:
        """Advance the cursor by one bar and emit a :class:`MarketEvent`."""
        if self._cursor + 1 >= len(self._bars):
            self.continue_backtest = False
            return
        self._cursor += 1
        self._events.put(MarketEvent(self._bars[self._cursor].timestamp))

    def get_latest_bars(self, symbol: str, n: int = 1) -> list[Bar]:
        """Return up to ``n`` bars ending at the cursor, oldest first."""
        self._check_symbol(symbol)
        self._check_started()
        if n <= 0:
            raise ValueError("n must be positive")
        start = max(0, self._cursor - n + 1)
        return self._bars[start : self._cursor + 1]

    def current_bar(self, symbol: str) -> Bar:
        """Return the bar at the cursor."""
        self._check_symbol(symbol)
        self._check_started()
        return self._bars[self._cursor]

    def latest_closes(self, symbol: str, n: int) -> list[float]:
        """Return up to ``n`` closing prices ending at the cursor, oldest first."""
        self._check_symbol(symbol)
        self._check_started()
        if n <= 0:
            raise ValueError("n must be positive")
        start = max(0, self._cursor - n + 1)
        return self._closes[start : self._cursor + 1]


class LookAheadDataHandler(HistoricBarDataHandler):
    """A DELIBERATELY BROKEN handler that can serve future bars.

    This exists for exactly one reason: stage 1 of the degradation study needs
    to reproduce what a naive backtest actually does, and the honest way to
    show that is to run the bug rather than describe it.

    Two leaks are available:

    * :meth:`peek_ahead` returns bars strictly after the cursor. This is the
      structural equivalent of aligning a signal to the same bar's return --
      the classic off-by-one ``.shift()`` error.
    * :meth:`full_sample_stats` returns the mean and standard deviation of the
      *entire* price history, including the part that has not happened yet.
      Standardising a feature with full-sample statistics is a subtler leak
      that survives in a lot of published research.

    Never use this class for anything you intend to believe.
    """

    def peek_ahead(self, symbol: str, n: int = 1) -> list[Bar]:
        """Return up to ``n`` bars *after* the cursor. This is the bug.

        Returns:
            Future bars, oldest first. Empty near the end of the stream.
        """
        self._check_symbol(symbol)
        self._check_started()
        if n <= 0:
            raise ValueError("n must be positive")
        end = min(len(self._bars), self._cursor + 1 + n)
        return self._bars[self._cursor + 1 : end]

    def all_closes(self, symbol: str) -> list[float]:
        """Return *every* close in the sample, past and future. This is the bug.

        Used by the naive strategy to standardise its signal with full-sample
        statistics, a leak that is easy to write and hard to spot in review.
        """
        self._check_symbol(symbol)
        return list(self._closes)

    def full_sample_stats(self, symbol: str) -> tuple[float, float]:
        """Return the mean and population stdev of every close in the sample.

        Including the future ones. That is the point.
        """
        self._check_symbol(symbol)
        n = len(self._closes)
        if n == 0:
            return 0.0, 1.0
        mean = sum(self._closes) / n
        var = sum((c - mean) ** 2 for c in self._closes) / n
        return mean, var**0.5


def bars_from_series(
    closes: Iterable[float],
    symbol: str = "SYNTH",
    volumes: Iterable[float] | None = None,
    start_index: int = 0,
) -> list[Bar]:
    """Build a list of :class:`Bar` from a close-price series.

    Open/high/low are derived from consecutive closes rather than simulated
    independently.  The strategies in this project trade at the close, so
    intrabar path is not used for execution; fabricating a richer intrabar
    path would add realism the engine does not consume.

    Args:
        closes: Closing prices in ascending time order.
        symbol: Instrument name to stamp on every bar.
        volumes: Optional matching volume series; defaults to a flat 1e6.
        start_index: Timestamp of the first bar.

    Returns:
        Bars with ascending integer timestamps.
    """
    closes = list(closes)
    vols = list(volumes) if volumes is not None else [1_000_000.0] * len(closes)
    if len(vols) != len(closes):
        raise ValueError("volumes must match closes in length")
    bars: list[Bar] = []
    prev = closes[0] if closes else 0.0
    for i, (c, v) in enumerate(zip(closes, vols)):
        o = prev
        bars.append(
            Bar(
                timestamp=start_index + i,
                symbol=symbol,
                open=o,
                high=max(o, c),
                low=min(o, c),
                close=c,
                volume=v,
            )
        )
        prev = c
    return bars
