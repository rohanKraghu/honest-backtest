"""Several instruments at once: a panel handler, a multi-asset book, a strategy.

The single-symbol engine is a stated limitation of the original design, and
it rules out the strategies most people actually run: anything that ranks,
pairs or rotates between instruments. This module lifts it without touching
the single-symbol path, which the headline result runs on.

The point-in-time guarantee carries over unchanged. Every instrument shares
one cursor, which advances once per bar for all of them together, so no
instrument can be read further ahead than another. That needs the
instruments to share a calendar; :func:`align_panel` builds one from dated
bars by keeping only the dates every instrument traded.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from queue import Queue

from .csvdata import load_csv_bars
from .data import Bar, DataHandler, LookAheadError
from .events import FillEvent, MarketEvent, OrderEvent, SignalEvent
from .financing import Financing
from .strategy import Strategy

#: Instrument name to its bars, oldest first.
Panel = Mapping[str, Sequence[Bar]]


def align_panel(panel: Panel) -> dict[str, list[Bar]]:
    """Put every instrument on one calendar, re-indexed from 0.

    With dated bars (``Bar.time`` set, as :func:`load_csv_bars` does), only
    the dates on which every instrument has a bar are kept: a missing day
    for one instrument removes that day for all, rather than inventing a
    price for it. Undated bars must already line up, bar for bar.

    Raises:
        ValueError: If the panel is empty, or undated series differ in length.
    """
    if not panel:
        raise ValueError("the panel has no instruments")
    dated = all(b.time is not None for bars in panel.values() for b in bars)
    if dated:
        common: set[datetime] = set.intersection(
            *({b.time for b in bars} for bars in panel.values())  # type: ignore[misc]
        )
        out = {}
        for symbol, bars in panel.items():
            kept = [b for b in bars if b.time in common]
            out[symbol] = [replace(b, timestamp=i) for i, b in enumerate(kept)]
        return out
    lengths = {len(bars) for bars in panel.values()}
    if len(lengths) != 1:
        raise ValueError(
            "undated series must have the same length; give the bars dates "
            "so they can be aligned on a calendar"
        )
    return {
        symbol: [replace(b, timestamp=i) for i, b in enumerate(bars)]
        for symbol, bars in panel.items()
    }


def load_csv_panel(
    paths: Sequence[str | Path], *, use_adjusted: bool = True, default_volume=None
) -> dict[str, list[Bar]]:
    """Load one CSV per instrument and align them on their common dates.

    Each file's stem (upper-cased) names its instrument, exactly as
    :func:`~honest_backtest.csvdata.load_csv_bars` does for one file.
    """
    panel = {}
    for path in paths:
        bars = load_csv_bars(
            path, use_adjusted=use_adjusted, default_volume=default_volume
        )
        if bars[0].symbol in panel:
            raise ValueError(f"two files are both named {bars[0].symbol}")
        panel[bars[0].symbol] = bars
    return align_panel(panel)


class MultiAssetDataHandler(DataHandler):
    """Point-in-time access to several instruments on one shared cursor.

    Args:
        events: Engine event queue.
        panel: Bars per instrument. Every series must have the same
            timestamps in the same order (see :func:`align_panel`).
    """

    def __init__(self, events: Queue, panel: Panel) -> None:
        if not panel:
            raise ValueError("the panel has no instruments")
        self.symbols = list(panel)
        self._bars = {s: list(panel[s]) for s in self.symbols}
        self._closes = {s: [b.close for b in self._bars[s]] for s in self.symbols}
        stamps = [b.timestamp for b in self._bars[self.symbols[0]]]
        for symbol in self.symbols[1:]:
            if [b.timestamp for b in self._bars[symbol]] != stamps:
                raise ValueError(
                    f"{symbol} is not on the same calendar as {self.symbols[0]}; "
                    "align the panel first"
                )
        self._stamps = stamps
        self._events = events
        self._cursor = -1
        self.continue_backtest = len(stamps) > 0

    def __len__(self) -> int:
        return len(self._stamps)

    @property
    def cursor(self) -> int:
        """Index of the bar currently being processed; ``-1`` before the first."""
        return self._cursor

    def _check(self, symbol: str) -> None:
        if symbol not in self._bars:
            raise KeyError(
                f"unknown symbol {symbol!r}; this handler serves {self.symbols}"
            )
        if self._cursor < 0:
            raise LookAheadError(
                "no bar has been delivered yet; call update_bars() before "
                "requesting market data"
            )

    def update_bars(self) -> None:
        """Advance every instrument by one bar and emit one :class:`MarketEvent`."""
        if self._cursor + 1 >= len(self._stamps):
            self.continue_backtest = False
            return
        self._cursor += 1
        self._events.put(MarketEvent(self._stamps[self._cursor]))

    def get_latest_bars(self, symbol: str, n: int = 1) -> list[Bar]:
        """Return up to ``n`` bars of ``symbol`` ending at the cursor."""
        self._check(symbol)
        if n <= 0:
            raise ValueError("n must be positive")
        return self._bars[symbol][max(0, self._cursor - n + 1) : self._cursor + 1]

    def current_bar(self, symbol: str) -> Bar:
        """Return ``symbol``'s bar at the cursor."""
        self._check(symbol)
        return self._bars[symbol][self._cursor]

    def latest_closes(self, symbol: str, n: int) -> list[float]:
        """Return up to ``n`` closes of ``symbol`` ending at the cursor."""
        self._check(symbol)
        if n <= 0:
            raise ValueError("n must be positive")
        return self._closes[symbol][max(0, self._cursor - n + 1) : self._cursor + 1]


@dataclass
class MultiAssetSnapshot:
    """The book at one point in time, for several instruments.

    Attributes:
        timestamp: Bar index.
        cash: Uninvested cash.
        positions: Units held per instrument (negative when short).
        prices: Mark price per instrument.
        equity: ``cash + sum(position * price)``.
    """

    timestamp: int
    cash: float
    positions: dict[str, float]
    prices: dict[str, float]
    equity: float

    @property
    def gross_exposure(self) -> float:
        """Sum of absolute position values."""
        return sum(abs(q * self.prices[s]) for s, q in self.positions.items())


@dataclass
class MultiAssetPortfolio:
    """Cash, positions and equity across several instruments.

    Behaves like :class:`~honest_backtest.portfolio.Portfolio`, one
    instrument at a time: a signal's target weight is a fraction of the
    whole book's equity, and only the difference from the current position
    is traded, subject to the same no-trade band. ``financing.max_leverage``
    caps *gross* exposure across the book, so a new position is clipped to
    whatever room the others leave.

    Args:
        events: Engine event queue.
        data: A :class:`MultiAssetDataHandler`.
        initial_capital: Starting cash.
        rebalance_threshold: No-trade band as a fraction of equity.
        limit_offset_bps: Rebalance with passive limit orders this far inside
            the close; ``None`` means market orders.
        limit_expiry_bars: Bars a limit order rests before cancellation.
        financing: Interest, borrow fees and gross leverage limit.
    """

    events: Queue
    data: DataHandler
    initial_capital: float = 1_000_000.0
    rebalance_threshold: float = 0.05
    limit_offset_bps: float | None = None
    limit_expiry_bars: int = 1
    financing: Financing | None = None

    cash: float = field(init=False)
    holdings: dict[str, float] = field(init=False)
    history: list[MultiAssetSnapshot] = field(init=False, default_factory=list)
    fills: list[FillEvent] = field(init=False, default_factory=list)
    total_commission: float = field(init=False, default=0.0)
    total_slippage: float = field(init=False, default=0.0)
    traded_notional: float = field(init=False, default=0.0)
    n_trades: int = field(init=False, default=0)
    total_financing: float = field(init=False, default=0.0)
    # Positions the orders placed this bar are heading for. Signals for
    # several instruments arrive before any of their orders fill, so the
    # leverage limit must count what was just ordered, not only what is held.
    _intended: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Validate settings and start flat."""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.rebalance_threshold < 0:
            raise ValueError("rebalance_threshold must be non-negative")
        self.cash = float(self.initial_capital)
        self.holdings = {s: 0.0 for s in self.data.symbols}

    def _prices(self) -> dict[str, float]:
        return {s: self.data.current_bar(s).close for s in self.data.symbols}

    def equity_at(self, prices: Mapping[str, float]) -> float:
        """Mark-to-market equity at the given prices."""
        return self.cash + sum(q * prices[s] for s, q in self.holdings.items())

    def on_market(self, event: MarketEvent) -> None:
        """Accrue financing for the bar just ended, then mark the whole book."""
        if self.financing is not None and self.history:
            last = self.history[-1].prices
            short_value = sum(max(0.0, -q) * last[s] for s, q in self.holdings.items())
            accrued = self.financing.accrual_for_book(self.cash, short_value)
            self.cash += accrued
            self.total_financing -= accrued
        prices = self._prices()
        self._intended = dict(self.holdings)
        self.history.append(
            MultiAssetSnapshot(
                timestamp=event.timestamp,
                cash=self.cash,
                positions=dict(self.holdings),
                prices=prices,
                equity=self.equity_at(prices),
            )
        )

    def on_signal(self, event: SignalEvent) -> None:
        """Turn one instrument's target weight into an order for the difference."""
        prices = self._prices()
        price = prices[event.symbol]
        equity = self.equity_at(prices)
        if price <= 0 or equity <= 0:
            return
        target_value = event.target_weight * equity
        if self.financing is not None and self.financing.max_leverage is not None:
            others = sum(
                abs(q * prices[s]) for s, q in self._intended.items() if s != event.symbol
            )
            room = max(0.0, self.financing.max_leverage * equity - others)
            target_value = max(-room, min(room, target_value))
        delta = target_value / price - self.holdings[event.symbol]
        if delta == 0.0 or abs(delta * price) < max(
            self.rebalance_threshold * equity, 1e-9
        ):
            return
        self._intended[event.symbol] = target_value / price
        direction = "BUY" if delta > 0 else "SELL"
        if self.limit_offset_bps is None:
            order = OrderEvent(event.symbol, event.timestamp, abs(delta), direction)
        else:
            sign = 1.0 if direction == "BUY" else -1.0
            order = OrderEvent(
                event.symbol,
                event.timestamp,
                abs(delta),
                direction,
                order_type="LMT",
                limit_price=price * (1.0 - sign * self.limit_offset_bps / 10_000.0),
                expiry_bars=self.limit_expiry_bars,
            )
        self.events.put(order)

    def on_fill(self, event: FillEvent) -> None:
        """Apply a fill to cash and that instrument's position."""
        self.cash -= event.signed_quantity * event.fill_price + event.commission
        self.holdings[event.symbol] += event.signed_quantity
        self.fills.append(event)
        self.total_commission += event.commission
        self.total_slippage += event.slippage_cost
        self.traded_notional += abs(event.quantity * event.fill_price)
        self.n_trades += 1

    @property
    def position(self) -> dict[str, float]:
        """Units held per instrument now (the engine reports it at the end)."""
        return dict(self.holdings)

    @property
    def equity_curve(self) -> list[float]:
        """Equity at each bar, in order."""
        return [s.equity for s in self.history]

    @property
    def timestamps(self) -> list[int]:
        """Bar index at each point of the equity curve."""
        return [s.timestamp for s in self.history]

    @property
    def positions(self) -> list[dict[str, float]]:
        """Positions held into each bar."""
        return [s.positions for s in self.history]


class CrossSectionalMomentumStrategy(Strategy):
    """Hold the instruments that have risen most, rebalancing on a schedule.

    Every ``rebalance_every`` bars, rank the instruments by their trailing
    ``lookback``-bar return and hold the top ``top_n`` in equal weight (and,
    with ``long_short``, short the bottom ``top_n``). The textbook
    cross-sectional strategy, and the simplest thing the single-symbol engine
    could not run.

    Args:
        events: Engine event queue.
        data: A :class:`MultiAssetDataHandler`.
        symbol: Ignored; present so the strategy fits a
            :class:`~honest_backtest.spec.StrategySpec` builder.
        lookback: Ranking horizon in bars.
        top_n: Instruments held on each side.
        rebalance_every: Bars between rebalances.
        long_short: Also short the weakest ``top_n``.
        gross: Total gross weight of the book, split equally.
    """

    def __init__(
        self,
        events: Queue,
        data: DataHandler,
        symbol: str | None = None,
        lookback: int = 60,
        top_n: int = 1,
        rebalance_every: int = 21,
        long_short: bool = False,
        gross: float = 1.0,
    ) -> None:
        n = len(data.symbols)
        if lookback < 1 or rebalance_every < 1:
            raise ValueError("lookback and rebalance_every must be >= 1")
        if not 1 <= top_n <= (n // 2 if long_short else n):
            raise ValueError(f"top_n={top_n} does not fit {n} instruments")
        self.events = events
        self.data = data
        self.lookback = int(lookback)
        self.top_n = int(top_n)
        self.rebalance_every = int(rebalance_every)
        self.long_short = bool(long_short)
        self.gross = float(gross)
        self._ranked = 0  # bars with enough history to rank on
        self._last: dict[str, float] = {}

    def calculate_signals(self, event: MarketEvent) -> None:
        """Re-rank and emit one target weight per instrument on rebalance bars."""
        scores = {}
        for symbol in self.data.symbols:
            closes = self.data.latest_closes(symbol, self.lookback + 1)
            if len(closes) < self.lookback + 1 or closes[0] <= 0:
                return
            scores[symbol] = math.log(closes[-1] / closes[0])
        self._ranked += 1
        if (self._ranked - 1) % self.rebalance_every:
            return
        ranked = sorted(scores, key=lambda s: (-scores[s], s))
        sides = 2 if self.long_short else 1
        weight = self.gross / (self.top_n * sides)
        targets = {s: 0.0 for s in self.data.symbols}
        for s in ranked[: self.top_n]:
            targets[s] = weight
        if self.long_short:
            for s in ranked[-self.top_n :]:
                targets[s] = -weight
        # Reductions first, so cash freed by sales is there for the buys.
        held = {s: self._last.get(s, 0.0) for s in targets}
        order = sorted(targets, key=lambda s: abs(targets[s]) - abs(held[s]))
        for s in order:
            self.events.put(SignalEvent(s, event.timestamp, targets[s]))
        self._last = targets
