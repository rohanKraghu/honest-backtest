"""Portfolio and position tracking with explicit cash accounting.

The portfolio is the only component that touches money. It does three jobs:

* marks the book to market on every bar and records the equity curve;
* turns a strategy's target *weight* into an :class:`OrderEvent` for the
  *difference* between current and target position;
* applies fills to cash and positions.

The accounting identity ``equity == cash + position * close`` holds after
every event, and the test suite asserts it bar by bar. That is not a
formality: a backtester that quietly loses track of cash will report
whatever the author hoped for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Queue

from .data import DataHandler
from .events import FillEvent, MarketEvent, OrderEvent, SignalEvent


@dataclass
class PortfolioSnapshot:
    """The book at one point in time.

    Attributes:
        timestamp: Bar index.
        cash: Uninvested cash.
        position: Units held (negative when short).
        price: Mark price used.
        equity: ``cash + position * price``.
    """

    timestamp: int
    cash: float
    position: float
    price: float
    equity: float


@dataclass
class Portfolio:
    """Tracks cash, position and equity for a single instrument.

    Args:
        events: Engine event queue.
        data: Data handler, used only to mark the book at the current bar.
        symbol: Instrument traded.
        initial_capital: Starting cash.
        rebalance_threshold: Minimum trade size, as a fraction of equity,
            below which a rebalance is skipped. Without a band, a continuous
            target weight generates a trade every single bar and the result
            is dominated by dust trades. Real desks use a band; so does this.
        allow_fractional: Whether fractional units may be traded. ``True``
            here, which is a simplification: it removes lot-size friction.
            For a liquid, high-priced instrument the difference is small, and
            rounding would add a noise term unrelated to the point of the
            study.
    """

    events: Queue
    data: DataHandler
    symbol: str = "SYNTH"
    initial_capital: float = 1_000_000.0
    rebalance_threshold: float = 0.05
    allow_fractional: bool = True

    cash: float = field(init=False)
    position: float = field(init=False, default=0.0)
    history: list[PortfolioSnapshot] = field(init=False, default_factory=list)
    fills: list[FillEvent] = field(init=False, default_factory=list)
    total_commission: float = field(init=False, default=0.0)
    total_slippage: float = field(init=False, default=0.0)
    traded_notional: float = field(init=False, default=0.0)
    n_trades: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.rebalance_threshold < 0:
            raise ValueError("rebalance_threshold must be non-negative")
        self.cash = float(self.initial_capital)

    def equity_at(self, price: float) -> float:
        """Mark-to-market equity if the instrument were worth ``price``."""
        return self.cash + self.position * price

    def on_market(self, event: MarketEvent) -> None:
        """Mark the book at the new bar's close and append to the equity curve.

        Called before the strategy sees the bar, so the recorded equity
        reflects the position that was actually held into this bar. Costs
        paid on this bar therefore show up in the *next* point of the curve,
        which is the correct attribution.
        """
        bar = self.data.current_bar(self.symbol)
        self.history.append(
            PortfolioSnapshot(
                timestamp=event.timestamp,
                cash=self.cash,
                position=self.position,
                price=bar.close,
                equity=self.equity_at(bar.close),
            )
        )

    def on_signal(self, event: SignalEvent) -> None:
        """Translate a target weight into an order for the position delta."""
        bar = self.data.current_bar(self.symbol)
        price = bar.close
        if price <= 0:
            return
        equity = self.equity_at(price)
        if equity <= 0:
            # Account is wiped out; refuse to trade rather than pretend
            # a negative-equity book can still take risk.
            return

        target_units = event.target_weight * equity / price
        if not self.allow_fractional:
            target_units = float(int(target_units))
        delta = target_units - self.position
        # Two separate guards. The band is a policy choice; the epsilon is a
        # correctness one -- a zero-size order would otherwise become a
        # zero-size fill and inflate the trade count, which is exactly the
        # kind of quiet miscount that makes turnover figures untrustworthy.
        if delta == 0.0 or abs(delta * price) < max(
            self.rebalance_threshold * equity, 1e-9
        ):
            return

        direction = "BUY" if delta > 0 else "SELL"
        self.events.put(
            OrderEvent(
                symbol=self.symbol,
                timestamp=event.timestamp,
                quantity=abs(delta),
                direction=direction,
            )
        )

    def on_fill(self, event: FillEvent) -> None:
        """Apply a fill to cash and position, and accumulate cost statistics."""
        self.cash -= event.signed_quantity * event.fill_price
        self.cash -= event.commission
        self.position += event.signed_quantity
        self.fills.append(event)

        self.total_commission += event.commission
        self.total_slippage += event.slippage_cost
        self.traded_notional += abs(event.quantity * event.fill_price)
        self.n_trades += 1

    @property
    def equity_curve(self) -> list[float]:
        """Equity at each bar, in order."""
        return [s.equity for s in self.history]

    @property
    def timestamps(self) -> list[int]:
        """Bar index at each point of the equity curve."""
        return [s.timestamp for s in self.history]

    @property
    def positions(self) -> list[float]:
        """Position held into each bar."""
        return [s.position for s in self.history]
