"""Event types flowing through the backtester's central queue.

The engine is event-driven rather than vectorised: every bar of market data
is pushed onto a queue as a :class:`MarketEvent`, and components react by
pushing further events.  The canonical causal chain is::

    MarketEvent -> SignalEvent -> OrderEvent -> FillEvent

Modelling this explicitly costs performance relative to a vectorised pandas
backtest, but it buys two things that matter for honesty:

1.  A component can only ever see the events it has actually received, which
    makes look-ahead bias an architectural impossibility rather than a
    convention that a careless ``.shift()`` can break.
2.  The same strategy object could be pointed at a live feed, because it
    consumes bars one at a time instead of an entire DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class EventType(str, Enum):
    """Discriminator for events on the queue."""

    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


Direction = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class Event:
    """Base class for all events. Frozen so that no consumer can mutate history."""

    type: EventType


@dataclass(frozen=True)
class MarketEvent(Event):
    """Signals that a new bar has become available for ``timestamp``.

    Deliberately carries no price payload.  Consumers must ask the
    :class:`~honest_backtest.data.DataHandler` for bars, which is the single
    chokepoint where point-in-time discipline is enforced.
    """

    timestamp: int

    def __init__(self, timestamp: int) -> None:
        object.__setattr__(self, "type", EventType.MARKET)
        object.__setattr__(self, "timestamp", timestamp)


@dataclass(frozen=True)
class SignalEvent(Event):
    """A strategy's desired exposure, expressed as a target portfolio weight.

    Using a target *weight* rather than a share count keeps the strategy free
    of accounting concerns; the portfolio is responsible for translating the
    weight into an order given current equity and price.

    Attributes:
        symbol: Instrument the signal refers to.
        timestamp: Bar at which the signal was formed.
        target_weight: Desired fraction of portfolio equity, in ``[-1, 1]``
            for an unlevered long/short book.
        strength: Optional diagnostic (e.g. the raw z-score) carried through
            for logging; it never affects execution.
    """

    symbol: str
    timestamp: int
    target_weight: float
    strength: float = 0.0

    def __init__(
        self,
        symbol: str,
        timestamp: int,
        target_weight: float,
        strength: float = 0.0,
    ) -> None:
        object.__setattr__(self, "type", EventType.SIGNAL)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "target_weight", float(target_weight))
        object.__setattr__(self, "strength", float(strength))


@dataclass(frozen=True)
class OrderEvent(Event):
    """An instruction to trade ``quantity`` units of ``symbol``.

    Attributes:
        symbol: Instrument to trade.
        timestamp: Bar at which the order is submitted.
        quantity: Absolute number of units (always non-negative).
        direction: ``"BUY"`` or ``"SELL"``.
        order_type: Only ``"MKT"`` is simulated; limit orders would require
            modelling queue position, which this project does not claim to do.
    """

    symbol: str
    timestamp: int
    quantity: float
    direction: Direction
    order_type: str = "MKT"

    def __init__(
        self,
        symbol: str,
        timestamp: int,
        quantity: float,
        direction: Direction,
        order_type: str = "MKT",
    ) -> None:
        if quantity < 0:
            raise ValueError("OrderEvent.quantity must be non-negative; use direction")
        if direction not in ("BUY", "SELL"):
            raise ValueError(f"unknown direction {direction!r}")
        object.__setattr__(self, "type", EventType.ORDER)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "quantity", float(quantity))
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "order_type", order_type)

    @property
    def signed_quantity(self) -> float:
        """Quantity with sign applied (positive for a buy)."""
        return self.quantity if self.direction == "BUY" else -self.quantity


@dataclass(frozen=True)
class FillEvent(Event):
    """The simulated result of an order reaching the market.

    ``fill_price`` is the price *after* slippage, so ``fill_price`` minus the
    reference price is the implicit cost; ``commission`` is the explicit cost.
    Keeping them separate lets the cost attribution in the report be honest
    about which friction did the damage.
    """

    symbol: str
    timestamp: int
    quantity: float
    direction: Direction
    fill_price: float
    commission: float
    reference_price: float

    def __init__(
        self,
        symbol: str,
        timestamp: int,
        quantity: float,
        direction: Direction,
        fill_price: float,
        commission: float,
        reference_price: float,
    ) -> None:
        object.__setattr__(self, "type", EventType.FILL)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "quantity", float(quantity))
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "fill_price", float(fill_price))
        object.__setattr__(self, "commission", float(commission))
        object.__setattr__(self, "reference_price", float(reference_price))

    @property
    def signed_quantity(self) -> float:
        """Quantity with sign applied (positive for a buy)."""
        return self.quantity if self.direction == "BUY" else -self.quantity

    @property
    def slippage_cost(self) -> float:
        """Cash lost to adverse fill price relative to the reference price."""
        return abs(self.fill_price - self.reference_price) * self.quantity
