"""Order execution simulation.

Turns an :class:`~honest_backtest.events.OrderEvent` into a
:class:`~honest_backtest.events.FillEvent` by asking the slippage model for a
price and the commission model for a fee. Splitting the two means the report
can say which friction cost how much, instead of lumping them into one
unfalsifiable "costs" number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from queue import Queue

from .commission import CommissionModel, ZeroCommission
from .data import DataHandler
from .events import FillEvent, OrderEvent
from .slippage import SlippageModel, ZeroSlippage

#: When an order placed on bar ``t`` is filled. See :class:`SimulatedExecutionHandler`.
FILL_TIMINGS = ("close", "next_open", "next_close")


@dataclass
class _Working:
    """An order, or what is left of one, waiting for a later bar."""

    order: OrderEvent
    remaining: float
    expires: int | None  # last bar index on which a limit order may fill
    fresh: bool  # a delayed market order not yet tried, per ``fill_timing``


class SimulatedExecutionHandler:
    """Fills orders against bars, by default completely at the current close.

    Filling at the close of the bar the signal was computed from is the
    default because it is what most backtests do, and it is optimistic: the
    close is only known once it has printed, by which time it can no longer
    be traded at. ``fill_timing`` removes that assumption:

    * ``"close"``: fill at bar ``t``'s close (the optimistic default).
    * ``"next_open"``: fill at bar ``t + 1``'s open. Realistic when the data
      has real opens; with opens derived from the previous close it is
      identical to ``"close"``.
    * ``"next_close"``: fill at bar ``t + 1``'s close. The conservative
      choice when no real open is available.

    Delayed orders are filled before anything else happens on the next bar,
    so the portfolio's snapshot of that bar already includes the fill. An
    order still pending when the data ends is never filled.

    Two further optimistic defaults can be switched off:

    * **Unlimited liquidity.** With ``max_participation`` set, no fill may
      exceed that fraction of the bar's volume. The rest carries to the next
      bar, filling at that bar's price, until it is done.
    * **Market orders only.** A ``"LMT"`` order rests from the next bar for
      ``expiry_bars`` bars. It fills only on a bar that trades strictly
      through its limit (a buy needs a low below the limit), at the limit,
      because touching the price says nothing about whether you were at the
      front of the queue. It pays commission but no spread or impact: a
      passive order is the one paid the spread, not paying it.

    A new order for a symbol cancels whatever is still working for it. The
    portfolio sizes every order from the position it actually holds, so a
    leftover would otherwise be traded twice.

    Args:
        events: Engine event queue.
        data: Data handler, used to read the current bar.
        slippage: Slippage model; defaults to :class:`ZeroSlippage`.
        commission: Commission model; defaults to :class:`ZeroCommission`.
        fill_timing: One of :data:`FILL_TIMINGS`.
        max_participation: Largest fraction of a bar's volume one fill may
            take, or ``None`` for no limit.
    """

    def __init__(
        self,
        events: Queue,
        data: DataHandler,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
        fill_timing: str = "close",
        max_participation: float | None = None,
    ) -> None:
        if fill_timing not in FILL_TIMINGS:
            raise ValueError(
                f"fill_timing must be one of {FILL_TIMINGS}, got {fill_timing!r}"
            )
        if max_participation is not None and not 0 < max_participation <= 1:
            raise ValueError("max_participation must be in (0, 1]")
        self.events = events
        self.data = data
        self.slippage = slippage or ZeroSlippage()
        self.commission = commission or ZeroCommission()
        self.fill_timing = fill_timing
        self.max_participation = max_participation
        self._working: list[_Working] = []
        self.cancelled_quantity = 0.0
        self.expired_quantity = 0.0

    def execute_order(self, order: OrderEvent) -> None:
        """Accept ``order``: fill what can be filled now, keep the rest working."""
        self._cancel(order.symbol)
        bar = self.data.current_bar(order.symbol)
        if order.order_type == "LMT":
            self._working.append(
                _Working(order, order.quantity, bar.timestamp + order.expiry_bars, False)
            )
        elif self.fill_timing == "close":
            self._fill_market(
                _Working(order, order.quantity, None, False), bar, emit=True
            )
        else:
            self._working.append(_Working(order, order.quantity, None, True))

    def working_orders(self) -> list[tuple[OrderEvent, float, int | None, bool]]:
        """What is still working, as ``(order, remaining, expires, fresh)``.

        For saving a paper-trading session; :meth:`restore_working` puts
        these back after a restart.
        """
        return [(w.order, w.remaining, w.expires, w.fresh) for w in self._working]

    def restore_working(
        self, items: Sequence[tuple[OrderEvent, float, int | None, bool]]
    ) -> None:
        """Replace what is working with orders saved by :meth:`working_orders`."""
        self._working = [_Working(*item) for item in items]

    def fill_pending(self) -> list[FillEvent]:
        """Fill what is still working, against the bar that just arrived.

        The engine calls this when a new bar arrives, before the portfolio or
        the strategy sees it, and applies the returned fills directly.
        """
        working, self._working = self._working, []
        fills: list[FillEvent] = []
        for item in working:
            bar = self.data.current_bar(item.order.symbol)
            if item.order.order_type == "LMT":
                fill = self._fill_limit(item, bar)
            else:
                if item.fresh and self.fill_timing == "next_open":
                    # Price the fill off the open: the slippage and commission
                    # models read the reference price from ``close``.
                    bar = replace(bar, close=bar.open)
                item.fresh = False
                fill = self._fill_market(item, bar, emit=False)
            if fill is not None:
                fills.append(fill)
        return fills

    def _cancel(self, symbol: str) -> None:
        keep = []
        for item in self._working:
            if item.order.symbol == symbol:
                self.cancelled_quantity += item.remaining
            else:
                keep.append(item)
        self._working = keep

    def _capacity(self, bar) -> float:
        if self.max_participation is None:
            return float("inf")
        return self.max_participation * max(bar.volume, 0.0)

    def _fill_market(self, item: _Working, bar, *, emit: bool) -> FillEvent | None:
        quantity = min(item.remaining, self._capacity(bar))
        item.remaining -= quantity
        if item.remaining > 1e-12:
            self._working.append(item)
        if quantity <= 0:
            return None
        part = item.order
        if quantity != part.quantity:
            part = part.with_quantity(quantity)
        fill = self._fill(part, bar, timestamp=bar.timestamp)
        if emit:
            self.events.put(fill)
            return None
        return fill

    def _fill_limit(self, item: _Working, bar) -> FillEvent | None:
        order, limit = item.order, item.order.limit_price
        assert limit is not None
        crossed = bar.low < limit if order.direction == "BUY" else bar.high > limit
        fill = None
        if crossed:
            quantity = min(item.remaining, self._capacity(bar))
            if quantity > 0:
                item.remaining -= quantity
                fill = FillEvent(
                    symbol=order.symbol,
                    timestamp=bar.timestamp,
                    quantity=quantity,
                    direction=order.direction,
                    fill_price=limit,
                    commission=self.commission.calculate(
                        order.with_quantity(quantity), limit
                    ),
                    reference_price=limit,
                )
        if item.remaining > 1e-12:
            if item.expires is not None and bar.timestamp >= item.expires:
                self.expired_quantity += item.remaining
            else:
                self._working.append(item)
        return fill

    def _fill(self, order: OrderEvent, bar, timestamp: int | None = None) -> FillEvent:
        fill_price = self.slippage.fill_price(order, bar)
        return FillEvent(
            symbol=order.symbol,
            timestamp=order.timestamp if timestamp is None else timestamp,
            quantity=order.quantity,
            direction=order.direction,
            fill_price=fill_price,
            commission=self.commission.calculate(order, fill_price),
            reference_price=bar.close,
        )
