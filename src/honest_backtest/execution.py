"""Order execution simulation.

Turns an :class:`~honest_backtest.events.OrderEvent` into a
:class:`~honest_backtest.events.FillEvent` by asking the slippage model for a
price and the commission model for a fee. Splitting the two means the report
can say which friction cost how much, instead of lumping them into one
unfalsifiable "costs" number.
"""

from __future__ import annotations

from dataclasses import replace
from queue import Queue

from .commission import CommissionModel, ZeroCommission
from .data import DataHandler
from .events import FillEvent, OrderEvent
from .slippage import SlippageModel, ZeroSlippage

#: When an order placed on bar ``t`` is filled. See :class:`SimulatedExecutionHandler`.
FILL_TIMINGS = ("close", "next_open", "next_close")


class SimulatedExecutionHandler:
    """Fills market orders, by default immediately at the current bar's close.

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

    One more optimistic assumption is worth naming: unlimited fill
    certainty, so an order never goes unfilled or partially filled.
    Modelling rejection would require a liquidity model this project does
    not have. The slippage model carries the size penalty instead.

    Args:
        events: Engine event queue.
        data: Data handler, used to read the current bar.
        slippage: Slippage model; defaults to :class:`ZeroSlippage`.
        commission: Commission model; defaults to :class:`ZeroCommission`.
        fill_timing: One of :data:`FILL_TIMINGS`.
    """

    def __init__(
        self,
        events: Queue,
        data: DataHandler,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
        fill_timing: str = "close",
    ) -> None:
        if fill_timing not in FILL_TIMINGS:
            raise ValueError(
                f"fill_timing must be one of {FILL_TIMINGS}, got {fill_timing!r}"
            )
        self.events = events
        self.data = data
        self.slippage = slippage or ZeroSlippage()
        self.commission = commission or ZeroCommission()
        self.fill_timing = fill_timing
        self._pending: list[OrderEvent] = []

    def execute_order(self, order: OrderEvent) -> None:
        """Fill ``order`` now, or hold it for the next bar, per ``fill_timing``."""
        if order.order_type != "MKT":
            raise NotImplementedError(
                "only market orders are simulated; limit orders need a queue-"
                "position model this project does not claim to have"
            )
        if self.fill_timing == "close":
            self.events.put(self._fill(order, self.data.current_bar(order.symbol)))
        else:
            self._pending.append(order)

    def fill_pending(self) -> list[FillEvent]:
        """Fill orders held from the previous bar, at this bar's open or close.

        The engine calls this when a new bar arrives, before the portfolio or
        the strategy sees it, and applies the returned fills directly.
        """
        pending, self._pending = self._pending, []
        fills = []
        for order in pending:
            bar = self.data.current_bar(order.symbol)
            if self.fill_timing == "next_open":
                # Price the fill off the open: the slippage and commission
                # models read the reference price from ``close``.
                bar = replace(bar, close=bar.open)
            fills.append(self._fill(order, bar, timestamp=bar.timestamp))
        return fills

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
