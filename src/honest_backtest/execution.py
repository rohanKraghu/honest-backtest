"""Order execution simulation.

Turns an :class:`~honest_backtest.events.OrderEvent` into a
:class:`~honest_backtest.events.FillEvent` by asking the slippage model for a
price and the commission model for a fee. Splitting the two means the report
can say which friction cost how much, instead of lumping them into one
unfalsifiable "costs" number.
"""

from __future__ import annotations

from queue import Queue

from .commission import CommissionModel, ZeroCommission
from .data import DataHandler
from .events import FillEvent, OrderEvent
from .slippage import SlippageModel, ZeroSlippage


class SimulatedExecutionHandler:
    """Fills every market order immediately at the current bar's close.

    This is optimistic in one respect that is worth naming: it assumes
    unlimited fill certainty, so an order never goes unfilled or partially
    filled. Modelling rejection would require a liquidity model this project
    does not have. The slippage model carries the size penalty instead.

    Args:
        events: Engine event queue.
        data: Data handler, used to read the current bar.
        slippage: Slippage model; defaults to :class:`ZeroSlippage`.
        commission: Commission model; defaults to :class:`ZeroCommission`.
    """

    def __init__(
        self,
        events: Queue,
        data: DataHandler,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
    ) -> None:
        self.events = events
        self.data = data
        self.slippage = slippage or ZeroSlippage()
        self.commission = commission or ZeroCommission()

    def execute_order(self, order: OrderEvent) -> None:
        """Simulate ``order`` and push the resulting fill onto the queue."""
        if order.order_type != "MKT":
            raise NotImplementedError(
                "only market orders are simulated; limit orders need a queue-"
                "position model this project does not claim to have"
            )
        bar = self.data.current_bar(order.symbol)
        fill_price = self.slippage.fill_price(order, bar)
        fee = self.commission.calculate(order, fill_price)
        self.events.put(
            FillEvent(
                symbol=order.symbol,
                timestamp=order.timestamp,
                quantity=order.quantity,
                direction=order.direction,
                fill_price=fill_price,
                commission=fee,
                reference_price=bar.close,
            )
        )
