r"""A moving-average crossover, written to be audited.

Run it through the ladder with::

    honest-backtest audit --data examples/sample_prices.csv \\
        --strategy examples/sma_crossover.py

The file only has to define ``SPEC``. Everything the strategy reads comes
from the point-in-time data handler, so it cannot see the future even if it
tries: asking for an unreached bar raises ``LookAheadError``.
"""

from __future__ import annotations

from honest_backtest import SignalEvent, Strategy, StrategySpec, param_grid


class MovingAverageCrossover(Strategy):
    """Long when the fast average is above the slow one, flat (or short) otherwise.

    Args:
        events: Engine event queue.
        data: Point-in-time data handler.
        symbol: Instrument to trade.
        fast: Fast moving-average window, bars.
        slow: Slow moving-average window, bars.
        allow_short: Go short instead of flat when the fast average is below.
    """

    def __init__(self, events, data, symbol, fast=20, slow=100, allow_short=False):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.events = events
        self.data = data
        self.symbol = symbol
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short

    def calculate_signals(self, event):
        """Emit a target weight once there is a full slow window of history."""
        closes = self.data.latest_closes(self.symbol, self.slow)
        if len(closes) < self.slow:
            return
        fast_ma = sum(closes[-self.fast :]) / self.fast
        slow_ma = sum(closes) / self.slow
        if fast_ma > slow_ma:
            weight = 1.0
        else:
            weight = -1.0 if self.allow_short else 0.0
        self.events.put(SignalEvent(self.symbol, event.timestamp, weight))


SPEC = StrategySpec(
    name="moving-average crossover",
    build=MovingAverageCrossover,
    grid=param_grid(fast=(10, 20, 50), slow=(100, 200)),
    warmup=200,
)
