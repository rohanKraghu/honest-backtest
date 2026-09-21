"""Importable helpers for the test suite.

Kept out of ``conftest.py`` so they can be imported by name; pytest puts the
``tests`` directory on ``sys.path``, so ``from helpers import ...`` resolves.
"""

from __future__ import annotations

from queue import Queue

from honest_backtest.data import Bar, HistoricBarDataHandler
from honest_backtest.events import MarketEvent
from honest_backtest.strategy import Strategy


class RecordingStrategy(Strategy):
    """Records every bar it is shown, tagged with the bar it was shown on.

    Used to prove, empirically and over a whole backtest, that the data
    handler never hands a component a bar from the future.
    """

    def __init__(self, events: Queue, data, symbol: str = "SYNTH", window: int = 50):
        self.events = events
        self.data = data
        self.symbol = symbol
        self.window = window
        #: ``(current_timestamp, observed_timestamp)`` for every bar observed.
        self.observations: list[tuple[int, int]] = []

    def calculate_signals(self, event: MarketEvent) -> None:
        """Ask the handler for a generous window and record what came back."""
        for bar in self.data.get_latest_bars(self.symbol, self.window):
            self.observations.append((event.timestamp, bar.timestamp))
        current = self.data.current_bar(self.symbol)
        self.observations.append((event.timestamp, current.timestamp))


class FixedWeightStrategy(Strategy):
    """Emits a scripted sequence of target weights, one per bar.

    Lets accounting tests drive an exact, known trade schedule instead of
    depending on whatever a real strategy happens to do.
    """

    def __init__(
        self, events: Queue, data, weights: list[float], symbol: str = "SYNTH"
    ) -> None:
        from honest_backtest.events import SignalEvent

        self._signal_cls = SignalEvent
        self.events = events
        self.data = data
        self.symbol = symbol
        self.weights = list(weights)
        self._i = 0

    def calculate_signals(self, event: MarketEvent) -> None:
        """Emit the next scripted weight, if there is one left."""
        if self._i >= len(self.weights):
            return
        weight = self.weights[self._i]
        self._i += 1
        self.events.put(
            self._signal_cls(
                symbol=self.symbol, timestamp=event.timestamp, target_weight=weight
            )
        )


def make_handler(bars: list[Bar]) -> tuple[Queue, HistoricBarDataHandler]:
    """Build a queue and a point-in-time handler over ``bars``."""
    q: Queue = Queue()
    return q, HistoricBarDataHandler(q, bars)
