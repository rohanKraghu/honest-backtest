"""Strategies. One honest, one deliberately broken.

Both implement the same idea -- time-series momentum, sized by a
volatility-normalised z-score -- so that the difference between the Sharpe
ratios they report is attributable to the look-ahead bias and nothing else.

The honest strategy accumulates its own state bar by bar, exactly as a live
strategy would. It cannot see the future because the data handler will not
show it the future.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from queue import Queue

from .data import DataHandler, LookAheadDataHandler
from .events import MarketEvent, SignalEvent


def _stdev(values: list[float] | deque[float]) -> float:
    """Sample standard deviation; returns 0.0 for fewer than two points."""
    vals = list(values)
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return math.sqrt(var)


def _clip(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


class Strategy(ABC):
    """Interface for signal-generating components.

    A strategy reacts to a :class:`~honest_backtest.events.MarketEvent` by
    pushing zero or more :class:`~honest_backtest.events.SignalEvent` onto the
    queue. It has no access to the portfolio, to cash, or to any bar the data
    handler has not delivered.
    """

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> None:
        """React to a new bar by emitting signals onto the event queue."""

    @property
    def uses_look_ahead(self) -> bool:
        """Whether this strategy knowingly reads future data.

        Reported in the results table so a stage that cheats is labelled as
        cheating rather than quietly compared against honest ones.
        """
        return False


class TimeSeriesMomentumStrategy(Strategy):
    """Point-in-time time-series momentum.

    At each bar ``t`` the strategy computes the ``lookback``-bar log return
    ending at ``t``, divides it by a *trailing* estimate of that quantity's
    standard deviation, and targets a portfolio weight proportional to the
    resulting z-score.

    Every statistic is trailing. The volatility normaliser uses only momentum
    observations the strategy has already lived through, so there is no
    full-sample standardisation leak.

    Args:
        events: Engine event queue.
        data: Point-in-time data handler.
        symbol: Instrument to trade.
        lookback: Momentum horizon in bars. This is the parameter the
            walk-forward study re-fits on each training window.
        vol_window: Trailing window for the volatility normaliser.
        min_vol_observations: Minimum momentum observations before the
            strategy is willing to take any position at all.
        z_scale: z-score corresponding to a full-size position.
        max_weight: Maximum absolute portfolio weight.
    """

    def __init__(
        self,
        events: Queue,
        data: DataHandler,
        symbol: str = "SYNTH",
        lookback: int = 20,
        vol_window: int = 90,
        min_vol_observations: int = 30,
        z_scale: float = 1.5,
        max_weight: float = 1.0,
    ) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if z_scale <= 0:
            raise ValueError("z_scale must be positive")
        self.events = events
        self.data = data
        self.symbol = symbol
        self.lookback = int(lookback)
        self.vol_window = int(vol_window)
        self.min_vol_observations = int(min_vol_observations)
        self.z_scale = float(z_scale)
        self.max_weight = float(max_weight)
        self._momentum_history: deque[float] = deque(maxlen=self.vol_window)

    def _momentum(self) -> float | None:
        """Trailing ``lookback``-bar log return, or ``None`` if not enough history."""
        closes = self.data.latest_closes(self.symbol, self.lookback + 1)
        if len(closes) < self.lookback + 1:
            return None
        if closes[0] <= 0 or closes[-1] <= 0:
            return None
        return math.log(closes[-1] / closes[0])

    def calculate_signals(self, event: MarketEvent) -> None:
        """Emit a target-weight signal for the current bar."""
        momentum = self._momentum()
        if momentum is None:
            return

        # Normalise with trailing observations only, and record the current
        # observation AFTER computing the scale so the current value does not
        # normalise itself.
        scale = _stdev(self._momentum_history)
        have_enough = len(self._momentum_history) >= self.min_vol_observations
        self._momentum_history.append(momentum)

        if not have_enough or scale <= 0.0:
            return

        z = momentum / scale
        weight = _clip(z / self.z_scale, -self.max_weight, self.max_weight)
        self.events.put(
            SignalEvent(
                symbol=self.symbol,
                timestamp=event.timestamp,
                target_weight=weight,
                strength=z,
            )
        )


class LookAheadMomentumStrategy(TimeSeriesMomentumStrategy):
    """The same strategy, written the way a naive backtest writes it.

    Two bugs, both extremely common, both reproduced here on purpose:

    1.  **Off-by-one signal alignment.** The momentum window is extended to
        include the *next* bar's close, then the position is taken at the
        current bar. In pandas this is a signal that was never shifted, and
        it is the single most common reason a student backtest shows a Sharpe
        above 3. Because the peeked bar's return is the very return the
        position earns, the strategy is partly reading its own answer.
    2.  **Full-sample standardisation.** The volatility normaliser is computed
        once from the entire price history, including bars that have not
        happened. Subtler, and it survives peer review more often than it
        should.

    Requires a :class:`~honest_backtest.data.LookAheadDataHandler`; a
    point-in-time handler will raise
    :class:`~honest_backtest.data.LookAheadError` instead of cooperating.
    """

    def __init__(self, *args, peek_bars: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.data, LookAheadDataHandler):
            raise TypeError(
                "LookAheadMomentumStrategy requires a LookAheadDataHandler; "
                "point-in-time handlers correctly refuse to serve future bars"
            )
        self.peek_bars = int(peek_bars)
        self._full_sample_scale: float | None = None

    @property
    def uses_look_ahead(self) -> bool:
        """Always ``True``. This strategy cheats and says so."""
        return True

    def _full_sample_momentum_scale(self) -> float:
        """Stdev of the momentum series over the ENTIRE sample. Leak #2."""
        if self._full_sample_scale is None:
            closes = self.data.all_closes(self.symbol)
            k = self.lookback
            moms = [
                math.log(closes[i] / closes[i - k])
                for i in range(k, len(closes))
                if closes[i] > 0 and closes[i - k] > 0
            ]
            self._full_sample_scale = _stdev(moms)
        return self._full_sample_scale

    def _momentum(self) -> float | None:
        """Momentum computed through the NEXT bar's close. Leak #1."""
        future = self.data.peek_ahead(self.symbol, self.peek_bars)
        if not future:
            return super()._momentum()
        # Window ends on the peeked bar, so it spans the return we are about
        # to earn. This is the off-by-one.
        closes = self.data.latest_closes(self.symbol, self.lookback)
        if len(closes) < self.lookback:
            return None
        start, end = closes[0], future[-1].close
        if start <= 0 or end <= 0:
            return None
        return math.log(end / start)

    def calculate_signals(self, event: MarketEvent) -> None:
        """Emit a target weight using peeked data and full-sample scaling."""
        momentum = self._momentum()
        if momentum is None:
            return
        scale = self._full_sample_momentum_scale()
        if scale <= 0.0:
            return
        z = momentum / scale
        weight = _clip(z / self.z_scale, -self.max_weight, self.max_weight)
        self.events.put(
            SignalEvent(
                symbol=self.symbol,
                timestamp=event.timestamp,
                target_weight=weight,
                strength=z,
            )
        )


class BuyAndHoldStrategy(Strategy):
    """Always fully long. The benchmark every strategy should have to beat.

    Args:
        events: Engine event queue.
        data: Data handler (unused beyond the symbol).
        symbol: Instrument to hold.
        weight: Target weight to hold.
    """

    def __init__(
        self,
        events: Queue,
        data: DataHandler,
        symbol: str = "SYNTH",
        weight: float = 1.0,
    ) -> None:
        self.events = events
        self.data = data
        self.symbol = symbol
        self.weight = float(weight)
        self._entered = False

    def calculate_signals(self, event: MarketEvent) -> None:
        """Emit one full-size long signal on the first bar, then nothing."""
        if self._entered:
            return
        self._entered = True
        self.events.put(
            SignalEvent(
                symbol=self.symbol, timestamp=event.timestamp, target_weight=self.weight
            )
        )
