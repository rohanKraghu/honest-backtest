"""Slippage models: the implicit cost of crossing the spread and moving price.

Slippage is modelled as a function that maps an order plus the prevailing bar
to an execution price. Every model here is *adverse by construction*: a buy
fills at or above the reference price, a sell at or below. That property is
asserted in the test suite, because a slippage model that can accidentally
improve your fill is a slippage model that flatters the backtest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import sqrt

from .data import Bar
from .events import OrderEvent


class SlippageModel(ABC):
    """Interface for slippage models."""

    name: str = "slippage"

    @abstractmethod
    def fill_price(self, order: OrderEvent, bar: Bar) -> float:
        """Return the execution price for ``order`` given ``bar``.

        Implementations must return a price that is never better than
        ``bar.close`` for the order's direction.
        """

    def reset(self) -> None:  # noqa: B027 - optional hook, a no-op by default
        """Forget any state from a previous run.

        Called by :func:`~honest_backtest.engine.run_backtest` before every
        run, because one model instance is shared by every run of a ladder.
        Stateless models need do nothing.
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class ZeroSlippage(SlippageModel):
    """Fills at the closing price. Free, instant, infinitely liquid.

    Not a realistic model. Included so the degradation study can isolate the
    contribution of slippage by toggling exactly one component.
    """

    name = "none"

    def fill_price(self, order: OrderEvent, bar: Bar) -> float:
        """Return the bar's closing price, unmodified."""
        return bar.close


class FixedBpsSlippage(SlippageModel):
    """Constant adverse move of ``half_spread_bps`` basis points.

    The simplest defensible model: you always pay half the bid-ask spread.
    It ignores order size, which means it understates the cost of trading a
    large position and overstates it for a tiny one.

    Args:
        half_spread_bps: Half-spread in basis points of the reference price.
    """

    name = "fixed_bps"

    def __init__(self, half_spread_bps: float = 2.0) -> None:
        if half_spread_bps < 0:
            raise ValueError("half_spread_bps must be non-negative")
        self.half_spread_bps = float(half_spread_bps)

    def fill_price(self, order: OrderEvent, bar: Bar) -> float:
        """Return the close moved adversely by the configured half-spread."""
        sign = 1.0 if order.direction == "BUY" else -1.0
        return bar.close * (1.0 + sign * self.half_spread_bps / 10_000.0)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FixedBpsSlippage(half_spread_bps={self.half_spread_bps})"


class SpreadPlusImpactSlippage(SlippageModel):
    """Half-spread plus a square-root market impact term.

    Impact is modelled as ``impact_coefficient * volatility * sqrt(participation)``
    where participation is the order's size as a fraction of the bar's volume.
    The square-root law is the standard empirical shape for temporary impact;
    the coefficient is the part nobody agrees on, which is why it is a
    constructor argument rather than a hard-coded number.

    This is the model used in the headline result. It is a simplification:
    it is a single-bar temporary impact model with no permanent component and
    no decay, so it will understate the cost of a strategy that trades the
    same direction repeatedly. :class:`PermanentImpactSlippage` adds that
    component.

    Args:
        half_spread_bps: Half-spread in basis points.
        impact_coefficient: Scales the square-root impact term.
        bar_volatility: Per-bar volatility used to scale impact. Defaults to
            a 1.26% daily move, i.e. 20% annualised.
    """

    name = "spread_plus_impact"

    def __init__(
        self,
        half_spread_bps: float = 2.0,
        impact_coefficient: float = 0.6,
        bar_volatility: float = 0.0126,
    ) -> None:
        if half_spread_bps < 0 or impact_coefficient < 0 or bar_volatility < 0:
            raise ValueError("slippage parameters must be non-negative")
        self.half_spread_bps = float(half_spread_bps)
        self.impact_coefficient = float(impact_coefficient)
        self.bar_volatility = float(bar_volatility)

    def fill_price(self, order: OrderEvent, bar: Bar) -> float:
        """Return the close moved adversely by half-spread plus impact."""
        sign = 1.0 if order.direction == "BUY" else -1.0
        spread_frac = self.half_spread_bps / 10_000.0
        volume = max(bar.volume, 1.0)
        participation = min(order.quantity / volume, 1.0)
        impact_frac = self.impact_coefficient * self.bar_volatility * sqrt(participation)
        return bar.close * (1.0 + sign * (spread_frac + impact_frac))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SpreadPlusImpactSlippage(half_spread_bps={self.half_spread_bps}, "
            f"impact_coefficient={self.impact_coefficient})"
        )


class PermanentImpactSlippage(SpreadPlusImpactSlippage):
    """Spread and temporary impact, plus permanent impact that decays.

    Each fill leaves a lasting push on the price in its own direction,
    ``permanent_coefficient * volatility * participation`` (linear in size,
    as in Almgren and Chriss), which then decays with a half-life of
    ``half_life_bars``. A later fill in the same direction pays the push
    still outstanding on top of its own spread and temporary impact, so a
    strategy that keeps buying is charged for walking the price up against
    itself.

    One deliberate asymmetry keeps the model adverse by construction: a
    fill against the outstanding push (selling after buying) is not paid
    back for it. The marks come from the data, which never saw your trades,
    so crediting the reversal would book a gain the equity curve cannot
    show. This overstates cost slightly for strategies that reverse quickly,
    which is the safe direction to be wrong in.

    Args:
        half_spread_bps: Half-spread in basis points.
        impact_coefficient: Scales the square-root temporary impact.
        bar_volatility: Per-bar volatility used to scale both impacts.
        permanent_coefficient: Scales the linear permanent impact.
        half_life_bars: Bars for the outstanding push to halve. ``inf``
            means it never decays.
    """

    name = "spread_impact_permanent"

    def __init__(
        self,
        half_spread_bps: float = 2.0,
        impact_coefficient: float = 0.6,
        bar_volatility: float = 0.0126,
        permanent_coefficient: float = 0.3,
        half_life_bars: float = 5.0,
    ) -> None:
        super().__init__(half_spread_bps, impact_coefficient, bar_volatility)
        if permanent_coefficient < 0 or not half_life_bars > 0:
            raise ValueError(
                "permanent_coefficient must be non-negative and half_life_bars positive"
            )
        self.permanent_coefficient = float(permanent_coefficient)
        self.half_life_bars = float(half_life_bars)
        self.reset()

    def reset(self) -> None:
        """Clear the outstanding push between runs."""
        # Per instrument: (signed push as a fraction of price, bar it was set).
        self._push: dict[str, tuple[float, int]] = {}

    def outstanding_push(self, timestamp: int, symbol: str = "SYNTH") -> float:
        """The signed push on ``symbol`` still outstanding at bar ``timestamp``."""
        push, since = self._push.get(symbol, (0.0, timestamp))
        elapsed = max(0, timestamp - since)
        return push * 0.5 ** (elapsed / self.half_life_bars)

    def fill_price(self, order: OrderEvent, bar: Bar) -> float:
        """Return the temporary-impact price plus any adverse outstanding push."""
        sign = 1.0 if order.direction == "BUY" else -1.0
        push = self.outstanding_push(bar.timestamp, order.symbol)
        adverse = max(0.0, sign * push)
        price = super().fill_price(order, bar) + bar.close * sign * adverse

        volume = max(bar.volume, 1.0)
        participation = min(order.quantity / volume, 1.0)
        self._push[order.symbol] = (
            push
            + sign * self.permanent_coefficient * self.bar_volatility * participation,
            bar.timestamp,
        )
        return price

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PermanentImpactSlippage(half_spread_bps={self.half_spread_bps}, "
            f"impact_coefficient={self.impact_coefficient}, "
            f"permanent_coefficient={self.permanent_coefficient}, "
            f"half_life_bars={self.half_life_bars})"
        )
