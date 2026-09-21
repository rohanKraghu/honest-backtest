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
    same direction repeatedly.

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
