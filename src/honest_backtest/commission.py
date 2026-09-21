"""Commission models: the explicit, invoiced cost of trading.

Kept separate from slippage so the report can attribute the damage to the
right friction. Every model returns a non-negative cash amount that is
subtracted from the account.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .events import OrderEvent


class CommissionModel(ABC):
    """Interface for commission models."""

    name: str = "commission"

    @abstractmethod
    def calculate(self, order: OrderEvent, fill_price: float) -> float:
        """Return the commission in account currency. Must be non-negative."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class ZeroCommission(CommissionModel):
    """No fees. Used to isolate the effect of commissions in the study."""

    name = "none"

    def calculate(self, order: OrderEvent, fill_price: float) -> float:
        """Return zero."""
        return 0.0


class PerShareCommission(CommissionModel):
    """Interactive-Brokers-style per-share fee with a floor and a cap.

    Args:
        per_share: Fee per share traded.
        minimum: Minimum charge per order.
        max_fraction_of_notional: Cap expressed as a fraction of trade value,
            mirroring the broker convention that the fee cannot exceed a
            given percentage of the trade.
    """

    name = "per_share"

    def __init__(
        self,
        per_share: float = 0.005,
        minimum: float = 1.0,
        max_fraction_of_notional: float = 0.01,
    ) -> None:
        if per_share < 0 or minimum < 0 or max_fraction_of_notional <= 0:
            raise ValueError("commission parameters must be non-negative")
        self.per_share = float(per_share)
        self.minimum = float(minimum)
        self.max_fraction_of_notional = float(max_fraction_of_notional)

    def calculate(self, order: OrderEvent, fill_price: float) -> float:
        """Return ``max(minimum, per_share * qty)`` capped at a share of notional."""
        if order.quantity <= 0:
            return 0.0
        notional = abs(order.quantity * fill_price)
        fee = max(self.minimum, self.per_share * order.quantity)
        return min(fee, self.max_fraction_of_notional * notional)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PerShareCommission(per_share={self.per_share}, minimum={self.minimum})"


class PercentOfNotionalCommission(CommissionModel):
    """Flat fee in basis points of traded notional.

    The right shape for FX, crypto and most retail equity platforms outside
    the US.

    Args:
        bps: Fee in basis points of the trade's notional value.
    """

    name = "percent_of_notional"

    def __init__(self, bps: float = 1.0) -> None:
        if bps < 0:
            raise ValueError("bps must be non-negative")
        self.bps = float(bps)

    def calculate(self, order: OrderEvent, fill_price: float) -> float:
        """Return ``bps / 10000`` of the trade's notional value."""
        return abs(order.quantity * fill_price) * self.bps / 10_000.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PercentOfNotionalCommission(bps={self.bps})"
