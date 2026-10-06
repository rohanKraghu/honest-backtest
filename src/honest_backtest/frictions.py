"""The frictions beyond spread and fees, bundled for the ladder.

Each one removes an assumption most backtests make without saying so:
unlimited liquidity (``max_participation``), always crossing the spread
(``limit_offset_bps``), and holding a position for free (``financing``).
The ladder applies them together as one rung, after next-bar execution, so
the audit shows what they cost on top of everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .financing import Financing


@dataclass(frozen=True)
class MarketFrictions:
    """Liquidity, order style and carry settings for a backtest.

    Attributes:
        max_participation: Largest fraction of a bar's volume one fill may
            take; ``None`` means unlimited liquidity.
        limit_offset_bps: Rebalance with passive limit orders this far inside
            the close; ``None`` means market orders.
        limit_expiry_bars: Bars a limit order rests before cancellation.
        financing: Interest, borrow fees and leverage limit.
    """

    max_participation: float | None = None
    limit_offset_bps: float | None = None
    limit_expiry_bars: int = 1
    financing: Financing | None = None

    @property
    def active(self) -> bool:
        """Whether any friction is switched on."""
        return (
            self.max_participation is not None
            or self.limit_offset_bps is not None
            or self.financing is not None
        )

    def describe(self) -> str:
        """One line naming what is switched on, for report headers."""
        parts = []
        if self.max_participation is not None:
            parts.append(f"fills capped at {self.max_participation:.2%} of volume")
        if self.limit_offset_bps is not None:
            parts.append(
                f"limit orders {self.limit_offset_bps:g}bp inside the close, "
                f"{self.limit_expiry_bars} bar(s)"
            )
        f = self.financing
        if f is not None:
            parts.append(
                f"cash {f.cash_rate:.2%}, margin {f.borrow_rate:.2%}, "
                f"short fee {f.short_fee:.2%}"
                + (f", leverage <= {f.max_leverage:g}x" if f.max_leverage else "")
            )
        return "; ".join(parts) or "none"

    def run_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :func:`~honest_backtest.engine.run_backtest`."""
        return {
            "max_participation": self.max_participation,
            "limit_offset_bps": self.limit_offset_bps,
            "limit_expiry_bars": self.limit_expiry_bars,
            "financing": self.financing,
        }
