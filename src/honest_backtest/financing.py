"""Carrying a position costs money even when nothing trades.

A backtest that only charges for trading treats a leveraged or short book as
free to hold. It is not: borrowed cash pays interest, borrowed shares pay a
lending fee, and a broker caps how much exposure an account may run. A
short-heavy or leveraged strategy that looks cheap without these can look
very different with them, which is the same lesson as the rest of the
ladder: every assumption you leave out flatters the result.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Financing:
    """Annual rates and limits for holding a position between bars.

    Accrued once per bar on the book held over that bar, at simple daily
    rates (``rate / bars_per_year``).

    Attributes:
        cash_rate: Interest earned on positive cash, per year. Short-sale
            proceeds sit in cash and earn it too; ``short_fee`` is the
            lender's cut that offsets it.
        borrow_rate: Interest paid on negative cash (a margin loan), per
            year. Usually above ``cash_rate``.
        short_fee: Stock-lending fee on the market value of a short
            position, per year.
        max_leverage: Largest gross exposure as a multiple of equity. Target
            weights beyond it are clipped. ``None`` means no limit.
        bars_per_year: Bars in a year, to turn annual rates into per-bar ones.
    """

    cash_rate: float = 0.0
    borrow_rate: float = 0.0
    short_fee: float = 0.0
    max_leverage: float | None = None
    bars_per_year: int = 252

    def __post_init__(self) -> None:
        """Reject impossible settings."""
        if self.borrow_rate < 0 or self.short_fee < 0:
            raise ValueError("borrow_rate and short_fee must be non-negative")
        if self.max_leverage is not None and not self.max_leverage > 0:
            raise ValueError("max_leverage must be positive")
        if self.bars_per_year <= 0:
            raise ValueError("bars_per_year must be positive")

    def accrual(self, cash: float, position: float, price: float) -> float:
        """Cash credited (positive) or charged (negative) for holding one bar.

        Args:
            cash: Cash held over the bar.
            position: Units held over the bar (negative when short).
            price: Mark price at the start of the bar.
        """
        rate = self.cash_rate if cash >= 0 else self.borrow_rate
        interest = cash * rate
        lending = -max(0.0, -position) * price * self.short_fee
        return (interest + lending) / self.bars_per_year

    def clip_weight(self, weight: float) -> float:
        """Clip a target weight to the leverage limit."""
        if self.max_leverage is None:
            return weight
        return max(-self.max_leverage, min(self.max_leverage, weight))
