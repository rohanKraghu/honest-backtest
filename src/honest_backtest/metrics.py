"""Performance statistics computed from an equity curve.

Deliberately few, and all of them standard. A long list of ratios is a common
way to distract from the fact that none of them are out of sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class PerformanceMetrics:
    """Summary statistics for one backtest run.

    Attributes:
        sharpe: Annualised Sharpe ratio, zero risk-free rate.
        total_return: Cumulative return over the evaluated window.
        annual_return: Geometric annualised return.
        annual_volatility: Annualised standard deviation of returns.
        max_drawdown: Worst peak-to-trough decline, as a negative fraction.
        n_periods: Number of return observations behind the statistics.
        n_trades: Fills executed.
        commission_cost: Total commission paid, in account currency.
        slippage_cost: Total implicit cost paid, in account currency.
        annual_turnover: Traded notional per year divided by average equity.
    """

    sharpe: float
    total_return: float
    annual_return: float
    annual_volatility: float
    max_drawdown: float
    n_periods: int
    n_trades: int = 0
    commission_cost: float = 0.0
    slippage_cost: float = 0.0
    annual_turnover: float = 0.0


def simple_returns(equity: np.ndarray | list[float]) -> np.ndarray:
    """Period-over-period simple returns of an equity curve.

    Args:
        equity: Equity values in time order.

    Returns:
        Array of length ``len(equity) - 1``. Empty if fewer than two points.
    """
    arr = np.asarray(equity, dtype=float)
    if arr.size < 2:
        return np.zeros(0)
    prev = arr[:-1]
    # Guard against a wiped-out account producing division by zero.
    safe_prev = np.where(prev == 0.0, np.nan, prev)
    out = arr[1:] / safe_prev - 1.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def annualised_sharpe(returns: np.ndarray, bars_per_year: int = 252) -> float:
    """Annualised Sharpe ratio with a zero risk-free rate.

    Args:
        returns: Per-bar simple returns.
        bars_per_year: Annualisation factor.

    Returns:
        ``mean / stdev * sqrt(bars_per_year)``, or 0.0 if undefined.

    Note:
        The square-root-of-time scaling assumes serially independent returns.
        The synthetic series here has deliberately autocorrelated returns, so
        this (like every Sharpe ratio ever quoted) is a mild approximation.
    """
    if returns.size < 2:
        return 0.0
    mean = float(returns.mean())
    sd = float(returns.std(ddof=1))
    # An exact `sd == 0.0` check is not enough: floating-point rounding leaves
    # a constant series with a standard deviation around 1e-19, which turns a
    # Sharpe ratio into something like 1e16. A degenerate curve carries no
    # information, and a backtester that reports an astronomical Sharpe for
    # one is worse than useless.
    if not sd > 1e-12 * max(abs(mean), 1e-12):
        return 0.0
    return float(mean / sd * sqrt(bars_per_year))


def max_drawdown(equity: np.ndarray | list[float]) -> float:
    """Worst peak-to-trough decline of an equity curve.

    Args:
        equity: Equity values in time order.

    Returns:
        A non-positive fraction, e.g. ``-0.23`` for a 23% drawdown.
    """
    arr = np.asarray(equity, dtype=float)
    if arr.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(arr)
    safe_max = np.where(running_max == 0.0, np.nan, running_max)
    dd = arr / safe_max - 1.0
    return float(np.nan_to_num(dd, nan=0.0).min())


def total_return(equity: np.ndarray | list[float]) -> float:
    """Cumulative return from first to last point of an equity curve."""
    arr = np.asarray(equity, dtype=float)
    if arr.size < 2 or arr[0] == 0:
        return 0.0
    return float(arr[-1] / arr[0] - 1.0)


def annualised_return(
    equity: np.ndarray | list[float], bars_per_year: int = 252
) -> float:
    """Geometric annualised return of an equity curve.

    Returns 0.0 if the account went to zero or below, where a geometric rate
    is not defined.
    """
    arr = np.asarray(equity, dtype=float)
    if arr.size < 2 or arr[0] <= 0 or arr[-1] <= 0:
        return 0.0
    years = (arr.size - 1) / bars_per_year
    if years <= 0:
        return 0.0
    return float((arr[-1] / arr[0]) ** (1.0 / years) - 1.0)


def compute_metrics(
    equity: np.ndarray | list[float],
    bars_per_year: int = 252,
    n_trades: int = 0,
    commission_cost: float = 0.0,
    slippage_cost: float = 0.0,
    traded_notional: float = 0.0,
    returns: np.ndarray | None = None,
) -> PerformanceMetrics:
    """Bundle the statistics above into a :class:`PerformanceMetrics`.

    Args:
        equity: Equity curve for the evaluated window.
        bars_per_year: Annualisation factor.
        n_trades: Number of fills.
        commission_cost: Total commission paid.
        slippage_cost: Total implicit cost paid.
        traded_notional: Gross notional traded, for the turnover figure.
        returns: Optional pre-computed return series. The walk-forward study
            passes a *stitched* series here, because concatenated out-of-sample
            windows do not form a single continuous equity curve and taking
            returns across a fold boundary would invent a jump.

    Returns:
        The populated metrics object.
    """
    arr = np.asarray(equity, dtype=float)
    rets = simple_returns(arr) if returns is None else np.asarray(returns, dtype=float)
    years = max(rets.size / bars_per_year, 1e-9)
    avg_equity = float(arr.mean()) if arr.size else 0.0
    turnover = (
        traded_notional / avg_equity / years if avg_equity > 0 and years > 0 else 0.0
    )

    if returns is None:
        tot = total_return(arr)
        ann = annualised_return(arr, bars_per_year)
        mdd = max_drawdown(arr)
    else:
        # Rebuild a notional curve from the stitched returns so that total
        # return and drawdown describe the out-of-sample experience only.
        curve = np.concatenate([[1.0], np.cumprod(1.0 + rets)])
        tot = total_return(curve)
        ann = annualised_return(curve, bars_per_year)
        mdd = max_drawdown(curve)

    return PerformanceMetrics(
        sharpe=annualised_sharpe(rets, bars_per_year),
        total_return=tot,
        annual_return=ann,
        annual_volatility=float(rets.std(ddof=1) * sqrt(bars_per_year))
        if rets.size > 1
        else 0.0,
        max_drawdown=mdd,
        n_periods=int(rets.size),
        n_trades=n_trades,
        commission_cost=commission_cost,
        slippage_cost=slippage_cost,
        annual_turnover=turnover,
    )
