"""Performance statistics computed from an equity curve.

Deliberately few, and all of them standard. A long list of ratios is a common
way to distract from the fact that none of them are out of sample.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import e, sqrt
from statistics import NormalDist

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


# --------------------------------------------------------------------------
# How much of a Sharpe ratio is luck: Bailey and Lopez de Prado (2012, 2014).
# --------------------------------------------------------------------------

#: Euler-Mascheroni constant, used in the expected maximum of N normals.
_EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def _per_period_sharpe(returns: np.ndarray) -> float:
    """Unannualised Sharpe, using the same degenerate-series guard as above."""
    rets = np.asarray(returns, dtype=float)
    if rets.size < 2:
        return 0.0
    mean, sd = float(rets.mean()), float(rets.std(ddof=1))
    if not sd > 1e-12 * max(abs(mean), 1e-12):
        return 0.0
    return mean / sd


def probabilistic_sharpe(returns: np.ndarray, benchmark: float = 0.0) -> float:
    """Probability that the true Sharpe exceeds ``benchmark``.

    The Probabilistic Sharpe Ratio (Bailey and Lopez de Prado, 2012). A
    Sharpe estimated from ``T`` returns has a standard error that grows
    with negative skew and fat tails, so the same point estimate is less
    convincing for a strategy that sells crash insurance:

        PSR = Phi((SR - SR*) * sqrt(T - 1) / sqrt(1 - g3 * SR + (g4 - 1) / 4 * SR^2))

    Args:
        returns: Per-bar returns.
        benchmark: The Sharpe to beat, **per bar** (not annualised).

    Returns:
        A probability in ``[0, 1]``; 0.5 when there is too little data.
    """
    rets = np.asarray(returns, dtype=float)
    n = rets.size
    if n < 3:
        return 0.5
    sr = _per_period_sharpe(rets)
    centred = rets - rets.mean()
    sd = float(rets.std(ddof=0))
    if sd == 0.0:
        return 0.5
    skew = float((centred**3).mean() / sd**3)
    kurt = float((centred**4).mean() / sd**4)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0.0:
        return 0.5
    return float(_NORMAL.cdf((sr - benchmark) * sqrt(n - 1) / sqrt(denom)))


def expected_max_sharpe(n_trials: int, trial_variance: float) -> float:
    """Expected largest of ``n_trials`` Sharpe ratios whose true value is zero.

    If you try ``N`` settings of a strategy with no edge, the best of them
    still shows a positive Sharpe by luck alone. Its expected size is

        sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))

    where ``V`` is the variance of the Sharpe ratios across the trials and
    ``g`` the Euler-Mascheroni constant. Per bar, like ``trial_variance``.
    """
    if n_trials < 2 or trial_variance <= 0.0:
        return 0.0
    return sqrt(trial_variance) * (
        (1.0 - _EULER_GAMMA) * _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
        + _EULER_GAMMA * _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * e))
    )


def deflated_sharpe(
    returns: np.ndarray, trial_sharpes: Sequence[float], bars_per_year: int = 252
) -> float:
    """Probability that a selected strategy's Sharpe is not just the luckiest trial.

    The Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014) is the
    :func:`probabilistic_sharpe` of the chosen strategy measured against the
    Sharpe that the best of all the trials would show with no edge at all.
    It is the right test for an in-sample fit: picking the best of a grid
    is exactly the selection it corrects for.

    Args:
        returns: Per-bar returns of the selected strategy.
        trial_sharpes: **Annualised** Sharpe of every setting tried, the
            selected one included.
        bars_per_year: Annualisation factor used for ``trial_sharpes``.

    Returns:
        A probability in ``[0, 1]``. With a single trial it equals the PSR
        against zero.
    """
    per_bar = np.asarray(trial_sharpes, dtype=float) / sqrt(bars_per_year)
    variance = float(per_bar.var(ddof=1)) if per_bar.size > 1 else 0.0
    return probabilistic_sharpe(returns, expected_max_sharpe(per_bar.size, variance))
