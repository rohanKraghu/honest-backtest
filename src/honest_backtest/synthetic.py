"""Synthetic price generation: GBM plus a deliberately injected weak signal.

Why synthetic data
------------------
This project is a demonstration harness for *methodology*, not a claim about
any market. Synthetic data is the right choice here because it makes the
ground truth knowable: we inject an edge of known size, so we can compare the
Sharpe ratio a backtest reports against the Sharpe ratio that is actually
there. With real data you can never separate "my method is biased" from "the
market really did that". Everything in :mod:`honest_backtest.data` is written
against the :class:`~honest_backtest.data.DataHandler` interface, so a CSV or
API source drops in without touching the engine.

The model
---------
A latent AR(1) state ``s_t`` drives a small, persistent tilt in the drift::

    s_t = phi * s_{t-1} + sqrt(1 - phi^2) * eta_t          eta ~ N(0, 1)
    r_t = mu_dt + alpha * sigma_dt * s_{t-1} + sigma_dt * z_t
    P_t = P_{t-1} * exp(r_t)

``s_t`` is standardised to unit variance, so ``alpha`` is directly the
per-bar information ratio available to an observer who knows ``s`` exactly.
Annualised, that observer's ceiling is ``alpha * sqrt(bars_per_year)``.

The state is *latent*: strategies never see it. They must infer it from past
returns, where it is buried under ``sigma_dt * z_t`` noise roughly
``1 / alpha`` times larger. That is what makes the injected edge weak and
therefore interesting -- a strong edge would survive any amount of friction
and would demonstrate nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np

from .data import Bar, bars_from_series


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of the synthetic price process.

    Attributes:
        n_bars: Number of bars to generate.
        initial_price: Starting price level.
        annual_drift: Annualised drift of the log-price process.
        annual_vol: Annualised volatility of log returns.
        bars_per_year: Bars per year, used for annualisation throughout.
        signal_alpha: Per-bar information ratio of the injected edge. This is
            the single knob that controls how exploitable the series is.
        signal_persistence: AR(1) coefficient of the latent state. Higher
            means the tilt lasts longer and a longer lookback can find it.
        average_daily_volume: Mean traded volume, consumed by the market
            impact slippage model.
        volume_log_vol: Log-normal dispersion of volume.
    """

    n_bars: int = 2520
    initial_price: float = 100.0
    annual_drift: float = 0.04
    annual_vol: float = 0.20
    bars_per_year: int = 252
    signal_alpha: float = 0.115
    signal_persistence: float = 0.94
    average_daily_volume: float = 1_000_000.0
    volume_log_vol: float = 0.35

    def __post_init__(self) -> None:
        if not 0.0 <= self.signal_persistence < 1.0:
            raise ValueError("signal_persistence must be in [0, 1)")
        if self.n_bars < 2:
            raise ValueError("need at least 2 bars")
        if self.annual_vol <= 0:
            raise ValueError("annual_vol must be positive")


@dataclass(frozen=True)
class SyntheticSeries:
    """A generated price path together with its (hidden) ground truth.

    Attributes:
        closes: Closing prices.
        volumes: Traded volumes.
        log_returns: Realised log returns; ``log_returns[0]`` is 0.
        latent_state: The AR(1) state that generated the edge. GROUND TRUTH --
            never passed to a strategy. It exists so that tests and the
            report can quantify the ceiling the strategy is shooting at.
        config: The configuration used.
        seed: The seed used.
    """

    closes: np.ndarray
    volumes: np.ndarray
    log_returns: np.ndarray
    latent_state: np.ndarray
    config: SyntheticConfig
    seed: int
    symbol: str = field(default="SYNTH")

    def __len__(self) -> int:
        return int(self.closes.shape[0])

    def to_bars(self) -> list[Bar]:
        """Convert to the bar list consumed by the data handlers."""
        return bars_from_series(
            self.closes.tolist(), symbol=self.symbol, volumes=self.volumes.tolist()
        )


def generate_price_series(
    config: SyntheticConfig | None = None, seed: int = 20260921
) -> SyntheticSeries:
    """Generate one reproducible synthetic price path.

    Args:
        config: Process parameters; defaults to :class:`SyntheticConfig`.
        seed: Seed for ``numpy.random.default_rng``. The PCG64 bit generator
            gives a stream that is stable across numpy versions, so a pinned
            seed reproduces the exact same path on any reviewer's machine.

    Returns:
        A :class:`SyntheticSeries`.
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(seed)
    n = cfg.n_bars

    dt = 1.0 / cfg.bars_per_year
    sigma_dt = cfg.annual_vol * sqrt(dt)
    mu_dt = (cfg.annual_drift - 0.5 * cfg.annual_vol**2) * dt

    # Latent AR(1) state, standardised to unit unconditional variance.
    phi = cfg.signal_persistence
    eta = rng.standard_normal(n)
    state = np.empty(n)
    state[0] = eta[0]
    innovation_scale = sqrt(1.0 - phi**2)
    for t in range(1, n):
        state[t] = phi * state[t - 1] + innovation_scale * eta[t]

    # Returns: GBM noise plus a drift tilt from the PREVIOUS bar's state, so
    # the edge is genuinely predictable rather than contemporaneous.
    z = rng.standard_normal(n)
    log_returns = np.empty(n)
    log_returns[0] = 0.0
    log_returns[1:] = (
        mu_dt + cfg.signal_alpha * sigma_dt * state[:-1] + sigma_dt * z[1:]
    )

    closes = cfg.initial_price * np.exp(np.cumsum(log_returns))
    volumes = cfg.average_daily_volume * np.exp(
        cfg.volume_log_vol * rng.standard_normal(n) - 0.5 * cfg.volume_log_vol**2
    )

    return SyntheticSeries(
        closes=closes,
        volumes=volumes,
        log_returns=log_returns,
        latent_state=state,
        config=cfg,
        seed=seed,
    )


def oracle_sharpe(series: SyntheticSeries) -> float:
    """Annualised Sharpe of a strategy that knows the latent state exactly.

    This is the ceiling: an omniscient, cost-free trader who sees ``s_{t-1}``
    and takes a full long or short position accordingly. No real strategy can
    reach it, because ``s`` is not observable. It is reported alongside the
    degradation table so the honest out-of-sample number can be read against
    what was actually injected rather than against zero.

    Args:
        series: The generated series, including its ground-truth state.

    Returns:
        Annualised Sharpe ratio of the oracle strategy.
    """
    positions = np.sign(series.latent_state[:-1])
    realised = series.log_returns[1:]
    pnl = positions * realised
    if pnl.std(ddof=1) == 0:
        return 0.0
    return float(pnl.mean() / pnl.std(ddof=1) * sqrt(series.config.bars_per_year))
