"""The degradation study: the same strategy, five times, getting worse.

Each stage removes one comfortable assumption. The strategy, the data and
the random seed are identical throughout, so every drop in Sharpe is
attributable to the assumption that was removed and to nothing else.

A detail that matters for the comparison to be fair: stages 1-4 are scored
over *exactly* the same bars as stage 5. Walk-forward necessarily discards
the first training window, so if the single-fit stages were scored over the
full sample they would be measured on a different period and the comparison
would be contaminated. Here the only thing that changes between stage 4 and
stage 5 is how the parameter was chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Sequence

import numpy as np

from .commission import CommissionModel, PerShareCommission, ZeroCommission
from .data import Bar
from .engine import BacktestResult, run_backtest
from .metrics import PerformanceMetrics, compute_metrics
from .slippage import SlippageModel, SpreadPlusImpactSlippage, ZeroSlippage
from .strategy import LookAheadMomentumStrategy, TimeSeriesMomentumStrategy
from .synthetic import SyntheticConfig, generate_price_series, oracle_sharpe
from .walkforward import Fold, WalkForwardSplitter

#: Lookbacks the study is allowed to choose from. Seven values is enough to
#: overfit noticeably and small enough to keep the whole run under a minute.
DEFAULT_LOOKBACK_GRID: tuple[int, ...] = (5, 10, 15, 20, 30, 45, 60)

#: Volatility-normalisation window and its minimum fill, shared by every run.
VOL_WINDOW = 90
MIN_VOL_OBSERVATIONS = 30

#: Bars replayed before a scored window so trailing statistics are warm.
WARMUP = max(DEFAULT_LOOKBACK_GRID) + MIN_VOL_OBSERVATIONS + 5

DEFAULT_INITIAL_CAPITAL = 1_000_000.0
DEFAULT_REBALANCE_THRESHOLD = 0.05


@dataclass(frozen=True)
class StudyConfig:
    """Configuration for one degradation study.

    Attributes:
        synthetic: Price process parameters.
        seed: Random seed for the price path.
        lookback_grid: Candidate momentum lookbacks.
        train_size: Walk-forward training window, in bars.
        test_size: Walk-forward test window, in bars.
        initial_capital: Starting cash for every run.
        rebalance_threshold: No-trade band as a fraction of equity.
        half_spread_bps: Half-spread used by the slippage model.
        impact_coefficient: Square-root impact coefficient.
        commission_per_share: Per-share fee.
        commission_minimum: Minimum fee per order.
    """

    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    seed: int = 20260921
    lookback_grid: tuple[int, ...] = DEFAULT_LOOKBACK_GRID
    train_size: int = 504
    test_size: int = 252
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    rebalance_threshold: float = DEFAULT_REBALANCE_THRESHOLD
    half_spread_bps: float = 2.0
    impact_coefficient: float = 0.6
    commission_per_share: float = 0.005
    commission_minimum: float = 1.0

    def slippage(self) -> SlippageModel:
        """Build the realistic slippage model from the configured parameters."""
        bar_vol = self.synthetic.annual_vol / (self.synthetic.bars_per_year**0.5)
        return SpreadPlusImpactSlippage(
            half_spread_bps=self.half_spread_bps,
            impact_coefficient=self.impact_coefficient,
            bar_volatility=bar_vol,
        )

    def commission(self) -> CommissionModel:
        """Build the commission model from the configured parameters."""
        return PerShareCommission(
            per_share=self.commission_per_share, minimum=self.commission_minimum
        )


@dataclass(frozen=True)
class StageResult:
    """One rung of the degradation ladder.

    Attributes:
        index: 1-based stage number.
        name: Short label for the table.
        assumption_removed: What this stage stopped pretending.
        metrics: Performance over the scored window.
        chosen_lookback: The parameter(s) used. A list for walk-forward,
            where a different value may be chosen for each fold.
        honest: Whether this number is one a reviewer should take seriously.
    """

    index: int
    name: str
    assumption_removed: str
    metrics: PerformanceMetrics
    chosen_lookback: list[int]
    honest: bool


@dataclass(frozen=True)
class StudyResult:
    """The complete study for one seed.

    Attributes:
        stages: The five stages in order.
        oracle_sharpe: Sharpe of a cost-free trader who knows the latent
            state exactly. The ceiling that was injected into the data.
        buy_hold_sharpe: Sharpe of buy-and-hold over the same window, with
            costs, as a benchmark.
        config: The configuration used.
        scored_start: First scored bar index, shared by every stage.
        scored_end: One past the last scored bar index.
        folds: The walk-forward folds used by stage 5.
        n_events: Events processed by the stage-5 engine, as evidence the
            event loop actually ran.
    """

    stages: list[StageResult]
    oracle_sharpe: float
    buy_hold_sharpe: float
    config: StudyConfig
    scored_start: int
    scored_end: int
    folds: list[Fold]
    n_events: dict[str, int]

    @property
    def honest_sharpe(self) -> float:
        """The only number in the study that is out of sample."""
        return self.stages[-1].metrics.sharpe

    @property
    def is_monotone(self) -> bool:
        """Whether Sharpe fell at every rung. Reported, never assumed."""
        sharpes = [s.metrics.sharpe for s in self.stages]
        return all(b <= a for a, b in zip(sharpes, sharpes[1:]))


def _momentum_factory(lookback: int):
    """Return a factory building an honest momentum strategy with ``lookback``."""

    def factory(events, data):
        return TimeSeriesMomentumStrategy(
            events=events,
            data=data,
            symbol=data.symbols[0],
            lookback=lookback,
            vol_window=VOL_WINDOW,
            min_vol_observations=MIN_VOL_OBSERVATIONS,
        )

    return factory


def _look_ahead_factory(lookback: int):
    """Return a factory building the deliberately leaky strategy."""

    def factory(events, data):
        return LookAheadMomentumStrategy(
            events=events,
            data=data,
            symbol=data.symbols[0],
            lookback=lookback,
            vol_window=VOL_WINDOW,
            min_vol_observations=MIN_VOL_OBSERVATIONS,
        )

    return factory


def _run(
    bars: Sequence[Bar],
    lookback: int,
    cfg: StudyConfig,
    *,
    slippage: SlippageModel,
    commission: CommissionModel,
    warmup: int,
    look_ahead: bool = False,
) -> BacktestResult:
    """Run one backtest over ``bars`` with the given friction settings."""
    factory = _look_ahead_factory(lookback) if look_ahead else _momentum_factory(lookback)
    return run_backtest(
        list(bars),
        factory,
        slippage=slippage,
        commission=commission,
        initial_capital=cfg.initial_capital,
        rebalance_threshold=cfg.rebalance_threshold,
        warmup=warmup,
        bars_per_year=cfg.synthetic.bars_per_year,
        allow_look_ahead=look_ahead,
    )


def fit_in_sample(
    bars: Sequence[Bar],
    cfg: StudyConfig,
    *,
    slippage: SlippageModel,
    commission: CommissionModel,
    warmup: int,
    look_ahead: bool = False,
) -> tuple[int, BacktestResult]:
    """Pick the lookback that maximises Sharpe on the very window being reported.

    This is the bad practice the study is built to expose: the parameter is
    chosen with full knowledge of the period it is then scored on. It is done
    deliberately and it is labelled as such in the output.

    Args:
        bars: Bars to replay.
        cfg: Study configuration.
        slippage: Slippage model.
        commission: Commission model.
        warmup: Leading bars excluded from scoring.
        look_ahead: Use the leaky handler and strategy.

    Returns:
        ``(best_lookback, result_for_that_lookback)``.
    """
    best: tuple[float, int, BacktestResult] | None = None
    for lookback in cfg.lookback_grid:
        result = _run(
            bars,
            lookback,
            cfg,
            slippage=slippage,
            commission=commission,
            warmup=warmup,
            look_ahead=look_ahead,
        )
        sharpe = result.metrics().sharpe
        if best is None or sharpe > best[0]:
            best = (sharpe, lookback, result)
    assert best is not None, "lookback grid must not be empty"
    return best[1], best[2]


def run_walk_forward(
    bars: Sequence[Bar],
    cfg: StudyConfig,
    folds: Sequence[Fold],
    *,
    slippage: SlippageModel,
    commission: CommissionModel,
) -> tuple[PerformanceMetrics, list[int], dict[str, int]]:
    """Re-fit on each training window, score only on the window that follows.

    Returns for each fold's test window are stitched into a single
    out-of-sample series. The stitching is done on *returns*, not on equity
    levels: concatenating equity curves across folds would invent a jump at
    each boundary.

    Args:
        bars: The full bar series.
        cfg: Study configuration.
        folds: Folds from :class:`~honest_backtest.walkforward.WalkForwardSplitter`.
        slippage: Slippage model.
        commission: Commission model.

    Returns:
        ``(metrics, chosen_lookback_per_fold, event_counts)``.
    """
    stitched: list[np.ndarray] = []
    chosen: list[int] = []
    commission_paid = 0.0
    slippage_paid = 0.0
    notional = 0.0
    trades = 0
    events_total: dict[str, int] = {}

    for fold in folds:
        train_bars = bars[fold.train_slice()]
        # Train-window runs are scored after their own warm-up; this is
        # in-sample by design -- that is what "fitting" means -- but it only
        # ever touches bars strictly before the test window.
        best_lookback, _ = fit_in_sample(
            train_bars,
            cfg,
            slippage=slippage,
            commission=commission,
            warmup=min(WARMUP, max(0, len(train_bars) - 2)),
        )
        chosen.append(best_lookback)

        eval_bars = bars[fold.eval_slice()]
        result = _run(
            eval_bars,
            best_lookback,
            cfg,
            slippage=slippage,
            commission=commission,
            warmup=fold.warmup,
        )
        stitched.append(result.returns())
        commission_paid += result.total_commission
        slippage_paid += result.total_slippage
        notional += result.traded_notional
        trades += result.n_trades
        for key, value in result.event_counts.items():
            events_total[key] = events_total.get(key, 0) + value

    returns = np.concatenate(stitched) if stitched else np.zeros(0)
    curve = cfg.initial_capital * np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    metrics = compute_metrics(
        curve,
        bars_per_year=cfg.synthetic.bars_per_year,
        n_trades=trades,
        commission_cost=commission_paid,
        slippage_cost=slippage_paid,
        traded_notional=notional,
        returns=returns,
    )
    return metrics, chosen, events_total


def run_study(cfg: StudyConfig | None = None) -> StudyResult:
    """Run the full five-stage degradation study for one seed.

    Args:
        cfg: Study configuration; defaults to :class:`StudyConfig`.

    Returns:
        A :class:`StudyResult` containing every stage.

    Raises:
        ValueError: If the configured series is too short for even one fold.
    """
    cfg = cfg or StudyConfig()
    series = generate_price_series(cfg.synthetic, seed=cfg.seed)
    all_bars = series.to_bars()

    splitter = WalkForwardSplitter(
        train_size=cfg.train_size, test_size=cfg.test_size, warmup=WARMUP
    )
    folds = splitter.split(len(all_bars))
    if not folds:
        raise ValueError(
            f"series of {len(all_bars)} bars is too short for train_size="
            f"{cfg.train_size} and test_size={cfg.test_size}"
        )

    scored_start = folds[0].test_start
    scored_end = folds[-1].test_end
    bars = all_bars[:scored_end]

    zero_slip: SlippageModel = ZeroSlippage()
    zero_comm: CommissionModel = ZeroCommission()
    real_slip = cfg.slippage()
    real_comm = cfg.commission()

    stage_specs = [
        (
            "Naive backtest",
            "nothing -- look-ahead present, zero costs",
            zero_slip,
            zero_comm,
            True,
        ),
        (
            "+ point-in-time data",
            "look-ahead bias (signal alignment and full-sample scaling)",
            zero_slip,
            zero_comm,
            False,
        ),
        ("+ slippage", "free execution at the close", real_slip, zero_comm, False),
        ("+ commissions", "zero brokerage fees", real_slip, real_comm, False),
    ]

    stages: list[StageResult] = []
    for i, (name, removed, slip, comm, leak) in enumerate(stage_specs, start=1):
        lookback, result = fit_in_sample(
            bars,
            cfg,
            slippage=slip,
            commission=comm,
            warmup=scored_start,
            look_ahead=leak,
        )
        stages.append(
            StageResult(
                index=i,
                name=name,
                assumption_removed=removed,
                metrics=result.metrics(),
                chosen_lookback=[lookback],
                honest=False,
            )
        )

    wf_metrics, wf_lookbacks, wf_events = run_walk_forward(
        all_bars, cfg, folds, slippage=real_slip, commission=real_comm
    )
    stages.append(
        StageResult(
            index=5,
            name="+ walk-forward OOS",
            assumption_removed="a single in-sample parameter fit",
            metrics=wf_metrics,
            chosen_lookback=wf_lookbacks,
            honest=True,
        )
    )

    # Benchmark: buy and hold over the same scored window, paying the same
    # frictions. A strategy that cannot beat this has not earned its fees.
    from .strategy import BuyAndHoldStrategy

    def bh_factory(events, data):
        return BuyAndHoldStrategy(events, data, symbol=data.symbols[0])

    from .engine import run_backtest as _rb

    bh = _rb(
        bars,
        bh_factory,
        slippage=real_slip,
        commission=real_comm,
        initial_capital=cfg.initial_capital,
        rebalance_threshold=cfg.rebalance_threshold,
        warmup=scored_start,
        bars_per_year=cfg.synthetic.bars_per_year,
    )

    return StudyResult(
        stages=stages,
        oracle_sharpe=oracle_sharpe(series),
        buy_hold_sharpe=bh.metrics().sharpe,
        config=cfg,
        scored_start=scored_start,
        scored_end=scored_end,
        folds=folds,
        n_events=wf_events,
    )


@dataclass(frozen=True)
class SeedSweep:
    """Stage Sharpe ratios across many independent price paths.

    Attributes:
        seeds: The seeds used.
        stage_names: Stage labels, in order.
        sharpes: ``sharpes[stage][seed]`` annualised Sharpe.
        monotone_flags: Per-seed flag for whether Sharpe fell at every rung.
            Reported rather than assumed: the degradation is a strong
            tendency, not a theorem, and a study that hid the exceptions
            would be committing the sin it is about.
    """

    seeds: list[int]
    stage_names: list[str]
    sharpes: list[list[float]]
    monotone_flags: list[bool] = field(default_factory=list)

    @property
    def n_monotone(self) -> int:
        """How many paths produced a strictly non-increasing ladder."""
        return sum(self.monotone_flags)

    def summary(self, stage_index: int) -> tuple[float, float, float]:
        """Return ``(mean, stdev, t_stat)`` for one stage across seeds.

        The t-statistic is ``mean / (stdev / sqrt(n))``, testing the null that
        the stage's true Sharpe is zero. It is the number that decides whether
        the honest result is an edge or a coin flip.
        """
        values = self.sharpes[stage_index]
        n = len(values)
        if n < 2:
            return (values[0] if values else 0.0), 0.0, 0.0
        m = mean(values)
        sd = pstdev(values) * (n / (n - 1)) ** 0.5
        t = m / (sd / n**0.5) if sd > 0 else 0.0
        return m, sd, t


def run_seed_sweep(
    n_seeds: int = 12, base_config: StudyConfig | None = None
) -> SeedSweep:
    """Repeat the whole study on ``n_seeds`` independent price paths.

    A single seed proves nothing: any ladder can be produced by picking the
    path that produces it. Re-running across independent paths is what turns
    the headline into a claim about the *method* rather than about one lucky
    sample.

    Args:
        n_seeds: Number of independent paths.
        base_config: Configuration to clone; only the seed varies.

    Returns:
        A :class:`SeedSweep`.
    """
    base = base_config or StudyConfig()
    seeds = [base.seed + 1000 * i for i in range(n_seeds)]
    stage_names: list[str] = []
    collected: list[list[float]] = []
    monotone: list[bool] = []

    for seed in seeds:
        from dataclasses import replace

        result = run_study(replace(base, seed=seed))
        if not stage_names:
            stage_names = [s.name for s in result.stages]
            collected = [[] for _ in stage_names]
        for i, stage in enumerate(result.stages):
            collected[i].append(stage.metrics.sharpe)
        monotone.append(result.is_monotone)

    return SeedSweep(
        seeds=seeds,
        stage_names=stage_names,
        sharpes=collected,
        monotone_flags=monotone,
    )
