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

The ladder itself, :func:`run_ladder`, knows nothing about momentum or
synthetic data: it takes any :class:`~honest_backtest.spec.StrategySpec` and
any bar series. :func:`run_study` is that ladder applied to the built-in
strategy on a synthetic path with a known edge.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields, replace
from statistics import mean, pstdev
from typing import Any

import numpy as np

from . import fast as fastpath
from .commission import CommissionModel, PerShareCommission, ZeroCommission
from .data import Bar
from .engine import BacktestResult, run_backtest
from .frictions import MarketFrictions
from .metrics import (
    PerformanceMetrics,
    compute_metrics,
    deflated_sharpe,
    probabilistic_sharpe,
)
from .slippage import SlippageModel, SpreadPlusImpactSlippage, ZeroSlippage
from .spec import Params, StrategySpec, param_grid
from .strategy import (
    BuyAndHoldStrategy,
    LookAheadMomentumStrategy,
    TimeSeriesMomentumStrategy,
)
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

    def settings(self) -> LadderSettings:
        """The shared account and walk-forward settings for the ladder."""
        return LadderSettings(
            train_size=self.train_size,
            test_size=self.test_size,
            initial_capital=self.initial_capital,
            rebalance_threshold=self.rebalance_threshold,
            bars_per_year=self.synthetic.bars_per_year,
        )


@dataclass(frozen=True)
class LadderSettings:
    """The account and validation settings every rung of a ladder shares.

    Attributes:
        train_size: Walk-forward training window, in bars.
        test_size: Walk-forward test window, in bars.
        initial_capital: Starting cash for every run.
        rebalance_threshold: No-trade band as a fraction of equity.
        bars_per_year: Annualisation factor.
        fill_timing: When orders fill once the ladder stops assuming they
            fill at the signal bar's close (see
            :class:`~honest_backtest.execution.SimulatedExecutionHandler`).
            ``"close"`` keeps that assumption and the ladder has no
            next-bar rung; anything else adds a "+ next-bar execution" rung
            after commissions, and the walk-forward rung inherits it.
        frictions: Liquidity, order-style and carry frictions. When any is
            active the ladder gains a "+ liquidity and carry" rung after the
            execution rungs, and the walk-forward rung and buy-and-hold
            inherit them.
        fast: Use :mod:`~honest_backtest.fast`, which records each setting's
            signals once and replays the accounting without the event
            queue. Results are identical (the test suite asserts it); runs
            the fast path cannot reproduce fall back to the engine.
    """

    train_size: int = 504
    test_size: int = 252
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    rebalance_threshold: float = DEFAULT_REBALANCE_THRESHOLD
    bars_per_year: int = 252
    fill_timing: str = "close"
    frictions: MarketFrictions | None = None
    fast: bool = False


@dataclass(frozen=True)
class StageResult:
    """One rung of the degradation ladder.

    Attributes:
        index: 1-based stage number.
        name: Short label for the table.
        assumption_removed: What this stage stopped pretending.
        metrics: Performance over the scored window.
        chosen_params: The parameter setting(s) used. One per walk-forward
            fold for the out-of-sample stage, where each fold re-fits.
        honest: Whether this number is one a reviewer should take seriously.
        p_edge: Probability the rung's Sharpe reflects a real edge rather
            than luck. For an in-sample rung this is the Deflated Sharpe
            Ratio, which discounts for having picked the best of the grid;
            for the walk-forward rung, which picked nothing on the scored
            data, it is the Probabilistic Sharpe Ratio against zero.
        returns: Per-bar returns over the scored window, for plotting.
    """

    index: int
    name: str
    assumption_removed: str
    metrics: PerformanceMetrics
    chosen_params: list[Params]
    honest: bool
    p_edge: float = float("nan")
    returns: np.ndarray = field(
        default_factory=lambda: np.zeros(0), compare=False, repr=False
    )

    @property
    def chosen_lookback(self) -> list[Any]:
        """The chosen value of a single-parameter grid, one per fit."""
        return [next(iter(p.values())) for p in self.chosen_params]


@dataclass(frozen=True)
class LadderResult:
    """Every rung of the ladder for one strategy on one bar series.

    Attributes:
        stages: The rungs in order; the last is the out-of-sample one.
        buy_hold_sharpe: Sharpe of buy-and-hold over the same window, with
            costs, as a benchmark.
        scored_start: First scored bar index, shared by every stage.
        scored_end: One past the last scored bar index.
        folds: The walk-forward folds used by the last stage.
        n_events: Events processed by the out-of-sample engine runs, as
            evidence the event loop actually ran.
        buy_hold_returns: Per-bar returns of the benchmark, for plotting.
    """

    stages: list[StageResult]
    buy_hold_sharpe: float
    scored_start: int
    scored_end: int
    folds: list[Fold]
    n_events: dict[str, int]
    buy_hold_returns: np.ndarray = field(compare=False, repr=False)

    @property
    def honest_sharpe(self) -> float:
        """The only number in the ladder that is out of sample."""
        return self.stages[-1].metrics.sharpe

    @property
    def is_monotone(self) -> bool:
        """Whether Sharpe fell at every rung. Reported, never assumed."""
        sharpes = [s.metrics.sharpe for s in self.stages]
        return all(b <= a for a, b in zip(sharpes, sharpes[1:], strict=False))


@dataclass(frozen=True)
class StudyResult(LadderResult):
    """The complete synthetic study for one seed.

    Attributes:
        oracle_sharpe: Sharpe of a cost-free trader who knows the latent
            state exactly. The ceiling that was injected into the data.
        config: The configuration used.
    """

    oracle_sharpe: float = 0.0
    config: StudyConfig = field(default_factory=StudyConfig)


def _build_momentum(events, data, symbol, *, lookback):
    return TimeSeriesMomentumStrategy(
        events=events,
        data=data,
        symbol=symbol,
        lookback=lookback,
        vol_window=VOL_WINDOW,
        min_vol_observations=MIN_VOL_OBSERVATIONS,
    )


def _build_look_ahead_momentum(events, data, symbol, *, lookback):
    return LookAheadMomentumStrategy(
        events=events,
        data=data,
        symbol=symbol,
        lookback=lookback,
        vol_window=VOL_WINDOW,
        min_vol_observations=MIN_VOL_OBSERVATIONS,
    )


def momentum_spec(lookback_grid: Sequence[int] = DEFAULT_LOOKBACK_GRID) -> StrategySpec:
    """The built-in time-series momentum strategy, with its leaky twin."""
    return StrategySpec(
        name="time-series momentum",
        build=_build_momentum,
        build_look_ahead=_build_look_ahead_momentum,
        grid=param_grid(lookback=tuple(lookback_grid)),
        warmup=WARMUP,
    )


def _run(
    bars: Sequence[Bar],
    spec: StrategySpec,
    params: Params,
    settings: LadderSettings,
    *,
    slippage: SlippageModel,
    commission: CommissionModel,
    warmup: int,
    look_ahead: bool = False,
    fill_timing: str = "close",
    realistic: bool = False,
    cache: fastpath.SignalCache | None = None,
) -> BacktestResult:
    """Run one backtest over ``bars`` with the given friction settings.

    ``realistic`` applies ``settings.frictions`` as well. With
    ``settings.fast`` and a ``cache``, the fast path is used whenever it
    reproduces the engine exactly.
    """
    extra = (
        settings.frictions.run_kwargs()
        if realistic and settings.frictions is not None
        else {}
    )
    if (
        settings.fast
        and cache is not None
        and fastpath.supports(fill_timing=fill_timing, **extra)
    ):
        signals, leaked = cache.get(bars, spec, params, look_ahead)
        return fastpath.simulate(
            bars,
            signals,
            slippage=slippage,
            commission=commission,
            initial_capital=settings.initial_capital,
            rebalance_threshold=settings.rebalance_threshold,
            warmup=warmup,
            bars_per_year=settings.bars_per_year,
            fill_timing=fill_timing,
            used_look_ahead=leaked,
        )
    return run_backtest(
        list(bars),
        spec.factory(params, look_ahead=look_ahead),
        slippage=slippage,
        commission=commission,
        initial_capital=settings.initial_capital,
        rebalance_threshold=settings.rebalance_threshold,
        warmup=warmup,
        bars_per_year=settings.bars_per_year,
        allow_look_ahead=look_ahead,
        symbol=bars[0].symbol,
        fill_timing=fill_timing,
        **extra,
    )


def fit_in_sample(
    bars: Sequence[Bar],
    spec: StrategySpec,
    settings: LadderSettings,
    *,
    slippage: SlippageModel,
    commission: CommissionModel,
    warmup: int,
    look_ahead: bool = False,
    fill_timing: str = "close",
    trial_sharpes: list[float] | None = None,
    realistic: bool = False,
    cache: fastpath.SignalCache | None = None,
) -> tuple[Params, BacktestResult]:
    """Pick the parameters that maximise Sharpe on the very window being reported.

    This is the bad practice the study is built to expose: the parameter is
    chosen with full knowledge of the period it is then scored on. It is done
    deliberately and it is labelled as such in the output.

    Args:
        bars: Bars to replay.
        spec: The strategy and its parameter grid.
        settings: Shared account settings.
        slippage: Slippage model.
        commission: Commission model.
        warmup: Leading bars excluded from scoring.
        look_ahead: Use the leaky handler and strategy.
        fill_timing: When orders fill.
        trial_sharpes: If given, the Sharpe of every setting tried is
            appended to it, in grid order, for the Deflated Sharpe Ratio.
        realistic: Also apply ``settings.frictions``.
        cache: Signals already replayed, for the fast path.

    Returns:
        ``(best_params, result_for_those_params)``.
    """
    best: tuple[float, Params, BacktestResult] | None = None
    for params in spec.grid:
        result = _run(
            bars,
            spec,
            params,
            settings,
            slippage=slippage,
            commission=commission,
            warmup=warmup,
            look_ahead=look_ahead,
            fill_timing=fill_timing,
            realistic=realistic,
            cache=cache,
        )
        sharpe = result.metrics().sharpe
        if trial_sharpes is not None:
            trial_sharpes.append(sharpe)
        if best is None or sharpe > best[0]:
            best = (sharpe, params, result)
    assert best is not None, "parameter grid must not be empty"
    return best[1], best[2]


def run_walk_forward(
    bars: Sequence[Bar],
    spec: StrategySpec,
    settings: LadderSettings,
    folds: Sequence[Fold],
    *,
    slippage: SlippageModel,
    commission: CommissionModel,
) -> tuple[PerformanceMetrics, list[Params], dict[str, int], np.ndarray]:
    """Re-fit on each training window, score only on the window that follows.

    Returns for each fold's test window are stitched into a single
    out-of-sample series. The stitching is done on *returns*, not on equity
    levels: concatenating equity curves across folds would invent a jump at
    each boundary.

    Args:
        bars: The full bar series.
        spec: The strategy and its parameter grid.
        settings: Shared account settings.
        folds: Folds from :class:`~honest_backtest.walkforward.WalkForwardSplitter`.
        slippage: Slippage model.
        commission: Commission model.

    Returns:
        ``(metrics, chosen_params_per_fold, event_counts, stitched_returns)``.
    """
    cache = fastpath.SignalCache() if settings.fast else None
    stitched: list[np.ndarray] = []
    chosen: list[Params] = []
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
        best_params, _ = fit_in_sample(
            train_bars,
            spec,
            settings,
            slippage=slippage,
            commission=commission,
            warmup=min(spec.warmup, max(0, len(train_bars) - 2)),
            fill_timing=settings.fill_timing,
            realistic=True,
            cache=cache,
        )
        chosen.append(best_params)

        eval_bars = bars[fold.eval_slice()]
        result = _run(
            eval_bars,
            spec,
            best_params,
            settings,
            slippage=slippage,
            commission=commission,
            warmup=fold.warmup,
            fill_timing=settings.fill_timing,
            realistic=True,
            cache=cache,
        )
        stitched.append(result.returns())
        commission_paid += result.total_commission
        slippage_paid += result.total_slippage
        notional += result.traded_notional
        trades += result.n_trades
        for key, value in result.event_counts.items():
            events_total[key] = events_total.get(key, 0) + value

    returns = np.concatenate(stitched) if stitched else np.zeros(0)
    curve = settings.initial_capital * np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    metrics = compute_metrics(
        curve,
        bars_per_year=settings.bars_per_year,
        n_trades=trades,
        commission_cost=commission_paid,
        slippage_cost=slippage_paid,
        traded_notional=notional,
        returns=returns,
    )
    return metrics, chosen, events_total, returns


def run_ladder(
    all_bars: Sequence[Bar],
    spec: StrategySpec,
    settings: LadderSettings,
    *,
    slippage: SlippageModel,
    commission: CommissionModel,
) -> LadderResult:
    """Run the degradation ladder for any strategy on any bar series.

    Every in-sample rung is scored over exactly the bars the walk-forward
    rung covers, so the only thing that changes from rung to rung is the
    assumption being removed. When ``spec`` has a look-ahead twin, the ladder
    opens with the naive, leaky backtest; otherwise it opens with a
    frictionless in-sample fit, which is what most backtests report.

    Args:
        all_bars: The full bar series, oldest first.
        spec: The strategy and its parameter grid.
        settings: Shared account and walk-forward settings.
        slippage: The realistic slippage model.
        commission: The realistic commission model.

    Returns:
        A :class:`LadderResult`.

    Raises:
        ValueError: If the series is too short for even one fold.
    """
    all_bars = list(all_bars)
    splitter = WalkForwardSplitter(
        train_size=settings.train_size, test_size=settings.test_size, warmup=spec.warmup
    )
    folds = splitter.split(len(all_bars))
    if not folds:
        raise ValueError(
            f"series of {len(all_bars)} bars is too short for train_size="
            f"{settings.train_size} and test_size={settings.test_size}"
        )

    scored_start = folds[0].test_start
    scored_end = folds[-1].test_end
    bars = all_bars[:scored_end]

    zero_slip: SlippageModel = ZeroSlippage()
    zero_comm: CommissionModel = ZeroCommission()

    stage_specs = []
    if spec.build_look_ahead is not None:
        stage_specs += [
            (
                "Naive backtest",
                "nothing -- look-ahead present, zero costs",
                zero_slip,
                zero_comm,
                True,
                "close",
            ),
            (
                "+ point-in-time data",
                "look-ahead bias (signal alignment and full-sample scaling)",
                zero_slip,
                zero_comm,
                False,
                "close",
            ),
        ]
    else:
        stage_specs.append(
            (
                "In-sample, no costs",
                "nothing -- zero costs, parameters fitted on the reported window",
                zero_slip,
                zero_comm,
                False,
                "close",
            )
        )
    stage_specs += [
        (
            "+ slippage",
            "free execution at the close",
            slippage,
            zero_comm,
            False,
            "close",
        ),
        ("+ commissions", "zero brokerage fees", slippage, commission, False, "close"),
    ]
    if settings.fill_timing != "close":
        stage_specs.append(
            (
                "+ next-bar execution",
                "filling at the close the signal was computed from",
                slippage,
                commission,
                False,
                settings.fill_timing,
            )
        )
    # Rungs from here on also carry liquidity, order-style and carry frictions.
    first_realistic = len(stage_specs) + 1
    if settings.frictions is not None and settings.frictions.active:
        stage_specs.append(
            (
                "+ liquidity and carry",
                "unlimited liquidity and free positions ("
                + settings.frictions.describe()
                + ")",
                slippage,
                commission,
                False,
                settings.fill_timing,
            )
        )

    cache = fastpath.SignalCache() if settings.fast else None
    stages: list[StageResult] = []
    for i, (name, removed, slip, comm, leak, timing) in enumerate(stage_specs, start=1):
        trials: list[float] = []
        params, result = fit_in_sample(
            bars,
            spec,
            settings,
            slippage=slip,
            commission=comm,
            warmup=scored_start,
            look_ahead=leak,
            fill_timing=timing,
            trial_sharpes=trials,
            realistic=i >= first_realistic,
            cache=cache,
        )
        stages.append(
            StageResult(
                index=i,
                name=name,
                assumption_removed=removed,
                metrics=result.metrics(),
                chosen_params=[params],
                honest=False,
                p_edge=deflated_sharpe(result.returns(), trials, settings.bars_per_year),
                returns=result.returns(),
            )
        )

    wf_metrics, wf_params, wf_events, wf_returns = run_walk_forward(
        all_bars, spec, settings, folds, slippage=slippage, commission=commission
    )
    stages.append(
        StageResult(
            index=len(stages) + 1,
            name="+ walk-forward OOS",
            assumption_removed="a single in-sample parameter fit",
            metrics=wf_metrics,
            chosen_params=wf_params,
            honest=True,
            p_edge=probabilistic_sharpe(wf_returns),
            returns=wf_returns,
        )
    )

    # Benchmark: buy and hold over the same scored window, paying the same
    # frictions. A strategy that cannot beat this has not earned its fees.
    def bh_factory(events, data):
        return BuyAndHoldStrategy(events, data, symbol=data.symbols[0])

    bh = run_backtest(
        bars,
        bh_factory,
        slippage=slippage,
        commission=commission,
        initial_capital=settings.initial_capital,
        rebalance_threshold=settings.rebalance_threshold,
        warmup=scored_start,
        bars_per_year=settings.bars_per_year,
        symbol=bars[0].symbol,
        fill_timing=settings.fill_timing,
        **(settings.frictions.run_kwargs() if settings.frictions is not None else {}),
    )

    return LadderResult(
        stages=stages,
        buy_hold_sharpe=bh.metrics().sharpe,
        scored_start=scored_start,
        scored_end=scored_end,
        folds=folds,
        n_events=wf_events,
        buy_hold_returns=bh.returns(),
    )


def run_study(cfg: StudyConfig | None = None, *, fast: bool = False) -> StudyResult:
    """Run the full five-stage degradation study for one seed.

    Args:
        cfg: Study configuration; defaults to :class:`StudyConfig`.
        fast: Use the fast path (identical results, see :mod:`~honest_backtest.fast`).

    Returns:
        A :class:`StudyResult` containing every stage.

    Raises:
        ValueError: If the configured series is too short for even one fold.
    """
    cfg = cfg or StudyConfig()
    series = generate_price_series(cfg.synthetic, seed=cfg.seed)
    ladder = run_ladder(
        series.to_bars(),
        momentum_spec(cfg.lookback_grid),
        replace(cfg.settings(), fast=fast),
        slippage=cfg.slippage(),
        commission=cfg.commission(),
    )
    return StudyResult(
        **{f.name: getattr(ladder, f.name) for f in fields(LadderResult)},
        oracle_sharpe=oracle_sharpe(series),
        config=cfg,
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


def _sweep_one(cfg: StudyConfig, fast: bool) -> tuple[list[str], list[float], bool]:
    """One seed of a sweep: stage names, stage Sharpes, monotone flag."""
    result = run_study(cfg, fast=fast)
    return (
        [s.name for s in result.stages],
        [s.metrics.sharpe for s in result.stages],
        result.is_monotone,
    )


def run_seed_sweep(
    n_seeds: int = 12,
    base_config: StudyConfig | None = None,
    *,
    fast: bool = False,
    workers: int = 1,
) -> SeedSweep:
    """Repeat the whole study on ``n_seeds`` independent price paths.

    A single seed proves nothing: any ladder can be produced by picking the
    path that produces it. Re-running across independent paths is what turns
    the headline into a claim about the *method* rather than about one lucky
    sample.

    Args:
        n_seeds: Number of independent paths.
        base_config: Configuration to clone; only the seed varies.
        fast: Use the fast path for every study (identical results).
        workers: Processes to spread the seeds over. Each seed is
            independent and deterministic, so the result does not depend on
            how many there are; 1 runs in this process.

    Returns:
        A :class:`SeedSweep`.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    base = base_config or StudyConfig()
    seeds = [base.seed + 1000 * i for i in range(n_seeds)]
    configs = [replace(base, seed=seed) for seed in seeds]
    if workers > 1 and n_seeds > 1:
        with ProcessPoolExecutor(max_workers=min(workers, n_seeds)) as pool:
            outcomes = list(pool.map(_sweep_one, configs, [fast] * n_seeds))
    else:
        outcomes = [_sweep_one(cfg, fast) for cfg in configs]

    stage_names = outcomes[0][0] if outcomes else []
    collected: list[list[float]] = [[] for _ in stage_names]
    for _, sharpes, _ in outcomes:
        for i, sharpe in enumerate(sharpes):
            collected[i].append(sharpe)

    return SeedSweep(
        seeds=seeds,
        stage_names=stage_names,
        sharpes=collected,
        monotone_flags=[flag for _, _, flag in outcomes],
    )
