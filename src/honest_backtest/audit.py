"""Auditing your own strategy on your own data.

``honest-backtest audit`` runs the degradation ladder from the README on a
strategy and a price file you supply, and reports how much of the in-sample
Sharpe survives realistic costs and walk-forward validation. There is no
oracle here: on real data nobody knows how big the true edge is, which is
exactly why the out-of-sample rung is the only one worth quoting.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .commission import CommissionModel, PercentOfNotionalCommission, PerShareCommission
from .data import Bar
from .experiments import LadderResult, LadderSettings, momentum_spec, run_ladder
from .leaks import LeakReport, detect_look_ahead
from .report import render_costs, render_markdown_table, render_table
from .slippage import SlippageModel, SpreadPlusImpactSlippage
from .spec import StrategySpec, param_names

#: Names accepted by ``--strategy`` that refer to strategies shipped with the package.
BUILT_IN_STRATEGIES = {"momentum": momentum_spec}


def load_spec(reference: str) -> StrategySpec:
    """Resolve ``--strategy`` to a :class:`StrategySpec`.

    Args:
        reference: A built-in name (``momentum``), a Python file whose
            module-level ``SPEC`` is a spec (``my_strategy.py``), or a file
            and attribute (``my_strategy.py:FAST_SPEC``). The attribute may
            also be a zero-argument function returning a spec.

    Returns:
        The strategy spec.

    Raises:
        ValueError: If the reference does not resolve to a spec.
    """
    if reference in BUILT_IN_STRATEGIES:
        return BUILT_IN_STRATEGIES[reference]()

    path_text, _, attr = reference.partition(":")
    path = Path(path_text)
    if not path.is_file():
        known = ", ".join(sorted(BUILT_IN_STRATEGIES))
        raise ValueError(
            f"--strategy {reference!r} is neither a file nor a built-in ({known})"
        )
    module_name = f"_honest_backtest_user_{path.stem}"
    loader_spec = importlib.util.spec_from_file_location(module_name, path)
    if loader_spec is None or loader_spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(loader_spec)
    sys.modules[module_name] = module
    loader_spec.loader.exec_module(module)

    attr = attr or "SPEC"
    if not hasattr(module, attr):
        raise ValueError(f"{path} defines no {attr!r}; expected a StrategySpec")
    value = getattr(module, attr)
    if callable(value) and not isinstance(value, StrategySpec):
        value = value()
    if not isinstance(value, StrategySpec):
        raise ValueError(f"{path}:{attr} is a {type(value).__name__}, not a StrategySpec")
    return value


def estimate_bar_volatility(bars: Sequence[Bar]) -> float:
    """Per-bar standard deviation of log returns over ``bars``.

    The impact model needs a volatility. The audit estimates it from the
    first training window only, which ends before any scored bar, so the
    cost model cannot borrow information from the period being judged.
    """
    closes = np.array([b.close for b in bars], dtype=float)
    if closes.size < 3:
        raise ValueError("need at least 3 bars to estimate volatility")
    return float(np.diff(np.log(closes)).std(ddof=1))


@dataclass(frozen=True)
class AuditConfig:
    """Costs and validation settings for an audit.

    Attributes:
        settings: Account and walk-forward settings.
        half_spread_bps: Half-spread paid on every fill.
        impact_coefficient: Square-root impact coefficient.
        commission_bps: If set, charge this many bps of notional instead of a
            per-share fee. The right shape for FX, crypto and most non-US
            equity brokers.
        commission_per_share: Per-share fee when ``commission_bps`` is unset.
        commission_minimum: Minimum per-share-model fee per order.
        check_leaks: Before running the ladder, replay the strategy with the
            future replaced at several points and confirm no past signal
            changes (see :mod:`honest_backtest.leaks`).
    """

    settings: LadderSettings = LadderSettings()
    half_spread_bps: float = 2.0
    impact_coefficient: float = 0.6
    commission_bps: float | None = None
    commission_per_share: float = 0.005
    commission_minimum: float = 1.0
    check_leaks: bool = True

    def commission(self) -> CommissionModel:
        """Build the configured commission model."""
        if self.commission_bps is not None:
            return PercentOfNotionalCommission(self.commission_bps)
        return PerShareCommission(
            per_share=self.commission_per_share, minimum=self.commission_minimum
        )

    def slippage(self, bars: Sequence[Bar]) -> SlippageModel:
        """Build the impact model, calibrated on the first training window."""
        return SpreadPlusImpactSlippage(
            half_spread_bps=self.half_spread_bps,
            impact_coefficient=self.impact_coefficient,
            bar_volatility=estimate_bar_volatility(bars[: self.settings.train_size]),
        )


@dataclass(frozen=True)
class AuditResult:
    """A ladder run on user data, with what is needed to report it."""

    ladder: LadderResult
    spec: StrategySpec
    bars: list[Bar]
    config: AuditConfig
    leaks: LeakReport | None = None

    @property
    def years_scored(self) -> float:
        """Length of the scored window in years."""
        n = self.ladder.scored_end - self.ladder.scored_start
        return n / self.config.settings.bars_per_year

    @property
    def honest_t_stat(self) -> float:
        """Approximate t-statistic of the out-of-sample Sharpe against zero.

        An annualised Sharpe ``S`` measured over ``Y`` years has a standard
        error of roughly ``1 / sqrt(Y)``, so ``S * sqrt(Y)`` is the number of
        standard errors it sits from zero.
        """
        return self.ladder.honest_sharpe * math.sqrt(self.years_scored)


def run_audit(
    bars: Sequence[Bar], spec: StrategySpec, config: AuditConfig | None = None
) -> AuditResult:
    """Run the degradation ladder for ``spec`` on ``bars``."""
    config = config or AuditConfig()
    bars = list(bars)
    leaks = detect_look_ahead(bars, spec) if config.check_leaks else None
    ladder = run_ladder(
        bars,
        spec,
        config.settings,
        slippage=config.slippage(bars),
        commission=config.commission(),
    )
    return AuditResult(ladder=ladder, spec=spec, bars=bars, config=config, leaks=leaks)


def _when(bar: Bar) -> str:
    return bar.time.date().isoformat() if bar.time is not None else f"bar {bar.timestamp}"


def render_audit_report(result: AuditResult, *, markdown: bool = False) -> str:
    """Render an audit in the order a reader should meet it."""
    ladder, cfg = result.ladder, result.config.settings
    first = result.bars[ladder.scored_start]
    last = result.bars[ladder.scored_end - 1]
    in_sample = ladder.stages[0].metrics.sharpe
    honest = ladder.honest_sharpe
    t = result.honest_t_stat
    verdict = (
        "distinguishable from zero at roughly the 5% level"
        if abs(t) > 2.0
        else "NOT distinguishable from zero"
    )
    leak_line = "skipped" if result.leaks is None else result.leaks.summary()
    lines = [
        f"honest-backtest audit - {result.spec.name}",
        f"  instrument          {first.symbol}",
        f"  data                {_when(result.bars[0])} to {_when(result.bars[-1])} "
        f"({len(result.bars)} bars)",
        f"  scored window       {_when(first)} to {_when(last)} "
        f"({result.years_scored:.1f} years, identical for every rung)",
        f"  walk-forward folds  {len(ladder.folds)} "
        f"(train {cfg.train_size}, test {cfg.test_size}, non-overlapping)",
        f"  settings tried      {len(result.spec.grid)} per fit",
        f"  next-bar fills      {cfg.fill_timing}",
        f"  look-ahead check    {leak_line}",
        "",
        render_table(ladder, param_header=param_names(result.spec)),
        "",
        render_costs(ladder),
        "",
        "Reading it:",
        f"  In-sample headline (rung 1)       Sharpe {in_sample:6.2f}"
        "   <- what a typical backtest reports",
        f"  Buy and hold, same window, costs  Sharpe {ladder.buy_hold_sharpe:6.2f}",
        f"  Walk-forward out of sample        Sharpe {honest:6.2f}"
        "   <- the only number worth quoting",
        f"  Out-of-sample t-stat ~ {t:.2f}: {verdict}.",
        f"  P(edge), out of sample: {ladder.stages[-1].p_edge * 100:.1f}% "
        "(Probabilistic Sharpe Ratio against zero)",
        f"  Sharpe fell monotonically at every rung: "
        f"{'yes' if ladder.is_monotone else 'NO - reported as measured'}",
    ]
    if result.leaks is not None and result.leaks.findings:
        lines.insert(
            0,
            "WARNING: this strategy's past signals change when the future does.\n"
            "Every number below is contaminated by look-ahead; fix the leak first.\n",
        )
    if len(result.spec.grid) > 1:
        lines.append(
            f"  The in-sample rungs picked the best of {len(result.spec.grid)} "
            "settings on\n  the window they report, so they are biased upward "
            "by construction."
        )
    if markdown:
        lines += [
            "",
            render_markdown_table(ladder, param_header=param_names(result.spec)),
        ]
    return "\n".join(lines)
