"""Rendering of study results as plain-text and Markdown tables."""

from __future__ import annotations

from .experiments import SeedSweep, StudyResult

_HEADERS = (
    "#",
    "Stage",
    "Sharpe",
    "Total return",
    "Ann. return",
    "Max DD",
    "Trades",
    "Lookback",
)


def _fmt_lookback(values: list[int]) -> str:
    """Render the chosen lookback(s) compactly."""
    if len(values) == 1:
        return str(values[0])
    unique = sorted(set(values))
    if len(unique) == 1:
        return f"{unique[0]} (all folds)"
    return "/".join(str(v) for v in values)


def _rows(result: StudyResult) -> list[tuple[str, ...]]:
    """Build the table body."""
    rows: list[tuple[str, ...]] = []
    for stage in result.stages:
        m = stage.metrics
        rows.append(
            (
                str(stage.index),
                stage.name,
                f"{m.sharpe:.2f}",
                f"{m.total_return * 100:+.1f}%",
                f"{m.annual_return * 100:+.1f}%",
                f"{m.max_drawdown * 100:.1f}%",
                f"{m.n_trades}",
                _fmt_lookback(stage.chosen_lookback),
            )
        )
    return rows


def render_table(result: StudyResult) -> str:
    """Render the degradation table as aligned plain text."""
    rows = _rows(result)
    widths = [
        max(len(_HEADERS[i]), max(len(r[i]) for r in rows)) for i in range(len(_HEADERS))
    ]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(_HEADERS)).rstrip()
    sep = "  ".join("-" * widths[i] for i in range(len(_HEADERS)))
    body = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        for row in rows
    ]
    return "\n".join([line, sep, *body])


def render_markdown_table(result: StudyResult) -> str:
    """Render the degradation table as Markdown, for pasting into the README."""
    rows = _rows(result)
    head = "| " + " | ".join(_HEADERS) + " |"
    sep = "| " + " | ".join("---" for _ in _HEADERS) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def render_costs(result: StudyResult) -> str:
    """Render where the money went, stage by stage."""
    lines = ["Cost attribution (account currency, over the scored window):"]
    for stage in result.stages:
        m = stage.metrics
        lines.append(
            f"  {stage.index}. {stage.name:<22} "
            f"slippage {m.slippage_cost:>12,.0f}   "
            f"commission {m.commission_cost:>10,.0f}   "
            f"turnover {m.annual_turnover:>5.1f}x/yr"
        )
    return "\n".join(lines)


def render_header(result: StudyResult) -> str:
    """Render the run's provenance: seed, window, folds."""
    cfg = result.config
    n_scored = result.scored_end - result.scored_start
    return "\n".join(
        [
            "honest-backtest - degradation study",
            f"  seed                {cfg.seed}",
            f"  bars generated      {cfg.synthetic.n_bars}",
            f"  scored window       bars [{result.scored_start}, {result.scored_end}) "
            f"= {n_scored} bars "
            f"({n_scored / cfg.synthetic.bars_per_year:.1f} years)",
            f"  walk-forward folds  {len(result.folds)} "
            f"(train {cfg.train_size}, test {cfg.test_size}, non-overlapping)",
            f"  injected edge       alpha={cfg.synthetic.signal_alpha}, "
            f"persistence={cfg.synthetic.signal_persistence}",
        ]
    )


def render_context(result: StudyResult) -> str:
    """Render the reference points the headline number should be read against."""
    naive = result.stages[0].metrics.sharpe
    honest = result.honest_sharpe
    return "\n".join(
        [
            "Reference points:",
            f"  Oracle (knows the latent state, zero costs)   Sharpe {result.oracle_sharpe:6.2f}"
            "   <- the edge that was injected",
            f"  Buy and hold, same window, same costs         Sharpe {result.buy_hold_sharpe:6.2f}",
            f"  Naive backtest (stage 1)                      Sharpe {naive:6.2f}"
            "   <- what a careless backtest would report",
            f"  Honest out-of-sample (stage 5)                Sharpe {honest:6.2f}"
            "   <- the only number worth quoting",
            "",
            f"  Sharpe destroyed by removing assumptions: {naive - honest:.2f} "
            f"({(1 - honest / naive) * 100:.0f}% of the naive figure)"
            if naive != 0
            else "",
            f"  Sharpe fell monotonically at every stage: "
            f"{'yes' if result.is_monotone else 'NO - reported as measured'}",
        ]
    )


def render_sweep(sweep: SeedSweep) -> str:
    """Render the multi-seed robustness check."""
    lines = [
        f"Robustness across {len(sweep.seeds)} independent price paths "
        "(mean Sharpe +/- stdev, and t-stat against zero):",
        "",
        f"  {'Stage':<24} {'mean':>7} {'stdev':>7} {'t':>7}",
        f"  {'-' * 24} {'-' * 7} {'-' * 7} {'-' * 7}",
    ]
    for i, name in enumerate(sweep.stage_names):
        m, sd, t = sweep.summary(i)
        lines.append(f"  {name:<24} {m:>7.2f} {sd:>7.2f} {t:>7.2f}")
    lines.append("")
    m, sd, t = sweep.summary(len(sweep.stage_names) - 1)
    verdict = (
        "distinguishable from zero at the 5% level"
        if abs(t) > 2.0
        else "NOT distinguishable from zero"
    )
    lines.append(f"  Verdict on the honest stage: {verdict} (|t| = {abs(t):.2f}).")
    if sweep.monotone_flags:
        n = len(sweep.monotone_flags)
        lines.append(
            f"  Sharpe fell at every rung on {sweep.n_monotone}/{n} paths. "
            "The degradation is a strong tendency, not a theorem -- the "
            "exceptions are reported, not dropped."
        )
    return "\n".join(lines)


def render_full_report(result: StudyResult, sweep: SeedSweep | None = None) -> str:
    """Render everything, in the order a reader should meet it."""
    parts = [
        render_header(result),
        "",
        render_table(result),
        "",
        render_costs(result),
        "",
        render_context(result),
    ]
    if sweep is not None:
        parts += ["", render_sweep(sweep)]
    return "\n".join(parts)
