"""Command-line interface for the degradation study and for audits.

The implementation lives inside the package so that both entry points -- the
``run_experiment.py`` shim at the repository root and the ``honest-backtest``
console script created by ``pip install .`` -- run exactly the same code.

With no subcommand it runs the synthetic study from the README. With
``audit`` it runs the same ladder on a strategy and price file you supply::

    honest-backtest audit --data prices.csv --strategy my_strategy.py
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from .audit import (
    AuditConfig,
    load_spec,
    render_audit_html,
    render_audit_report,
    run_audit,
)
from .csvdata import has_real_opens, load_csv_bars
from .experiments import LadderSettings, StudyConfig, run_seed_sweep, run_study
from .financing import Financing
from .frictions import MarketFrictions
from .html_report import study_html
from .report import render_full_report, render_markdown_table
from .synthetic import SyntheticConfig

#: The headline seed. It is the date the project was written, fixed before any
#: results were seen; the multi-seed sweep exists so no claim rests on it.
DEFAULT_SEED = 20260921


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="run_experiment.py",
        description="Run the honest-backtest degradation study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="seed for the headline price path"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=12,
        help="number of independent paths for the robustness sweep (1 disables it)",
    )
    parser.add_argument(
        "--bars", type=int, default=2520, help="bars to generate (252 per year)"
    )
    parser.add_argument(
        "--train-size", type=int, default=504, help="walk-forward training window, bars"
    )
    parser.add_argument(
        "--test-size", type=int, default=252, help="walk-forward test window, bars"
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="also print the table in Markdown, for the README",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        metavar="PATH",
        help="also write a self-contained HTML report with charts to PATH",
    )
    return parser


def build_audit_parser() -> argparse.ArgumentParser:
    """Build the parser for ``honest-backtest audit``."""
    parser = argparse.ArgumentParser(
        prog="honest-backtest audit",
        description=(
            "Run the degradation ladder on your own strategy and price data: "
            "in-sample, then slippage, then commissions, then walk-forward."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", required=True, help="CSV with a date and close column, oldest first"
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="'momentum', a file defining SPEC, or file.py:NAME",
    )
    parser.add_argument("--symbol", default=None, help="instrument name for the report")
    parser.add_argument(
        "--no-adjust",
        action="store_true",
        help="ignore an adjusted-close column and trade raw prices",
    )
    parser.add_argument(
        "--default-volume",
        type=float,
        default=None,
        help="volume to assume if the file has none (feeds the impact model)",
    )
    parser.add_argument(
        "--train-size", type=int, default=504, help="training window, bars"
    )
    parser.add_argument("--test-size", type=int, default=252, help="test window, bars")
    parser.add_argument("--bars-per-year", type=int, default=252, help="annualisation")
    parser.add_argument(
        "--capital", type=float, default=1_000_000.0, help="starting cash"
    )
    parser.add_argument(
        "--rebalance-band", type=float, default=0.05, help="no-trade band, fraction"
    )
    parser.add_argument("--half-spread-bps", type=float, default=2.0, help="half-spread")
    parser.add_argument(
        "--impact", type=float, default=0.6, help="square-root impact coefficient"
    )
    parser.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        help="bps of notional per trade (default: per-share fee)",
    )
    parser.add_argument(
        "--commission-per-share", type=float, default=0.005, help="per-share fee"
    )
    parser.add_argument(
        "--fill",
        choices=("auto", "close", "next_open", "next_close"),
        default="auto",
        help=(
            "when the next-bar rung fills orders; auto uses next_open if the "
            "file has real opens, else next_close; close drops the rung"
        ),
    )
    parser.add_argument(
        "--permanent-impact",
        type=float,
        default=0.0,
        help="coefficient of decaying permanent impact (0 = temporary only)",
    )
    parser.add_argument(
        "--impact-half-life",
        type=float,
        default=5.0,
        help="bars for permanent impact to halve",
    )
    parser.add_argument(
        "--max-participation",
        type=float,
        default=None,
        help="cap each fill at this fraction of bar volume, e.g. 0.05",
    )
    parser.add_argument(
        "--limit-offset-bps",
        type=float,
        default=None,
        help="rebalance with limit orders this far inside the close",
    )
    parser.add_argument(
        "--limit-expiry", type=int, default=1, help="bars a limit order rests"
    )
    parser.add_argument(
        "--cash-rate", type=float, default=None, help="annual interest on cash"
    )
    parser.add_argument(
        "--borrow-rate", type=float, default=None, help="annual margin loan rate"
    )
    parser.add_argument(
        "--short-fee", type=float, default=None, help="annual short-borrow fee"
    )
    parser.add_argument(
        "--max-leverage", type=float, default=None, help="cap on gross leverage"
    )
    parser.add_argument(
        "--no-leak-check",
        action="store_true",
        help="skip replaying the strategy with altered futures to look for look-ahead",
    )
    parser.add_argument("--markdown", action="store_true", help="also print Markdown")
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        metavar="PATH",
        help="also write a self-contained HTML report with charts to PATH",
    )
    return parser


def _frictions(args: argparse.Namespace) -> MarketFrictions | None:
    """Liquidity, order-style and carry settings from audit flags, if any."""
    rates = (args.cash_rate, args.borrow_rate, args.short_fee, args.max_leverage)
    financing = None
    if any(r is not None for r in rates):
        financing = Financing(
            cash_rate=args.cash_rate or 0.0,
            borrow_rate=args.borrow_rate or 0.0,
            short_fee=args.short_fee or 0.0,
            max_leverage=args.max_leverage,
            bars_per_year=args.bars_per_year,
        )
    frictions = MarketFrictions(
        max_participation=args.max_participation,
        limit_offset_bps=args.limit_offset_bps,
        limit_expiry_bars=args.limit_expiry,
        financing=financing,
    )
    return frictions if frictions.active else None


def audit_main(argv: list[str]) -> int:
    """Run ``honest-backtest audit`` and print the report."""
    args = build_audit_parser().parse_args(argv)
    bars = load_csv_bars(
        args.data,
        symbol=args.symbol,
        use_adjusted=not args.no_adjust,
        default_volume=args.default_volume,
    )
    spec = load_spec(args.strategy)
    fill = args.fill
    if fill == "auto":
        fill = "next_open" if has_real_opens(bars) else "next_close"
    config = AuditConfig(
        settings=LadderSettings(
            train_size=args.train_size,
            test_size=args.test_size,
            initial_capital=args.capital,
            rebalance_threshold=args.rebalance_band,
            bars_per_year=args.bars_per_year,
            fill_timing=fill,
            frictions=_frictions(args),
        ),
        half_spread_bps=args.half_spread_bps,
        impact_coefficient=args.impact,
        commission_bps=args.commission_bps,
        commission_per_share=args.commission_per_share,
        check_leaks=not args.no_leak_check,
        permanent_impact=args.permanent_impact,
        impact_half_life=args.impact_half_life,
    )
    started = time.perf_counter()
    result = run_audit(bars, spec, config)
    print(render_audit_report(result, markdown=args.markdown))
    if args.html is not None:
        args.html.write_text(render_audit_html(result), encoding="utf-8")
        print(f"\nHTML report written to {args.html}")
    print()
    print(f"Completed in {time.perf_counter() - started:.1f}s.")
    # A leak makes every number above meaningless, so fail loudly for CI.
    return 1 if result.leaks is not None and result.leaks.findings else 0


def main(argv: list[str] | None = None) -> int:
    """Run the study (or an audit) and print the report.

    Args:
        argv: Command-line arguments; ``None`` means ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "audit":
        return audit_main(argv[1:])
    args = build_parser().parse_args(argv)

    config = StudyConfig(
        synthetic=SyntheticConfig(n_bars=args.bars),
        seed=args.seed,
        train_size=args.train_size,
        test_size=args.test_size,
    )

    started = time.perf_counter()
    result = run_study(config)
    sweep = None
    if args.seeds > 1:
        sweep = run_seed_sweep(args.seeds, replace(config, seed=args.seed))
    elapsed = time.perf_counter() - started

    print(render_full_report(result, sweep))
    print()
    print(
        "Events processed by the stage-5 event loop: "
        + ", ".join(f"{k} {v:,}" for k, v in sorted(result.n_events.items()))
    )
    print(f"Completed in {elapsed:.1f}s.")
    print()
    print(
        "Reminder: the price series is synthetic. These numbers demonstrate a\n"
        "methodology, they are not a claim that this strategy makes money."
    )

    if args.markdown:
        print()
        print(render_markdown_table(result))

    if args.html is not None:
        args.html.write_text(study_html(result), encoding="utf-8")
        print(f"\nHTML report written to {args.html}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
