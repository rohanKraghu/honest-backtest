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

from .audit import AuditConfig, load_spec, render_audit_report, run_audit
from .csvdata import load_csv_bars
from .experiments import LadderSettings, StudyConfig, run_seed_sweep, run_study
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
    parser.add_argument("--markdown", action="store_true", help="also print Markdown")
    return parser


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
    config = AuditConfig(
        settings=LadderSettings(
            train_size=args.train_size,
            test_size=args.test_size,
            initial_capital=args.capital,
            rebalance_threshold=args.rebalance_band,
            bars_per_year=args.bars_per_year,
        ),
        half_spread_bps=args.half_spread_bps,
        impact_coefficient=args.impact,
        commission_bps=args.commission_bps,
        commission_per_share=args.commission_per_share,
    )
    started = time.perf_counter()
    result = run_audit(bars, spec, config)
    print(render_audit_report(result, markdown=args.markdown))
    print()
    print(f"Completed in {time.perf_counter() - started:.1f}s.")
    return 0


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

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
