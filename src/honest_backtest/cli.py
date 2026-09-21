"""Command-line interface for the degradation study.

The implementation lives inside the package so that both entry points -- the
``run_experiment.py`` shim at the repository root and the ``honest-backtest``
console script created by ``pip install .`` -- run exactly the same code.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

from .experiments import StudyConfig, run_seed_sweep, run_study
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


def main(argv: list[str] | None = None) -> int:
    """Run the study and print the report.

    Args:
        argv: Command-line arguments; ``None`` means ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
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
