"""The documented entry point must actually work.

A reviewer's first action is to run one command. If that command is broken,
nothing else in the repository matters, so it is smoke-tested here with a
deliberately tiny configuration.
"""

from __future__ import annotations

import pytest

from honest_backtest.cli import DEFAULT_SEED, build_parser, main

TINY = ["--bars", "800", "--train-size", "252", "--test-size", "252", "--seeds", "1"]


def test_defaults_are_the_documented_ones():
    args = build_parser().parse_args([])
    assert args.seed == DEFAULT_SEED
    assert args.bars == 2520
    assert args.train_size == 504
    assert args.test_size == 252
    assert args.seeds == 12


def test_the_headline_command_runs_and_prints_a_table(capsys):
    assert main(TINY) == 0
    out = capsys.readouterr().out
    for expected in (
        "Naive backtest",
        "+ point-in-time data",
        "+ slippage",
        "+ commissions",
        "+ walk-forward OOS",
        "Sharpe",
        "Oracle",
    ):
        assert expected in out, f"missing {expected!r} from the report"


def test_the_report_says_the_data_is_synthetic(capsys):
    """The disclaimer is part of the output, not just the README."""
    main(TINY)
    out = capsys.readouterr().out
    assert "synthetic" in out.lower()
    assert "not a claim" in out.lower()


def test_the_seed_sweep_runs_when_asked(capsys):
    assert main(TINY[:-2] + ["--seeds", "2"]) == 0
    out = capsys.readouterr().out
    assert "Robustness across 2 independent price paths" in out
    assert "Verdict on the honest stage" in out


def test_markdown_output_is_a_markdown_table(capsys):
    main(TINY + ["--markdown"])
    out = capsys.readouterr().out
    assert "| # | Stage | Sharpe |" in out


def test_an_impossible_configuration_fails_loudly():
    with pytest.raises(ValueError, match="too short"):
        main(["--bars", "300", "--train-size", "504", "--seeds", "1"])


def _without_timing(out: str) -> list[str]:
    return [
        line
        for line in out.splitlines()
        if not line.startswith(("Completed in", "Events processed", "Stage 5 ran"))
    ]


def test_fast_and_parallel_flags_print_the_same_report(capsys):
    """--fast and --workers change how long the study takes, not what it says."""
    args = TINY[:-2] + ["--seeds", "2"]
    main(args)
    slow = capsys.readouterr().out
    main(args + ["--fast", "--workers", "2"])
    fast = capsys.readouterr().out
    assert _without_timing(fast) == _without_timing(slow)
    assert "fast path" in fast
