"""``honest-backtest paper`` from the command line: run, journal, resume."""

from __future__ import annotations

from pathlib import Path

import pytest

from honest_backtest.cli import main
from honest_backtest.experiments import momentum_spec
from honest_backtest.live import read_journal
from honest_backtest.paper_cli import parse_params

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SAMPLE = (EXAMPLES / "sample_prices.csv").read_text().splitlines(keepends=True)
STRATEGY = str(EXAMPLES / "sma_crossover.py")


def _args(data, journal, *extra):
    return [
        "paper",
        "--data",
        str(data),
        "--strategy",
        STRATEGY,
        "--params",
        "fast=20,slow=100",
        "--journal",
        str(journal),
        "--once",
        "--symbol",
        "TEST",
        *extra,
    ]


def test_a_stretch_of_history_replays_as_live_bars(tmp_path, capsys):
    data, journal = tmp_path / "prices.csv", tmp_path / "paper.jsonl"
    data.write_text("".join(SAMPLE))
    assert main(_args(data, journal, "--live-from", "2024-06-01")) == 0
    out = capsys.readouterr().out
    rows = read_journal(journal)
    live = [line for line in SAMPLE[1:] if line >= "2024-06-01"]
    assert len(rows) == 1 + len(live)
    assert rows[1]["time"].startswith("2024-06-03")
    assert f"Live bars          {len(live)}" in out


def test_an_existing_journal_is_not_overwritten(tmp_path):
    data, journal = tmp_path / "prices.csv", tmp_path / "paper.jsonl"
    data.write_text("".join(SAMPLE))
    journal.write_text("keep me\n")
    with pytest.raises(SystemExit, match="--resume"):
        main(_args(data, journal, "--live-from", "2024-06-01"))
    assert journal.read_text() == "keep me\n"


def test_a_restart_with_resume_gives_the_uninterrupted_journal(tmp_path, capsys):
    whole_data, whole = tmp_path / "whole.csv", tmp_path / "whole.jsonl"
    whole_data.write_text("".join(SAMPLE))
    main(_args(whole_data, whole, "--live-from", "2024-03-01"))

    # The same days, but the file stops growing in May and the trader restarts.
    data, journal = tmp_path / "prices.csv", tmp_path / "paper.jsonl"
    cut = next(i for i, line in enumerate(SAMPLE) if i and line >= "2024-05-15")
    data.write_text("".join(SAMPLE[:cut]))
    main(_args(data, journal, "--live-from", "2024-03-01"))
    with data.open("a") as handle:
        handle.write("".join(SAMPLE[cut:]))
    main(_args(data, journal, "--resume"))
    capsys.readouterr()
    assert read_journal(journal) == read_journal(whole)


def test_params_must_name_the_grid_parameters():
    spec = momentum_spec()
    assert parse_params("lookback=20", spec) == {"lookback": 20}
    with pytest.raises(ValueError, match="takes lookback"):
        parse_params("fast=20", spec)
    with pytest.raises(ValueError, match="name=value"):
        parse_params("20", spec)


def test_resuming_with_a_different_setting_is_refused(tmp_path, capsys):
    data, journal = tmp_path / "prices.csv", tmp_path / "paper.jsonl"
    data.write_text("".join(SAMPLE))
    main(_args(data, journal, "--live-from", "2024-08-01"))
    argv = _args(data, journal, "--resume")
    argv[argv.index("fast=20,slow=100")] = "fast=50,slow=100"
    with pytest.raises(SystemExit, match="different setting"):
        main(argv)
