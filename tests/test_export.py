"""The JSON export must say what the text report says, and nothing different.

A machine-readable result is only trustworthy if it is the same result: the
tests read the JSON back and find every number in the printed report, and
check that a config file and the equivalent flags write identical JSON.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import date, timedelta

import numpy as np
import pytest

from honest_backtest.cli import main
from honest_backtest.export import _plain, to_json
from honest_backtest.report import _fmt_params, _fmt_probability
from honest_backtest.synthetic import SyntheticConfig, generate_price_series

TINY = ["--bars", "800", "--train-size", "252", "--test-size", "252", "--seeds", "2"]


def _strict(text: str):
    """Parse JSON, refusing the NaN and Infinity that Python would accept."""

    def refuse(name):
        raise ValueError(f"non-standard JSON constant {name}")

    return json.loads(text, parse_constant=refuse)


@pytest.fixture(scope="module")
def price_csv(tmp_path_factory):
    """Four years of dated synthetic prices written as a CSV."""
    series = generate_price_series(SyntheticConfig(n_bars=1008), seed=3)
    path = tmp_path_factory.mktemp("data") / "prices.csv"
    day = date(2020, 1, 1)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Date", "Close", "Volume"])
        for close, volume in zip(series.closes, series.volumes, strict=True):
            writer.writerow([day.isoformat(), close, volume])
            day += timedelta(days=1)
    return path


def _table_rows(out: str) -> dict[int, list[str]]:
    """The printed ladder's rows, split into cells, keyed by rung number."""
    rows = {}
    for line in out.splitlines():
        cells = re.split(r"\s{2,}", line.strip())
        if len(cells) == 9 and cells[0].isdigit():
            rows[int(cells[0])] = cells
    return rows


def _check_ladder_against_text(ladder: dict, out: str) -> None:
    rows = _table_rows(out)
    assert sorted(rows) == [s["index"] for s in ladder["stages"]]
    for stage in ladder["stages"]:
        m = stage["metrics"]
        p_edge = float("nan") if stage["p_edge"] is None else stage["p_edge"]
        assert rows[stage["index"]] == [
            str(stage["index"]),
            stage["name"],
            f"{m['sharpe']:.2f}",
            _fmt_probability(p_edge),
            f"{m['total_return'] * 100:+.1f}%",
            f"{m['annual_return'] * 100:+.1f}%",
            f"{m['max_drawdown'] * 100:.1f}%",
            str(m["n_trades"]),
            _fmt_params(stage["chosen_params"]),
        ]
        assert f"slippage {m['slippage_cost']:>12,.0f}" in out
    assert f"Sharpe {ladder['buy_and_hold']['sharpe']:6.2f}" in out


def test_study_json_matches_the_printed_report(tmp_path, capsys):
    """Every rung, the oracle, buy and hold and the sweep agree with the text."""
    path = tmp_path / "study.json"
    assert main(TINY + ["--json", str(path)]) == 0
    out = capsys.readouterr().out
    doc = _strict(path.read_text())
    assert doc["kind"] == "study" and doc["schema_version"] == 1
    assert doc["settings"]["synthetic"]["n_bars"] == 800
    assert doc["settings"]["train_size"] == 252
    _check_ladder_against_text(doc["ladder"], out)
    assert f"Sharpe {doc['oracle_sharpe']:6.2f}" in out
    assert len(doc["ladder"]["folds"]) == 2
    sweep = doc["sweep"]
    assert len(sweep["seeds"]) == 2
    for stage in sweep["stages"]:
        assert (
            f"{stage['name']:<24} {stage['mean']:>7.2f} {stage['stdev']:>7.2f} "
            f"{stage['t']:>7.2f}"
        ) in out
    assert f"JSON results written to {path}" in out


def test_audit_json_matches_the_printed_report(price_csv, tmp_path, capsys):
    """Ladder, costs, leak check, t-stat and the CPCV spread agree with the text."""
    path = tmp_path / "audit.json"
    argv = [
        "audit",
        "--data",
        str(price_csv),
        "--strategy",
        "momentum",
        "--train-size",
        "252",
        "--test-size",
        "252",
        "--commission-bps",
        "1.5",
        "--cpcv",
        "4,2",
        "--cpcv-purge",
        "20",
        "--json",
        str(path),
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    doc = _strict(path.read_text())
    assert doc["kind"] == "audit"
    assert doc["strategy"]["name"] == "time-series momentum"
    assert doc["settings"]["fill_timing"] == "next_close"
    assert doc["costs"]["commission"] == {"model": "percent of notional", "bps": 1.5}
    assert doc["data"]["first"] == "2020-01-01"
    _check_ladder_against_text(doc["ladder"], out)
    assert doc["leak_check"]["ran"] and doc["leak_check"]["passed"]
    assert doc["leak_check"]["summary"] in out
    assert f"t-stat ~ {doc['honest_t_stat']:.2f}" in out
    cpcv = doc["cpcv"]
    assert len(cpcv["path_sharpes"]) == 3 and len(cpcv["splits"]) == 6
    assert f"median {cpcv['median']:6.2f}" in out
    q1, q3 = cpcv["quartiles"]
    assert f"quartiles {q1:.2f} to {q3:.2f}" in out
    assert f"Paths below zero: {cpcv['n_below_zero']} of 3" in out


def test_a_config_writes_the_same_json_as_the_flags(tmp_path, capsys):
    """Study: settings by file or by flag give byte-identical results."""
    by_flags = tmp_path / "flags.json"
    main(TINY + ["--seed", "11", "--json", str(by_flags)])
    config = tmp_path / "study.json"
    config.write_text(
        json.dumps(
            {"bars": 800, "train_size": 252, "test_size": 252, "seeds": 2, "seed": 11}
        )
    )
    by_file = tmp_path / "file.json"
    main(["--config", str(config), "--json", str(by_file)])
    capsys.readouterr()
    assert by_file.read_text() == by_flags.read_text()


def test_an_audit_config_writes_the_same_json_as_the_flags(price_csv, tmp_path, capsys):
    """Audit: including a JSON path set in the file, relative to the file."""
    flags = [
        "--data",
        str(price_csv),
        "--strategy",
        "momentum",
        "--train-size",
        "252",
        "--test-size",
        "252",
        "--no-leak-check",
        "--cpcv",
        "4,2",
        "--cpcv-purge",
        "20",
        "--cpcv-embargo",
        "3",
    ]
    by_flags = tmp_path / "flags.json"
    main(["audit", *flags, "--json", str(by_flags)])
    config = tmp_path / "audit.yaml"
    config.write_text(
        f"data: {price_csv}\nstrategy: momentum\ntrain_size: 252\ntest_size: 252\n"
        "no_leak_check: true\ncpcv: [4, 2]\ncpcv_purge: 20\ncpcv_embargo: 3\n"
        "json: from_file.json\n"
    )
    main(["audit", "--config", str(config)])
    capsys.readouterr()
    assert (tmp_path / "from_file.json").read_text() == by_flags.read_text()


def test_undefined_numbers_become_null_not_nan():
    """``NaN`` is not JSON; strict readers would refuse the whole file."""
    doc = {"p": float("nan"), "inf": float("inf"), "n": np.int64(3), "a": np.array([1.5])}
    assert _plain(doc) == {"p": None, "inf": None, "n": 3, "a": [1.5]}
    assert _strict(to_json(doc)) == {"p": None, "inf": None, "n": 3, "a": [1.5]}
