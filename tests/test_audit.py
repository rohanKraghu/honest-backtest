"""``honest-backtest audit`` must work on a stranger's strategy and data."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import pytest

from honest_backtest.audit import (
    AuditConfig,
    estimate_bar_volatility,
    load_spec,
    render_audit_report,
    run_audit,
)
from honest_backtest.cli import main
from honest_backtest.csvdata import load_csv_bars
from honest_backtest.experiments import LadderSettings
from honest_backtest.spec import StrategySpec
from honest_backtest.synthetic import SyntheticConfig, generate_price_series

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SMALL = LadderSettings(train_size=252, test_size=252)

STRATEGY_FILE = """
from honest_backtest import BuyAndHoldStrategy, StrategySpec, param_grid

SPEC = StrategySpec(
    name="file buy and hold",
    build=lambda events, data, symbol: BuyAndHoldStrategy(events, data, symbol=symbol),
    grid=param_grid(),
)

def make_other():
    return StrategySpec(name="other", build=SPEC.build, grid=SPEC.grid)

NOT_A_SPEC = 3
"""


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


@pytest.fixture
def strategy_file(tmp_path):
    """A user strategy file defining SPEC and a few other attributes."""
    path = tmp_path / "user_strategy.py"
    path.write_text(STRATEGY_FILE)
    return path


def test_built_in_names_resolve():
    spec = load_spec("momentum")
    assert spec.build_look_ahead is not None


def test_a_file_resolves_to_its_spec(strategy_file):
    assert load_spec(str(strategy_file)).name == "file buy and hold"


def test_a_named_factory_function_resolves(strategy_file):
    assert load_spec(f"{strategy_file}:make_other").name == "other"


def test_bad_references_fail_with_a_reason(strategy_file):
    with pytest.raises(ValueError, match="neither a file nor a built-in"):
        load_spec("no_such_strategy")
    with pytest.raises(ValueError, match="defines no 'MISSING'"):
        load_spec(f"{strategy_file}:MISSING")
    with pytest.raises(ValueError, match="not a StrategySpec"):
        load_spec(f"{strategy_file}:NOT_A_SPEC")


def test_cost_model_is_calibrated_before_the_scored_window(price_csv):
    """Changing every scored bar must not change the volatility the costs use."""
    bars = load_csv_bars(price_csv)
    config = AuditConfig(settings=SMALL)
    before = config.slippage(bars).bar_volatility
    doubled = bars[:252] + [
        b.__class__(**{**b.__dict__, "close": b.close * 2}) for b in bars[252:]
    ]
    assert config.slippage(doubled).bar_volatility == before
    assert before == pytest.approx(estimate_bar_volatility(bars[:252]))


def test_momentum_audit_keeps_the_look_ahead_rung(price_csv):
    bars = load_csv_bars(price_csv)
    result = run_audit(bars, load_spec("momentum"), AuditConfig(settings=SMALL))
    names = [s.name for s in result.ladder.stages]
    assert names[0] == "Naive backtest"
    assert names[-1] == "+ walk-forward OOS"
    assert result.years_scored == pytest.approx(3.0)


def test_report_names_dates_and_the_honest_number(price_csv):
    bars = load_csv_bars(price_csv)
    spec = load_spec("momentum")
    small_spec = StrategySpec(
        name=spec.name, build=spec.build, grid=spec.grid[:2], warmup=spec.warmup
    )
    report = render_audit_report(run_audit(bars, small_spec, AuditConfig(settings=SMALL)))
    assert "2020-01-01 to" in report
    assert "the only number worth quoting" in report
    assert "best of 2 settings" in report


def test_the_audit_command_runs_end_to_end(price_csv, strategy_file, capsys):
    argv = [
        "audit",
        "--data",
        str(price_csv),
        "--strategy",
        str(strategy_file),
        "--train-size",
        "252",
        "--test-size",
        "252",
        "--commission-bps",
        "1",
        "--markdown",
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "honest-backtest audit - file buy and hold" in out
    assert "In-sample, no costs" in out
    assert "+ walk-forward OOS" in out
    assert "| # | Stage | Sharpe |" in out


def test_the_shipped_example_audits(capsys):
    """The command printed in the README works from a fresh clone."""
    argv = [
        "audit",
        "--data",
        str(EXAMPLES / "sample_prices.csv"),
        "--strategy",
        str(EXAMPLES / "sma_crossover.py"),
    ]
    assert main(argv) == 0
    assert "moving-average crossover" in capsys.readouterr().out
