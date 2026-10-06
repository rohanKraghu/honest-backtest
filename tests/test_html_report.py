"""The HTML report: self-contained, complete, and safe with odd names."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from honest_backtest.audit import AuditConfig, render_audit_html, run_audit
from honest_backtest.cli import main
from honest_backtest.experiments import LadderSettings, StudyConfig, run_study
from honest_backtest.html_report import ReportContext, render_html, study_html
from honest_backtest.spec import StrategySpec, param_grid
from honest_backtest.strategy import BuyAndHoldStrategy
from honest_backtest.synthetic import SyntheticConfig, generate_price_series

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def study():
    """A short study, shared across tests."""
    return run_study(
        StudyConfig(
            synthetic=SyntheticConfig(n_bars=1008),
            seed=7,
            lookback_grid=(5, 20),
            train_size=252,
            test_size=252,
        )
    )


def test_every_rung_keeps_its_scored_returns(study):
    n = study.scored_end - study.scored_start
    assert all(len(s.returns) == n for s in study.stages)
    assert len(study.buy_hold_returns) == n


def test_the_page_is_self_contained(study):
    page = study_html(study)
    assert page.startswith("<!doctype html>")
    assert "<script" not in page
    # Nothing is fetched: no external URLs in src or href attributes.
    assert not re.search(r'(src|href)="https?://', page)
    assert page.count("<svg") == 3


def test_the_page_carries_every_rung_and_the_references(study):
    page = study_html(study)
    for stage in study.stages:
        assert stage.name in page
    assert f"{study.honest_sharpe:.2f}" in page
    assert "Oracle" in page and "Buy and hold" in page
    assert "prefers-color-scheme: dark" in page


def test_names_are_escaped(study):
    page = render_html(study, ReportContext(title="<b>x</b> & y"))
    assert "<b>x</b>" not in page
    assert "&lt;b&gt;x&lt;/b&gt; &amp; y" in page


def test_an_audit_page_carries_the_leak_verdict():
    bars = generate_price_series(SyntheticConfig(n_bars=600), seed=3).to_bars()
    spec = StrategySpec(
        name="hold",
        build=lambda events, data, symbol: BuyAndHoldStrategy(
            events, data, symbol=symbol
        ),
        grid=param_grid(),
    )
    result = run_audit(
        bars, spec, AuditConfig(settings=LadderSettings(train_size=150, test_size=150))
    )
    page = render_audit_html(result)
    assert "Audit: hold" in page
    assert "Look-ahead check" in page and "passed" in page


def test_the_command_line_writes_the_file(tmp_path):
    out = tmp_path / "audit.html"
    code = main(
        [
            "audit",
            "--data",
            str(EXAMPLES / "sample_prices.csv"),
            "--strategy",
            str(EXAMPLES / "sma_crossover.py"),
            "--html",
            str(out),
        ]
    )
    assert code == 0
    page = out.read_text(encoding="utf-8")
    assert "moving-average crossover" in page
    assert "2016-12-09" in page or "2016-12-08" in page
