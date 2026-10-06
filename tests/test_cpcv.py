"""Running a strategy through combinatorial purged cross-validation.

The splitter's guarantees (see ``test_cpcv_splits.py``) are only worth
something if the runner respects them: every training run must replay only
its own training segment, every test run must replay only the group it
scores plus history strictly before it, and the paths must stitch back into
the whole window.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from honest_backtest import experiments
from honest_backtest.audit import (
    AuditConfig,
    render_audit_html,
    render_audit_report,
    run_audit,
)
from honest_backtest.cli import main
from honest_backtest.commission import ZeroCommission
from honest_backtest.csvdata import load_csv_bars
from honest_backtest.experiments import (
    CPCVResult,
    LadderSettings,
    fit_on_segments,
    momentum_spec,
    run_cpcv,
)
from honest_backtest.financing import Financing
from honest_backtest.frictions import MarketFrictions
from honest_backtest.metrics import annualised_sharpe
from honest_backtest.slippage import FixedBpsSlippage
from honest_backtest.spec import StrategySpec
from honest_backtest.synthetic import SyntheticConfig, generate_price_series
from honest_backtest.walkforward import CombinatorialPurgedSplitter

SETTINGS = LadderSettings(train_size=252, test_size=252)
COSTS = {"slippage": FixedBpsSlippage(2.0), "commission": ZeroCommission()}


@pytest.fixture(scope="module")
def bars():
    """A short synthetic path, long enough for six groups after warm-up."""
    return generate_price_series(SyntheticConfig(n_bars=900), seed=7).to_bars()


@pytest.fixture(scope="module")
def spec():
    """Momentum with a two-setting grid, to keep the runs quick."""
    full = momentum_spec((10, 30))
    return StrategySpec(name=full.name, build=full.build, grid=full.grid, warmup=60)


@pytest.fixture(scope="module")
def result(bars, spec) -> CPCVResult:
    """One cross-validation run shared by the read-only tests."""
    splitter = CombinatorialPurgedSplitter(5, 2, purge=spec.warmup, embargo=10)
    return run_cpcv(bars, spec, SETTINGS, splitter, **COSTS)


def test_every_path_covers_the_whole_window(result):
    """Each path is a complete out-of-sample walk through the window."""
    assert result.n_paths == 4 == len(result.path_sharpes)
    for returns in result.path_returns:
        assert returns.shape == (result.end - result.start,)
    assert result.start == 60 and result.end == 900


def test_path_sharpes_are_the_sharpes_of_the_path_returns(result):
    """The distribution reported is computed from the stitched paths."""
    expected = [annualised_sharpe(r) for r in result.path_returns]
    np.testing.assert_allclose(result.path_sharpes, expected)
    q1, q3 = result.quartiles
    assert q1 <= result.median <= q3
    assert result.share_below_zero == np.mean(result.path_sharpes < 0)


def test_a_group_is_scored_with_the_setting_its_split_chose(result):
    """Two paths that took the same setting for a group report the same returns."""
    lengths = [b - a for a, b in result.groups]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    for g in range(result.n_groups):
        by_setting: dict[str, np.ndarray] = {}
        for path, returns in zip(result.paths, result.path_returns, strict=True):
            setting = repr(result.chosen_params[path[g][1]])
            chunk = returns[offsets[g] : offsets[g + 1]]
            if setting in by_setting:
                np.testing.assert_array_equal(by_setting[setting], chunk)
            by_setting[setting] = chunk


def test_runs_replay_only_allowed_bars(bars, spec, monkeypatch):
    """Training replays its own segments; testing replays a group and its past."""
    calls: list[tuple[int, int, int]] = []
    original = experiments._run

    def recording_run(run_bars, *args, **kwargs):
        calls.append(
            (run_bars[0].timestamp, run_bars[-1].timestamp + 1, kwargs["warmup"])
        )
        return original(run_bars, *args, **kwargs)

    monkeypatch.setattr(experiments, "_run", recording_run)
    splitter = CombinatorialPurgedSplitter(5, 2, purge=spec.warmup, embargo=10)
    result = run_cpcv(bars, spec, SETTINGS, splitter, **COSTS)

    expected_training = [
        (a, b, spec.warmup)
        for split in result.splits
        for _ in spec.grid
        for a, b in split.train_segments
        if b - a >= spec.warmup + 2
    ]
    training, testing = calls[: len(expected_training)], calls[len(expected_training) :]
    assert training == expected_training
    for split in result.splits:
        for a, b in split.train_segments:
            for t_start, t_end in split.test_blocks:
                assert b <= t_start - splitter.purge or a >= t_end + splitter.purge

    allowed_tests = {(a - spec.warmup, b, spec.warmup) for a, b in result.groups}
    assert set(testing) <= allowed_tests
    assert set(testing) == allowed_tests, "every group is scored"
    assert len(testing) <= len(result.groups) * len(spec.grid), "a pair ran twice"


def test_training_ignores_every_bar_outside_its_segments(bars, spec):
    """Rewriting the test, purge and embargo bars cannot change the fit."""
    splitter = CombinatorialPurgedSplitter(5, 2, purge=spec.warmup, embargo=10)
    rng = np.random.default_rng(0)
    for split in splitter.split(len(bars), spec.warmup):
        train = set(split.train_indices())
        altered = [
            b if i in train else _scaled(b, rng.uniform(0.6, 1.4))
            for i, b in enumerate(bars)
        ]
        trials, trials_altered = [], []
        fit_on_segments(
            bars, split.train_segments, spec, SETTINGS, trial_sharpes=trials, **COSTS
        )
        fit_on_segments(
            altered,
            split.train_segments,
            spec,
            SETTINGS,
            trial_sharpes=trials_altered,
            **COSTS,
        )
        assert trials == trials_altered


def test_test_scores_ignore_the_future_and_the_distant_past(bars, spec, result):
    """A group's returns depend only on the group and its warm-up history."""
    rng = np.random.default_rng(1)
    g_start, g_end = result.groups[2]
    altered = [
        b if g_start - spec.warmup <= i < g_end else _scaled(b, rng.uniform(0.6, 1.4))
        for i, b in enumerate(bars)
    ]
    params = result.chosen_params[result.paths[0][2][1]]
    window = slice(g_start - spec.warmup, g_end)
    original = experiments._run(
        bars[window], spec, params, SETTINGS, warmup=spec.warmup, **COSTS
    ).returns()
    rerun = experiments._run(
        altered[window], spec, params, SETTINGS, warmup=spec.warmup, **COSTS
    ).returns()
    offset = sum(b - a for a, b in result.groups[:2])
    np.testing.assert_array_equal(original, rerun)
    np.testing.assert_array_equal(
        result.path_returns[0][offset : offset + g_end - g_start], original
    )


def test_cross_validation_is_deterministic(bars, spec, result):
    """The same inputs give the same distribution to the last digit."""
    splitter = CombinatorialPurgedSplitter(5, 2, purge=spec.warmup, embargo=10)
    again = run_cpcv(bars, spec, SETTINGS, splitter, **COSTS)
    np.testing.assert_array_equal(again.path_sharpes, result.path_sharpes)


def test_the_fast_path_gives_the_same_distribution(bars, spec, result):
    """``fast`` changes how the runs are computed, never what they return."""
    splitter = CombinatorialPurgedSplitter(5, 2, purge=spec.warmup, embargo=10)
    fast = run_cpcv(bars, spec, replace(SETTINGS, fast=True), splitter, **COSTS)
    assert fast.chosen_params == result.chosen_params
    for got, want in zip(fast.path_returns, result.path_returns, strict=True):
        np.testing.assert_array_equal(got, want)


def test_frictions_apply_as_they_do_on_the_walk_forward_rung(bars, spec, result):
    """The spread is printed beside the walk-forward Sharpe, so it pays the same costs."""
    frictions = MarketFrictions(financing=Financing(cash_rate=0.05))
    splitter = CombinatorialPurgedSplitter(5, 2, purge=spec.warmup, embargo=10)
    carried = run_cpcv(
        bars, spec, replace(SETTINGS, frictions=frictions), splitter, **COSTS
    )
    assert not np.array_equal(carried.path_sharpes, result.path_sharpes)


def test_the_window_needs_warm_up_history(bars, spec):
    """A first group with no history before it is refused, not scored cold."""
    splitter = CombinatorialPurgedSplitter(4, 1)
    with pytest.raises(ValueError, match="warm-up history"):
        run_cpcv(bars, spec, SETTINGS, splitter, start=10, **COSTS)
    with pytest.raises(ValueError, match="past the"):
        run_cpcv(bars, spec, SETTINGS, splitter, end=len(bars) + 1, **COSTS)


def test_segments_too_short_to_warm_up_are_refused(bars, spec):
    """If the purge eats every training segment, say so instead of fitting noise."""
    with pytest.raises(ValueError, match="no training segment"):
        fit_on_segments(bars, [(0, 30), (100, 140)], spec, SETTINGS, **COSTS)


def _scaled(bar, factor: float):
    return replace(
        bar,
        open=bar.open * factor,
        high=bar.high * factor,
        low=bar.low * factor,
        close=bar.close * factor,
    )


# --------------------------------------------------------------------------
# In the audit
# --------------------------------------------------------------------------


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


def test_the_audit_ladder_is_unchanged_by_cross_validation(price_csv, spec):
    """CPCV is an extra: every rung and reference keeps its exact number."""
    bars = load_csv_bars(price_csv)
    base = AuditConfig(settings=SETTINGS, check_leaks=False)
    plain = run_audit(bars, spec, base)
    extra = run_audit(bars, spec, replace(base, cpcv=(4, 2), cpcv_embargo=5))
    assert plain.cpcv is None and extra.cpcv is not None
    assert extra.ladder.stages == plain.ladder.stages
    assert extra.ladder.buy_hold_sharpe == plain.ladder.buy_hold_sharpe
    cpcv = extra.cpcv
    assert (cpcv.start, cpcv.end) == (plain.ladder.scored_start, plain.ladder.scored_end)
    assert (cpcv.purge, cpcv.embargo) == (spec.warmup, 5)
    report = render_audit_report(extra)
    assert render_audit_report(plain) in report
    assert "Combinatorial purged cross-validation (4 groups, 2 held out" in report
    assert f"median {cpcv.median:6.2f}" in report
    assert f"Paths below zero: {cpcv.n_below_zero} of 3" in report


def test_the_audit_command_takes_cpcv_flags(price_csv, tmp_path, capsys):
    """``--cpcv N,K`` prints the distribution after the ladder and charts it."""
    page = tmp_path / "audit.html"
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
        "--no-leak-check",
        "--cpcv",
        "5,2",
        "--cpcv-purge",
        "30",
        "--cpcv-embargo",
        "4",
        "--html",
        str(page),
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert out.index("+ walk-forward OOS") < out.index("Combinatorial purged")
    assert "purge 30, embargo 4 bars" in out
    assert "10 splits, 4 paths" in out
    html = page.read_text()
    assert "Out of sample, across 4 paths" in html
    assert html.count("<circle") == 4


def test_a_bad_cpcv_flag_is_refused(capsys):
    """``--cpcv`` needs two integers with K below N."""
    for bad in ("6", "2,2", "a,b"):
        with pytest.raises(SystemExit):
            main(["audit", "--data", "x.csv", "--strategy", "momentum", "--cpcv", bad])
    assert "N,K" in capsys.readouterr().err


def test_pages_without_cpcv_have_no_cpcv_section(price_csv, spec):
    """The HTML section appears only when cross-validation was run."""
    bars = load_csv_bars(price_csv)
    result = run_audit(bars, spec, AuditConfig(settings=SETTINGS, check_leaks=False))
    assert "across" not in render_audit_html(result).split("<h2>The ladder</h2>")[1]
