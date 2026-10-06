"""Liquidity, carry and permanent impact reach the ladder and the audit command."""

from __future__ import annotations

from pathlib import Path

from honest_backtest.cli import main
from honest_backtest.commission import PerShareCommission
from honest_backtest.experiments import LadderSettings, momentum_spec, run_ladder
from honest_backtest.financing import Financing
from honest_backtest.frictions import MarketFrictions
from honest_backtest.slippage import FixedBpsSlippage
from honest_backtest.synthetic import SyntheticConfig, generate_price_series

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _ladder(frictions):
    bars = generate_price_series(SyntheticConfig(n_bars=1008), seed=7).to_bars()
    return run_ladder(
        bars,
        momentum_spec((5, 20)),
        LadderSettings(train_size=252, test_size=252, frictions=frictions),
        slippage=FixedBpsSlippage(2.0),
        commission=PerShareCommission(),
    )


def test_inactive_frictions_leave_the_ladder_unchanged():
    plain = _ladder(None)
    inactive = _ladder(MarketFrictions())
    assert [s.name for s in plain.stages] == [s.name for s in inactive.stages]
    assert [s.metrics for s in plain.stages] == [s.metrics for s in inactive.stages]


def test_active_frictions_add_one_rung_before_walk_forward():
    plain = _ladder(None)
    costly = _ladder(
        MarketFrictions(
            max_participation=0.0005,
            financing=Financing(borrow_rate=0.08, short_fee=0.05),
        )
    )
    names = [s.name for s in costly.stages]
    assert names[-2:] == ["+ liquidity and carry", "+ walk-forward OOS"]
    assert len(names) == len(plain.stages) + 1
    # The rungs before the new one are untouched.
    assert [s.metrics for s in costly.stages[:-2]] == [
        s.metrics for s in plain.stages[:-1]
    ]
    assert costly.stages[-2].metrics.sharpe < costly.stages[-3].metrics.sharpe


def test_the_audit_command_accepts_the_new_flags(capsys):
    code = main(
        [
            "audit",
            "--data",
            str(EXAMPLES / "sample_prices.csv"),
            "--strategy",
            str(EXAMPLES / "sma_crossover.py"),
            "--permanent-impact",
            "0.3",
            "--max-participation",
            "0.05",
            "--borrow-rate",
            "0.06",
            "--no-leak-check",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "+ liquidity and carry" in out
    assert "fills capped at 5.00% of volume" in out
