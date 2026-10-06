"""Run a cross-sectional momentum book over several instruments.

Usage::

    python examples/cross_sectional_momentum.py

Four synthetic instruments, ranked every month on their trailing three-month
return; the strongest is held long and the weakest short. The same run is
repeated with each friction switched on in turn, so the cost of each is
visible. The prices are synthetic, so the numbers demonstrate the machinery,
not a strategy.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from honest_backtest import (  # noqa: E402
    CrossSectionalMomentumStrategy,
    Financing,
    PermanentImpactSlippage,
    PerShareCommission,
    SyntheticConfig,
    bars_from_series,
    generate_price_series,
    run_backtest,
)


def main() -> None:
    """Print the book's Sharpe as each friction is added."""
    panel = {}
    for seed in (11, 12, 13, 14):
        series = generate_price_series(SyntheticConfig(n_bars=1260), seed=seed)
        panel[f"S{seed}"] = bars_from_series(
            series.closes, symbol=f"S{seed}", volumes=series.volumes
        )

    def strategy(events, data):
        return CrossSectionalMomentumStrategy(
            events, data, lookback=63, top_n=1, rebalance_every=21, long_short=True
        )

    runs = [
        ("No costs", {}),
        ("+ spread, impact, permanent impact", {"slippage": PermanentImpactSlippage()}),
        (
            "+ commissions",
            {"slippage": PermanentImpactSlippage(), "commission": PerShareCommission()},
        ),
        (
            "+ volume cap (0.2% of each bar)",
            {
                "slippage": PermanentImpactSlippage(),
                "commission": PerShareCommission(),
                "max_participation": 0.002,
            },
        ),
        (
            "+ 4% short fee and 6% margin rate",
            {
                "slippage": PermanentImpactSlippage(),
                "commission": PerShareCommission(),
                "max_participation": 0.002,
                # No interest on cash: Sharpe here is measured against zero, so
                # a cash rate would raise it without any skill.
                "financing": Financing(borrow_rate=0.06, short_fee=0.04),
            },
        ),
    ]
    print(f"{'Run':<38} {'Sharpe':>7} {'Return':>9} {'Financing':>11} {'Unfilled':>10}")
    for name, kwargs in runs:
        result = run_backtest(panel, strategy, warmup=64, **kwargs)
        m = result.metrics()
        print(
            f"{name:<38} {m.sharpe:7.2f} {m.total_return * 100:+8.1f}% "
            f"{result.total_financing:11,.0f} {result.unfilled_quantity:10,.0f}"
        )


if __name__ == "__main__":
    main()
