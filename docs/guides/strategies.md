# Writing a strategy

A strategy file defines `SPEC`, a `StrategySpec` that names a builder, the
parameter grid the in-sample fit may search, and the warm-up the strategy
needs before it can trade:

```python
--8<-- "examples/sma_crossover.py:15:59"
```

## The rules

- **Read prices only through `self.data`.** `latest_closes`,
  `get_latest_bars` and `current_bar` return bars up to the current one and no
  further. Asking for the future raises `LookAheadError`.
- **Emit target weights.** Put a `SignalEvent(symbol, timestamp, weight)` on
  the queue; the portfolio turns it into an order for the difference from what
  is held, inside a no-trade band (5% of equity by default).
- **Build fresh state in `__init__`.** The builder is called once per
  backtest, so nothing carries over between runs or walk-forward folds.
- **Every grid setting is a test.** A bigger grid fits more noise, and the
  Deflated Sharpe Ratio on the in-sample rungs charges for it.
- **`warmup` is the history the strategy needs.** Walk-forward folds replay
  that many bars before each test window so trailing statistics are warm.

## Referring to a strategy

`--strategy file.py` uses the file's `SPEC`; `--strategy file.py:NAME` uses a
different attribute (a spec, or a function returning one); `--strategy
momentum` audits the built-in time-series momentum strategy.

## Running one setting directly

```python
from honest_backtest import load_csv_bars, run_backtest
from honest_backtest.audit import load_spec

bars = load_csv_bars("prices.csv")
spec = load_spec("my_strategy.py")
result = run_backtest(bars, spec.factory({"fast": 20, "slow": 100}),
                      fill_timing="next_open")
print(result.metrics())
```

Several instruments run on one shared cursor when `run_backtest` is given a
dict of bars per symbol; see `examples/cross_sectional_momentum.py` in the
repository.
