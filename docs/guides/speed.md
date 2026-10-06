# Faster sweeps

The event loop costs speed: about 26 ms per thousand bars, two orders of
magnitude slower than a vectorised backtest. Most of a study's time, though,
goes on work it repeats: rerunning the same strategy with the same parameters
over the same bars while only the cost model changes.

`--fast` records each setting's signals once and replays them through the
portfolio's sizing rules and the same slippage and commission objects, in the
engine's order:

```bash
honest-backtest --fast --workers 4
honest-backtest audit --data prices.csv --strategy my_strategy.py --fast
```

The signals come from driving the real strategy through the real point-in-time
handler, so no strategy logic is rewritten and there is no second
implementation to drift. The test suite asserts that the results are equal to
the engine's, not merely close, for every fill timing and cost model.

Volume caps, limit orders and financing are not reproduced by the fast path, so
a rung that uses them runs on the engine.

`--workers` runs the seed sweep's independent price paths in separate
processes. Each path is deterministic, so the result does not depend on the
number of workers. On four cores the full study drops from about 36 seconds to
about 8.

In code, `LadderSettings(fast=True)` turns it on for `run_ladder` and the audit,
and `run_seed_sweep(..., fast=True, workers=4)` for the sweep.
