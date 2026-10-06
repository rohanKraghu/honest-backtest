# Paper trading

The strategy you audited runs unchanged on bars as they arrive. The paper
trader is built from the same portfolio, execution handler and cost models as
a backtest and driven by the engine's own loop, so a replayed feed reproduces a
backtest's equity curve and trade blotter exactly; the test suite asserts it.
Orders are simulated and never leave the process.

## From the command line

`honest-backtest paper` follows a CSV file that something else appends to, such
as a scheduled download, and trades each new row as it appears:

```bash
honest-backtest paper --data prices.csv --strategy my_strategy.py \
    --params fast=20,slow=100 --journal paper.jsonl
```

- **Warm-up.** Rows already in the file are replayed through the strategy with
  every signal thrown away, so its indicators and internal state are exactly
  what they would be at that point in a backtest, and nothing trades on the
  past.
- **Fills.** Orders fill on the next bar, at its open when the file has real
  opens and at its close otherwise, because a bar's close has already printed
  by the time the bar is complete. Spread, impact and commission are the
  audit's, calibrated on the history.
- **The setting.** `--params` fixes it. Without it the setting with the best
  in-sample Sharpe on the history is used, and the header says so, since that
  is an optimistic choice.
- **Trying it out.** `--live-from DATE` treats rows from that date on as live,
  which replays the end of a file as if it were arriving now, and `--once`
  stops when there are no new rows instead of waiting.

```bash
honest-backtest paper --data examples/sample_prices.csv \
    --strategy examples/sma_crossover.py --live-from 2024-06-01 --once
```

## The journal and restarts

Every bar is appended to the journal as one JSON line, flushed immediately:
cash, position, equity, the bar's fills, orders still working and running
costs. After a restart, `--resume` restores the book from the last line and
keeps the setting and cost calibration the run began with, so a resumed run
writes the same journal, line for line, as one that never stopped. Resuming
with a different `--params` is refused, as is overwriting an existing journal
without `--resume`.

A row that changes after it was traded on stops the run with `FeedError`:
accepting a revised past would make the record unreproducible. For the same
reason adjusted closes are off unless you pass `--adjusted`, since an
adjustment factor rescales history whenever a dividend is paid.

## In code

`PaperTrader` takes any `Feed`:

- `ReplayFeed(bars, pace=0.0)` replays a list of bars.
- `CsvTailFeed(path)` follows a growing CSV file.
- `PollingFeed(fetch, interval=60)` calls a function you write, typically
  one that asks a broker or data vendor for its latest completed bars, and
  emits each bar newer than the last. Returning the same bar again, or the
  last few bars every time, is fine.

```python
from honest_backtest import PaperTrader, PollingFeed
from honest_backtest.audit import load_spec

def latest_bars():
    ...  # return completed Bars with .time set, or None

spec = load_spec("my_strategy.py")
trader = PaperTrader(
    PollingFeed(latest_bars, interval=60),
    spec.factory({"fast": 20, "slow": 100}),
    history=history_bars,
    symbol="SPY",
    journal="paper.jsonl",
)
trader.run(on_bar=lambda state: print(state.bar.time, state.equity))
```

## Limits

It trades one instrument. Fills are simulated, so queue position, partial
fills beyond the volume cap and broker rejections are not modelled. A
permanent-impact model's decaying push is not carried across a restart.
