# honest-backtest

An event-driven backtesting framework built to demonstrate a negative result: that
a strategy which looks outstanding under the assumptions most backtests quietly
make is worthless once those assumptions are removed one at a time.

The headline output is not a return figure. It is the rate at which a return
figure decays when you stop lying to yourself.

---

## The problem

Most published backtests are wrong in the same few ways, and every one of those
ways biases the result upward:

| What the backtest assumes | What is actually true |
| --- | --- |
| The signal can be computed from data available at the time | A single mis-signed shift lets it read tomorrow's price |
| Normalisation statistics are known in advance | Mean and variance are estimated from the whole sample, including the future |
| Orders fill at the closing price | You cross a spread and move the market |
| Trading is free | Brokers invoice you |
| The best parameter is the one that fit best | The best-fitting parameter is partly fitted to noise |

None of these are exotic. They are what you get by default when you write a
backtest as a few lines of pandas. This repository removes them one at a time and
measures the damage.

## The headline result

Same strategy, same price path, same random seed, at every stage. Each row
removes exactly one comfortable assumption. **Every stage is scored over
identical bars** — bars 504 to 2520, eight years — so nothing here is a
consequence of measuring different periods.

| # | Stage | Sharpe | Total return | Ann. return | Max DD | Trades | Lookback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Naive backtest | **6.94** | +80925.7% | +131.0% | -3.1% | 2132 | 5 |
| 2 | + point-in-time data | 0.43 | +46.7% | +4.9% | -26.4% | 1665 | 20 |
| 3 | + slippage | 0.27 | +24.0% | +2.7% | -31.1% | 1664 | 20 |
| 4 | + commissions | 0.26 | +23.0% | +2.6% | -31.4% | 1665 | 20 |
| 5 | + walk-forward OOS | **-0.19** | -22.7% | -3.2% | -29.8% | 1789 | 5/5/45/30/20/20/5/5 |

Reference points from the same run:

```
Oracle (knows the latent state, zero costs)   Sharpe   1.30   <- the edge that was injected
Buy and hold, same window, same costs         Sharpe   0.09
Naive backtest (stage 1)                      Sharpe   6.94   <- what a careless backtest reports
Honest out-of-sample (stage 5)                Sharpe  -0.19   <- the only number worth quoting
```

**The honest answer is that this strategy does not work.** That is the finding.
Nothing was tuned to rescue it.

### Reading the table

- **Look-ahead bias alone accounts for 94% of the apparent edge.** Stage 1 to
  stage 2 is a fall from 6.94 to 0.43, and no money changed hands to cause it.
  Only the alignment of the signal changed.
- **Stage 1 is impossible, and that is the tell.** A cost-free trader who knows
  the hidden state exactly can only achieve Sharpe 1.30 on this data. Stage 1
  reports 6.94, five times the theoretical maximum, with a 3.1% maximum drawdown
  over eight years. When a backtest reports a number the data cannot support, the
  backtest is reading its own answer. Implausibly smooth equity curves are
  evidence of a bug, not of skill.
- **Commissions barely matter; slippage does.** Stage 3 to stage 4 costs only
  0.01 of Sharpe. That is not a bug in the fee model — measured over the run, a
  per-share commission of half a cent works out to **0.31bp of traded notional**
  against **5.74bp of spread-plus-impact slippage**, an 18x difference on $462M
  of notional traded. The explicit invoice is an order of magnitude smaller than
  the implicit cost. Backtests that model commissions and ignore slippage have
  modelled the smaller problem.
- **Walk-forward is what turns a marginal result into a negative one.** Stages 2
  to 4 choose the lookback that performed best over the very window they then
  report. Stage 5 re-fits on each training window and scores only the window that
  follows. The chosen lookback lurches between 5 and 45 across folds — the
  in-sample optimum is not stable, which is exactly what fitting noise looks like.

### It is not one lucky path

A single seed proves nothing: any ladder can be produced by picking the path that
produces it. The same study across 12 independent price paths:

```
Stage                       mean   stdev       t
------------------------ ------- ------- -------
Naive backtest              6.91    0.20  119.95
+ point-in-time data        0.57    0.38    5.23
+ slippage                  0.40    0.34    4.03
+ commissions               0.38    0.34    3.89
+ walk-forward OOS          0.12    0.49    0.86

Verdict on the honest stage: NOT distinguishable from zero (|t| = 0.86).
Sharpe fell at every rung on 11/12 paths.
```

Two things worth being explicit about:

1. The naive backtest's t-statistic of **120** is the point. It is overwhelmingly
   "significant" and completely fake. Statistical significance computed on a
   contaminated backtest measures the contamination.
2. **The ladder is monotone on 11 of 12 paths, not 12 of 12.** One path breaks it.
   That is reported rather than dropped, because dropping it would be the same
   class of error the project is about. The degradation is a strong tendency, not
   a theorem.

The default seed is `20260921` — the date the project was written, fixed before
any results were seen. The multi-seed sweep exists so the claim does not rest on
that choice.

## Reproducing it

One command, about 32 seconds:

```bash
git clone <this repo> && cd honest-backtest
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py
```

Everything above is printed by that command. Useful variants:

```bash
python run_experiment.py --seeds 1         # headline table only, ~2 seconds
python run_experiment.py --seed 42         # a different price path
python run_experiment.py --markdown        # emit the table in Markdown
pytest                                     # 115 tests, ~6 seconds
```

Results are deterministic: the same seed reproduces the same numbers to the last
digit, which is asserted in the test suite.

## Architecture

The engine is **event-driven, not vectorised**. A central queue carries four
event types in strict causal order:

```
                 ┌──────────────────────────────────────────┐
                 │            EVENT QUEUE                   │
                 └──────────────────────────────────────────┘
                    ▲          ▲          ▲          ▲
                    │          │          │          │
   DataHandler ─────┘          │          │          │
     emits MarketEvent         │          │          │
          │                    │          │          │
          ▼                    │          │          │
       Strategy ───────────────┘          │          │
     emits SignalEvent (target weight)    │          │
          │                               │          │
          ▼                               │          │
      Portfolio ───────────────────────────┘         │
     emits OrderEvent (position delta)                │
          │                                           │
          ▼                                           │
   ExecutionHandler ────────────────────────────────── ┘
     applies SlippageModel + CommissionModel
     emits FillEvent
          │
          ▼
      Portfolio   updates cash, position, equity curve
```

The outer loop advances market time exactly once per iteration; the inner loop
drains every event the new bar caused before time moves again. A component can
only ever act on events it has actually received.

This is slower than a vectorised backtest — about **26 ms per thousand bars**,
or roughly 39,000 events per second, where the pandas equivalent is
sub-millisecond. That is the trade, and it buys two things. Look-ahead bias becomes an architectural impossibility rather
than a convention a careless `.shift()` can break; and the same `Strategy` object
could be driven by a live feed without modification, because it consumes bars one
at a time.

### Point-in-time discipline is enforced in code

`HistoricBarDataHandler` owns the full price history but exposes it only through
accessors that clamp to an internal cursor. The cursor advances exactly once per
`update_bars()`, which is the same call that emits the `MarketEvent`. There is no
public method that returns a bar the engine has not reached, and asking for one
raises `LookAheadError` rather than returning data:

```python
def peek_ahead(self, symbol: str, n: int = 1) -> list[Bar]:
    raise LookAheadError(
        f"{type(self).__name__} is a point-in-time data handler and cannot "
        "serve future bars. If you are seeing this, a strategy tried to look ahead."
    )
```

The prohibition is a method that raises rather than a method that is absent, so
it is written down and testable.

Stage 1 needs the bug in order to demonstrate it, so `LookAheadDataHandler` is a
subclass that re-enables peeking, and `LookAheadMomentumStrategy` uses it. Both
are named to be conspicuous in a diff, the strategy refuses to construct against
a point-in-time handler, and the results table labels the stage as cheating.
Running the bug is more convincing than describing it.

### Module layout

```
src/honest_backtest/
├── events.py        MarketEvent → SignalEvent → OrderEvent → FillEvent (frozen dataclasses)
├── data.py          DataHandler interface; point-in-time handler; the deliberate leak
├── synthetic.py     GBM + injected weak signal; the oracle ceiling
├── strategy.py      Strategy interface; momentum; the leaky variant; buy-and-hold
├── slippage.py      SlippageModel: Zero / FixedBps / SpreadPlusImpact
├── commission.py    CommissionModel: Zero / PerShare / PercentOfNotional
├── execution.py     Order → Fill, applying both cost models
├── portfolio.py     Cash, positions, equity curve, order sizing, trade blotter
├── engine.py        The event loop
├── metrics.py       Sharpe, drawdown, returns, turnover
├── walkforward.py   Rolling / anchored train-test splits
├── experiments.py   The five-stage degradation study
└── report.py        Table rendering
```

## The data

The demo runs on **synthetic** prices, generated with a fixed seed, because a
labelled ground truth (the true size of the edge) is only available when the
signal is injected:

```
s_t = φ·s_{t-1} + √(1-φ²)·η_t                  latent AR(1) state, unit variance
r_t = μΔt + α·σ√Δt·s_{t-1} + σ√Δt·z_t          returns, with a weak predictable tilt
P_t = P_{t-1}·exp(r_t)
```

`α = 0.115` is the per-bar information ratio available to an observer who knows
`s` exactly — an annualised ceiling of about 1.3. The state is latent: strategies
never see it and must infer it from past returns, where it sits under noise
roughly `1/α` times larger.

**This is a demonstration harness, not a performance claim.** No conclusion about
any real market follows from it. Synthetic data was chosen because it makes the
ground truth knowable. Because the size of the injected edge is known, the Sharpe a backtest *reports* can be compared against
the Sharpe that is actually *there* — stage 1's 6.94 against a ceiling of 1.30.
With real data you can never separate "my method is biased" from "the market
really did that".

`α` was chosen so the degradation is visible: large enough that the injected edge
is real, small enough that frictions and honest validation can destroy it. That
choice is disclosed here rather than buried, and the oracle Sharpe is printed on
every run so the honest number can be read against the ceiling instead of against
zero.

### Using real data instead

`DataHandler` is an abstract base class with four methods — `update_bars`,
`get_latest_bars`, `current_bar`, `latest_closes` — none of which require random
access to the future. A CSV reader, a database cursor or a live websocket feed
can all satisfy it, and the engine, strategies, portfolio and execution handler
need no changes. Point the new handler at `run_backtest` and everything else
works.

The one thing a real adapter would have to preserve is the cursor discipline: if
your handler can serve a bar the engine has not reached, the guarantee is gone
and the test suite will not catch it for you.

## Tests

115 tests, covering the things that would invalidate the result if they were
wrong rather than the things that are easy to test:

```bash
pytest
```

| File | What it pins down |
| --- | --- |
| `test_data_handler.py` | The point-in-time guarantee: exhaustively, at every cursor position and every window length, no future bar is ever returned; and over a full backtest, no strategy is ever shown a bar it should not have. Also that the deliberate leak really leaks, and cannot be used by accident. |
| `test_portfolio.py` | `equity == cash + position × price` at every bar; the bar-to-bar identity `Δequity = position × Δprice + traded PnL − fees`; and a full reconciliation of the run against the trade blotter. |
| `test_costs.py` | Slippage is adverse by construction for every model and both directions; impact grows with size and is concave; layering each friction strictly reduces final equity. |
| `test_walkforward.py` | Test windows are non-overlapping and contiguous; training always ends before testing begins; warm-up bars come from the training region; overlap must be explicitly opted into. |
| `test_degradation.py` | The headline claim itself, so it cannot drift away from the code: look-ahead inflates Sharpe, each friction reduces it, every stage is scored over identical bars. |
| `test_synthetic.py` | The seed reproduces exactly; the injected edge is forward-looking and weak. |
| `test_metrics.py`, `test_engine.py` | Statistics against hand-computed values; causal event ordering; the queue is fully drained; runs are deterministic. |
| `test_cli.py` | The documented one-line command actually runs and prints the table, including the synthetic-data disclaimer. |

Two real bugs were caught by these tests while writing them, which is the
argument for having them:

- A constant return series produced a Sharpe ratio of `7.3e16`, because an exact
  `sd == 0.0` check misses the `1e-19` that floating-point rounding leaves behind.
- Zero-quantity orders were becoming zero-quantity fills and inflating the trade
  count, which would have made every turnover figure wrong.

## Design decisions and their trade-offs

Where a choice was genuinely arguable, it is recorded here rather than presented
as obvious.

**Event-driven over vectorised.** Costs roughly two orders of magnitude in
speed. Bought in exchange: look-ahead bias becomes structurally impossible, and
the strategy code is live-tradeable as written. For this project the whole point
is the guarantee, so the trade is easy. For a large parameter sweep it would not
be.

**No pandas.** A DataFrame in the hot path is how vectorised backtests get
written; keeping it out of the engine makes the point-in-time discipline harder
to circumvent. numpy is used only for generation and statistics. The cost is that
a real CSV adapter has to do its own parsing.

**Single instrument.** Every accessor takes a `symbol` and the interface
anticipates multi-asset, but the handler serves one. Implementing portfolio
optimisation and cross-sectional signals without a real multi-asset dataset to
test against would be speculative. This is a stated limitation, not an oversight.

**Fractional shares.** Removes lot-size friction. For a liquid, high-priced
instrument the difference is small, and rounding would add a noise term unrelated
to the point of the study. `Portfolio(allow_fractional=False)` turns it off.

**Fills are always complete.** Orders never get rejected or partially filled;
the slippage model carries the size penalty instead. Modelling rejection needs a
liquidity model this project does not have. This is optimistic, and it is named
as optimistic rather than left implicit.

**A 5% no-trade band.** Without a band, a continuous target weight trades every
bar and the result is dominated by dust. Real desks use a band. The value is a
judgement call, and it materially affects the cost figures — turnover runs around
44x per year even with it.

**Costs land in the following period's return.** The equity snapshot is taken
when a bar arrives, before that bar's own fills, so a trade's cost shows up in
the next period. This is the correct attribution, and it means the last snapshot
lags the blotter by one fill — pinned by a test so nobody "fixes" it later and
silently shifts every cost by one period.

**Temporary impact only.** The slippage model has a square-root impact term with
no permanent component and no decay, so it understates the cost of a strategy
that trades the same direction repeatedly. The square-root shape is well
supported empirically; the coefficient is the part nobody agrees on, which is why
it is a constructor argument rather than a hard-coded number.

**Walk-forward returns are stitched on returns, not equity levels.** Concatenating
fold equity curves would invent a jump at each boundary. Each fold's scored
window begins one bar before the test window so the first realised return is
captured, which is what makes stage 5 cover exactly the same bars as stages 1–4.

## What this project does not claim

- That the strategy makes money. It does not; the honest Sharpe is negative on
  the headline seed and statistically zero across seeds.
- That these frictions are calibrated to any particular venue. They are plausible
  equity-like defaults and they are all constructor arguments.
- That a positive walk-forward result would have been sufficient. It would have
  been necessary, not sufficient — with 12 paths and one strategy, multiple
  testing is still lurking.
- That synthetic data tells you anything about real markets. It tells you about
  your method.

## License

MIT.
