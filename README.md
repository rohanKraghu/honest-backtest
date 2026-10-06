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

| # | Stage | Sharpe | P(edge) | Total return | Ann. return | Max DD | Trades | Lookback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Naive backtest | **6.94** | >99.9% | +80925.7% | +131.0% | -3.1% | 2132 | 5 |
| 2 | + point-in-time data | 0.43 | 77.8% | +46.7% | +4.9% | -26.4% | 1665 | 20 |
| 3 | + slippage | 0.27 | 60.9% | +24.0% | +2.7% | -31.1% | 1664 | 20 |
| 4 | + commissions | 0.26 | 59.7% | +23.0% | +2.6% | -31.4% | 1665 | 20 |
| 5 | + walk-forward OOS | **-0.19** | 29.8% | -22.7% | -3.2% | -29.8% | 1789 | 5/5/45/30/20/20/5/5 |

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

- **P(edge) is the probability the Sharpe is more than luck.** For stages 1
  to 4 it is the Deflated Sharpe Ratio (Bailey and López de Prado, 2014),
  which asks whether the Sharpe beats the best one you would expect from
  trying that many lookbacks on pure noise, and corrects for skew and fat
  tails. Stage 5 chose nothing on the bars it reports, so it gets the plain
  Probabilistic Sharpe Ratio against zero. The naive rung's >99.9% is the
  leak talking: significance computed on a backtest that reads the future
  measures the leak. The honest rung's 29.8% says a negative Sharpe this
  size is unremarkable for a strategy with no edge.

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
python run_experiment.py --html study.html # also write the charts as a web page
python run_experiment.py --json study.json # also write every number as JSON
python run_experiment.py --config examples/study.json   # settings from a file
python run_experiment.py --fast --workers 4  # same numbers, about 8 seconds
pytest                                     # 347 tests, ~40 seconds
```

Results are deterministic: the same seed reproduces the same numbers to the last
digit, which is asserted in the test suite.

## Auditing your own strategy

The same ladder runs on any strategy and any price file:

```bash
pip install -e .
honest-backtest audit --data examples/sample_prices.csv --strategy examples/sma_crossover.py
```

```
#  Stage                 Sharpe  P(edge)  Total return  Ann. return  Max DD  Trades  fast,slow
1  In-sample, no costs   0.10    45.7%    +3.8%         +0.5%        -27.3%  35      20,100
2  + slippage            0.08    43.4%    +1.0%         +0.1%        -27.9%  35      20,100
3  + commissions         0.08    43.3%    +0.9%         +0.1%        -27.9%  35      20,100
4  + next-bar execution  0.09    43.8%    +2.2%         +0.3%        -29.5%  27      50,100
5  + walk-forward OOS    -0.12   36.3%    -19.9%        -2.7%        -42.2%  40      50,100/50,100/20,200/...

  Buy and hold, same window, costs  Sharpe   0.28
  Walk-forward out of sample        Sharpe  -0.12   <- the only number worth quoting
  Out-of-sample t-stat ~ -0.35: NOT distinguishable from zero.
  P(edge), out of sample: 36.3% (Probabilistic Sharpe Ratio against zero)
```

`examples/sample_prices.csv` is ten years of **synthetic** prices with weekday
dates, generated by `examples/make_sample_prices.py`, so the command works from
a fresh clone. Point `--data` at your own file for a real answer.

**The data file** needs a date column and a close column, oldest row first;
open, high, low, volume and an adjusted close are used when present (column
names are case-insensitive). The loader refuses dates out of order or
duplicated, and non-positive prices. An adjusted-close column rescales the
whole bar, so a split does not look like a crash; `--no-adjust` turns that off.
A file without volume needs `--default-volume`, because volume drives the
impact model and a made-up value should be a deliberate choice.

**The strategy file** defines `SPEC`, a `StrategySpec` naming a builder, the
parameter grid the in-sample fit may search, and the warm-up it needs:

```python
SPEC = StrategySpec(
    name="moving-average crossover",
    build=MovingAverageCrossover,          # (events, data, symbol, **params) -> Strategy
    grid=param_grid(fast=(10, 20, 50), slow=(100, 200)),
    warmup=200,
)
```

`--strategy file.py:NAME` picks a different attribute, and `--strategy momentum`
audits the built-in strategy. A strategy reads prices only through the
point-in-time handler, so it cannot look ahead even by accident.

**What changes from the synthetic study.** A strategy without a deliberately
leaky twin gets four rungs, opening with a frictionless in-sample fit, since
that is what most backtests report. There is no oracle on real data, so the
out-of-sample Sharpe is read against buy-and-hold and against zero, with its
approximate t-stat (Sharpe × √years). The impact model's volatility is
estimated from the first training window only, which ends before any scored
bar. Costs, windows and capital are all flags; `honest-backtest audit --help`
lists them. `--fast` prints the same report sooner (see the design notes on the
fast path).

## Paper trading

The claim that a strategy here is live-tradeable as written is tested, not just
stated. `honest-backtest paper` follows a price file that something else appends
to (a scheduled download, say) and trades each new row with the audit's spread,
impact and commission model:

```bash
honest-backtest paper --data prices.csv --strategy my_strategy.py \
    --params fast=20,slow=100 --journal paper.jsonl
```

Rows already in the file warm the strategy up: they are replayed through it with
every signal thrown away, so its indicators and any internal state are exactly
what they would be at that point in a backtest, and nothing trades on the past.
Orders fill on the next bar (at its open when the file has real opens, else its
close), because a bar's close has printed by the time the bar is complete. To
see it work on the bundled sample, `--live-from` replays the last months as if
they were arriving and `--once` stops when the file runs out:

```bash
honest-backtest paper --data examples/sample_prices.csv \
    --strategy examples/sma_crossover.py --live-from 2024-06-01 --once
```

```
2024-08-29  close 112.4685  position 8,994.09  equity 1,003,811.98
2024-08-30  close 111.4421  position 8,994.09  equity 994,580.45

Live bars          65
Fills              1
Equity             994,580.45 (-0.54% over these bars)
Costs paid         slippage 941.39, commission 44.97
```

Without `--params` the setting with the best in-sample Sharpe on the history is
used, and the header says so, since that is an optimistic choice. Every bar is
appended to the journal (JSON lines, flushed per bar) with cash, position,
equity, fills and working orders. After a restart, `--resume` restores the book
from it and keeps the setting and cost calibration it began with; a resumed run
writes the same journal, line for line, as one that never stopped. A row that
changes after it was traded on stops the run, since accepting a revised past
would make the record unreproducible; for the same reason adjusted closes are
off unless you pass `--adjusted`.

In code, `PaperTrader` takes any `Feed`: `ReplayFeed` for a list of bars,
`CsvTailFeed` for a growing file, or `PollingFeed` around a function you write
that asks a broker or data vendor for its latest completed bars. It is built
from the same portfolio, execution handler and cost models as `run_backtest` and
driven by the engine's own loop, so a replayed feed reproduces a backtest's
equity curve and blotter exactly, which `test_live.py` asserts. It is
single-instrument, fills are simulated (orders never leave the process), and a
permanent-impact model's decaying push is not carried across a restart.

### More than one out-of-sample path

Walk-forward produces one out-of-sample path, so its Sharpe is a single draw.
`--cpcv N,K` adds combinatorial purged cross-validation (López de Prado,
*Advances in Financial Machine Learning*, ch. 12) after the ladder, and reports
how that Sharpe is spread across paths:

```bash
honest-backtest audit --data examples/sample_prices.csv \
    --strategy examples/sma_crossover.py --cpcv 6,2
```

```
Combinatorial purged cross-validation (6 groups, 2 held out per split; purge 200, embargo 0 bars):
  15 splits, 5 paths over the same 2016 bars, every bar tested 5 times
  Path Sharpe  median  -0.06   quartiles -0.09 to 0.03   range -0.21 to 0.13
  Paths below zero: 3 of 5 (60%)
  Walk-forward's one path scored -0.12; the spread above is how far
  one out-of-sample path can land from another with different training data.
```

How it works, and what it guarantees:

- **The window is the ladder's.** The scored window is cut into `N`
  contiguous groups and every combination of `K` groups is held out once,
  giving `C(N, K)` splits. Every rung and every path covers the same bars,
  and none of them the window the cost model was calibrated on.
- **Training never sees a test bar.** The training set is every other bar,
  less a **purge** on both sides of each test block (`--cpcv-purge`, default
  the strategy's warm-up, since that is how far a lookback reaches across a
  boundary) and an **embargo** after it (`--cpcv-embargo`, default 0).
  Training segments are replayed one at a time, each warming up on its own
  first bars, so no test, purged or embargoed bar reaches a training run
  even as indicator history. The setting with the best pooled Sharpe over
  the segments wins, as in every other fit.
- **Testing warms up on the past only.** A held-out group is scored with the
  bars just before it replayed unscored, exactly as a walk-forward fold is:
  they are strictly earlier, a live trader would have had them, and nothing
  after the group is replayed.
- **Paths rebuild the whole window.** Every bar is tested `C(N-1, K-1)` times,
  and the tests are assigned to that many paths, each a complete walk through
  the window stitched on returns. The tests in `test_cpcv_splits.py` and
  `test_cpcv.py` pin each of these.

The paths are not independent draws: they share bars, and their training
sets overlap heavily, so the spread is a picture of how much one
out-of-sample number depends on which data it was trained on, not a
confidence interval. With `N = 6, K = 2` there are only five paths; more
groups give more paths but shorter, more heavily purged training segments,
and a purge that leaves no segment long enough to warm up is refused. Unlike
walk-forward, CPCV trains on bars after the ones it tests; the purge and
embargo address information that leaks across the boundary, not a market
that changes, which is why walk-forward stays the headline and this is an
extra. The HTML report (`--html`) draws the paths as dots against the
walk-forward number. Liquidity and carry settings and `--fast` apply to
every CPCV run exactly as they do to the walk-forward rung. In code,
`CombinatorialPurgedSplitter` produces the splits and paths and `run_cpcv`
runs any `StrategySpec` through them.

### Settings files and JSON results

Every flag of the study and of `audit` can live in a file, and every number
in the report can be written out for another program:

```bash
honest-backtest audit --config examples/audit.yaml --json audit.json
python run_experiment.py --config examples/study.json --seeds 1
```

A config file is a flat mapping whose keys are the long option names, with
dashes or underscores (`train_size: 504`, `cpcv: [6, 2]`, `no_leak_check:
true`). Each value goes through the same conversion and checks as the flag, so
an unknown key or a value the flag would refuse is an error, not a silent
default. Flags on the command line override the file, and relative paths in
the file resolve against the file's own directory. JSON configs need nothing
extra; YAML needs PyYAML (`pip install pyyaml`, or `pip install -e ".[yaml]"`),
and without it a `.yaml` file is refused with that instruction.
`examples/audit.yaml` is the audit above with comments; `examples/study.json`
is the headline study.

`--json PATH` writes strict JSON (undefined values are `null`, never `NaN`):
the settings, every rung's metrics, P(edge) and chosen parameters, the folds,
the cost models and the costs paid, buy and hold, and either the oracle and
seed sweep (study) or the look-ahead check, t-stat and cross-validation
distribution (audit). The tests read it back and find each number in the
printed report, and check that a config file and the same flags write
byte-identical JSON.

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
runs on a live feed without modification, because it consumes bars one at a
time (see [Paper trading](#paper-trading)).

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
├── slippage.py      SlippageModel: Zero / FixedBps / SpreadPlusImpact / PermanentImpact
├── commission.py    CommissionModel: Zero / PerShare / PercentOfNotional
├── execution.py     Order → Fill: timing, volume cap, partial fills, limit orders
├── portfolio.py     Cash, positions, equity curve, order sizing, trade blotter
├── financing.py     Interest on cash and margin, short-borrow fees, leverage cap
├── frictions.py     Liquidity, order style and carry, bundled for the ladder
├── multiasset.py    Panel handler, multi-asset book, cross-sectional momentum
├── engine.py        The event loop
├── fast.py          Signal replay that reproduces the engine exactly, for sweeps
├── live.py          Live feeds, the live handler and the paper trader
├── metrics.py       Sharpe, drawdown, returns, turnover; Probabilistic and Deflated Sharpe
├── walkforward.py   Rolling / anchored train-test splits; combinatorial purged CV splits
├── spec.py          StrategySpec: builder + parameter grid + warm-up
├── experiments.py   The generic ladder, the five-stage synthetic study, CPCV runs
├── csvdata.py       Real price files into bars, with order and adjustment checks
├── audit.py         The ladder on your own strategy and data
├── leaks.py         The look-ahead check: replace the future, compare the past
├── report.py        Table rendering
├── html_report.py   The self-contained HTML report
├── paper_cli.py     `honest-backtest paper`
├── config.py        --config: JSON or YAML settings, checked like flags
├── export.py        --json: every number in the report, machine-readable
└── cli.py           `honest-backtest` and `honest-backtest audit`
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

For a CSV, use `load_csv_bars` or the `audit` command above. Beyond files,
`DataHandler` is an abstract base class with four methods — `update_bars`,
`get_latest_bars`, `current_bar`, `latest_closes` — none of which require random
access to the future. A database cursor or a live websocket feed can satisfy it
too, and the engine, strategies, portfolio and execution handler need no
changes.

The one thing a real adapter would have to preserve is the cursor discipline: if
your handler can serve a bar the engine has not reached, the guarantee is gone
and the test suite will not catch it for you.

**The look-ahead check.** The point-in-time handler stops a strategy that
uses its accessors from seeing tomorrow, but not one that goes around them
(reaching into the handler's private list, normalising with statistics a
helper computed over the whole file, or keeping state between runs). So
before running the ladder, the audit tests the definition of look-ahead
directly: at five cut points it replaces every bar after the cut with a
different, plausible future and replays every setting in the grid. Any
signal at or before the cut that moves is a leak. The report then opens
with a warning and the command exits with status 1, so it can gate CI. A
strategy whose signals differ between two runs on identical data is
reported as uncheckable rather than passed. The check cannot see a strategy
that reads its own copy of the data, since the altered future never reaches
it. `--no-leak-check` skips it; in code, `detect_look_ahead(bars, spec)`
returns the same report.

**Liquidity, carry and permanent impact.** Four more assumptions can be
removed from an audit, and each is off unless asked for:

```bash
honest-backtest audit --data prices.csv --strategy my_strategy.py \
    --permanent-impact 0.3 --impact-half-life 5 \
    --max-participation 0.05 --borrow-rate 0.06 --short-fee 0.02 --max-leverage 2
```

`--permanent-impact` swaps in the decaying permanent impact model on every
slippage rung. The volume cap (`--max-participation`), passive limit orders
(`--limit-offset-bps`, `--limit-expiry`) and financing (`--cash-rate`,
`--borrow-rate`, `--short-fee`, `--max-leverage`) add one "+ liquidity and
carry" rung after the execution rungs, which walk-forward and buy and hold
inherit, so the rungs before it are unchanged.

**A page to send someone.** `--html report.html` (on the study and on
`audit`) writes a single self-contained HTML file: a Sharpe waterfall that
starts at the in-sample headline and steps down rung by rung to the
out-of-sample number, every rung's equity curve on a log axis against buy and
hold, slippage and commission paid per rung, and the full table with P(edge).
It is inline SVG and CSS with no scripts and nothing fetched, so it opens
offline and can be attached to an email or a pull request as it is, and it
follows the reader's light or dark setting.

**When orders fill.** The synthetic study fills at the close of the bar that
produced the signal, which assumes you can see a close and trade on it in the
same instant. Audits default to `--fill auto`, which adds a "+ next-bar
execution" rung: orders fill at the next bar's open when the file has real
opens, and at the next bar's close when it only has closes (a derived open is
just the previous close, so "next open" would quietly be the same instant).
The walk-forward rung and buy-and-hold use the same timing. `--fill close`
keeps the old behaviour, and `--fill next_open` or `--fill next_close` force
one. The engine itself takes `fill_timing` on `run_backtest` and
`LadderSettings`.

## Tests

347 tests, covering the things that would invalidate the result if they were
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
| `test_cli.py` | The documented one-line command actually runs and prints the table, including the synthetic-data disclaimer; `--fast` and `--workers` print the same report. |
| `test_spec.py` | The ladder runs on strategies other than momentum, with four rungs when there is no leaky twin; the built-in study is exactly the generic ladder on the momentum spec. |
| `test_csvdata.py` | Real files load with dates and adjustment; out-of-order dates, duplicates, bad prices and a silently missing volume are refused. |
| `test_audit.py` | The `audit` command runs end to end on a user file, strategy references resolve or fail with a reason, and the cost model is calibrated before the scored window. |
| `test_fill_timing.py` | Next-bar fills happen exactly one bar later at that bar's open or close, a signal on the final bar never fills, and the ladder gains its next-bar rung only when asked. |
| `test_deflated_sharpe.py` | PSR, expected maximum Sharpe and DSR against their definitions: more trials and negative skew both lower the probability, one trial makes DSR equal PSR, and every rung of a study reports one. |
| `test_leaks.py` | The look-ahead check passes honest strategies and catches the leaky twin, a strategy reading the handler's private list, one refused by the handler, and state shared between runs; random strategies are flagged as uncheckable; a leaky audit opens with a warning and exits non-zero. |
| `test_html_report.py` | The HTML report is self-contained (no scripts, nothing fetched), carries every rung and reference, escapes names, and is written by `--html`; every rung keeps returns for exactly the scored window. |
| `test_permanent_impact.py` | Repeated same-direction fills pay more, the push decays with its half-life, a reversal is never paid back, and a shared model is reset between runs so reruns are identical. |
| `test_liquidity.py` | A volume cap splits orders across bars, a newer order cancels the leftover instead of trading it twice, capped runs reconcile against the blotter, and limit orders fill only when price trades through, at the limit, and expire. |
| `test_financing.py` | Cash interest, margin interest and short fees compound exactly on a flat price, the leverage cap clips targets, and the bar-to-bar accounting identity holds with financing included. |
| `test_multiasset.py` | No instrument is read ahead of the shared cursor, panels align on common dates, a one-instrument panel reproduces the single-asset engine exactly, the book reconciles per instrument, and gross leverage is capped across it. |
| `test_audit_frictions.py` | Inactive frictions change nothing, active ones add exactly one rung and leave earlier rungs untouched, and the audit flags work end to end. |
| `test_fast.py` | The fast path returns exactly the engine's equity, fills and costs (equality, not closeness) for every fill timing, cost model and leaky strategy; the fast study and a parallel seed sweep match the serial engine; it declines settings it cannot reproduce. |
| `test_live.py` | A replayed feed reproduces the backtest's equity and blotter exactly for every fill timing; history warms a stateful strategy to exactly its backtest state and is never traded; a live strategy never sees a bar early; out-of-order bars, rewritten rows and undated polls are refused; a resumed journal matches an uninterrupted one line for line. |
| `test_paper_cli.py` | `paper` replays a stretch of history as live bars, refuses to overwrite a journal, resumes after the file grows to give the uninterrupted journal exactly, and refuses to resume with a different setting. |
| `test_cpcv_splits.py` | Combinatorial purged splits, over a spread of group counts, purges and embargoes: no training bar inside a test block or within its purge or embargo, nothing else dropped, every bar tested exactly `C(N-1, K-1)` times, and every path walking the whole window using each test once. |
| `test_cpcv.py` | The CPCV runner replays only each split's training segments and each group with the bars before it; rewriting every bar outside a split's training set cannot change its fit; path Sharpes come from the stitched paths; the fast path gives the same paths and frictions apply as on the walk-forward rung; `--cpcv` leaves every rung unchanged and reports the spread in text and HTML. |
| `test_config.py` | A JSON or YAML config reproduces the same flags' output exactly; flags override the file; paths resolve against the file; unknown keys and bad values are refused like bad flags; a YAML file without PyYAML asks for it. |
| `test_export.py` | `--json` parses as strict JSON and every rung, reference, sweep row, leak check and CPCV statistic matches the printed report; permanent impact, liquidity and carry settings are recorded; a config and the same flags write byte-identical JSON. |

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
the strategy code is live-tradeable as written, which `paper` demonstrates. For this project the whole point
is the guarantee, so the trade is easy. For a large parameter sweep it would not
be, which is what the fast path is for.

**A fast path that must agree to the last digit.** Most of a study's time goes
on rerunning the same strategy with the same parameters over the same bars
while only the cost model changes. `--fast` (or `LadderSettings(fast=True)`)
records each setting's signals once, by driving the real strategy through the
real point-in-time handler, then replays them through the portfolio's sizing
rules and the same slippage and commission objects in the engine's order. No
signal logic is rewritten, so there is no second implementation to drift, and
`test_fast.py` asserts bit-for-bit equality rather than closeness. Volume caps,
limit orders and financing are not reproduced, so a ladder with any of those
falls back to the engine for that rung. The seed sweep also takes `--workers`
to run seeds in separate processes; each seed is deterministic, so the result
does not depend on the count. On four cores the full study drops from about 36
seconds to about 8.

**No pandas.** A DataFrame in the hot path is how vectorised backtests get
written; keeping it out of the engine makes the point-in-time discipline harder
to circumvent. numpy is used only for generation and statistics. The cost is that
a real CSV adapter has to do its own parsing.

**One calendar for every instrument.** `run_backtest` also takes a panel, a
dict of bars per instrument (`multiasset.py`). All instruments share one cursor
that advances for all of them at once, so none can be read further ahead than
another; `align_panel` puts dated series on one calendar by keeping only the
days every instrument traded, rather than inventing prices for the gaps. A
one-instrument panel reproduces the single-asset engine exactly, which a test
pins. The degradation ladder and `audit` still run on one instrument; a
panel is run directly with `run_backtest` (see
`examples/cross_sectional_momentum.py`).

**Fractional shares.** Removes lot-size friction. For a liquid, high-priced
instrument the difference is small, and rounding would add a noise term unrelated
to the point of the study. `Portfolio(allow_fractional=False)` turns it off.

**Fills are complete unless you say otherwise.** By default an order fills in
full at one price and the slippage model carries the size penalty. With
`max_participation` a fill may take only that share of the bar's volume, and the
rest works on later bars until it is done or a newer order replaces it (the
portfolio sizes every order from what it actually holds, so a leftover would
otherwise be traded twice). With `limit_offset_bps` the portfolio rebalances
with passive limit orders that rest from the next bar and fill at the limit
only when price trades strictly through it, since touching it says nothing
about queue position. They pay no spread, and the trades they miss are the
cost; `BacktestResult.unfilled_quantity` reports how much.

**A 5% no-trade band.** Without a band, a continuous target weight trades every
bar and the result is dominated by dust. Real desks use a band. The value is a
judgement call, and it materially affects the cost figures — turnover runs around
44x per year even with it.

**Costs land in the following period's return.** The equity snapshot is taken
when a bar arrives, before that bar's own fills, so a trade's cost shows up in
the next period. This is the correct attribution, and it means the last snapshot
lags the blotter by one fill — pinned by a test so nobody "fixes" it later and
silently shifts every cost by one period.

**Temporary impact by default.** The headline slippage model has a square-root
impact term with no permanent component, so it understates the cost of a
strategy that trades the same direction repeatedly. The square-root shape is
well supported empirically; the coefficient is the part nobody agrees on, which
is why it is a constructor argument rather than a hard-coded number.
`PermanentImpactSlippage` adds a linear permanent push per fill that decays
with a half-life, charged to later fills in the same direction. It is never
credited back on a reversal: the marks come from data that never saw your
trades, so crediting it would book a gain the equity curve cannot show.

**Holding is free by default.** `Financing` charges interest on a margin loan,
a lending fee on short market value, pays interest on positive cash, and caps
gross leverage (across the whole book for a panel). It accrues once per bar on
the book held over that bar, and the accounting identity test includes it.
Sharpe is measured against zero, so a cash rate raises it without any skill;
leave `cash_rate` at zero when quoting Sharpe.

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
  been necessary, not sufficient. P(edge) deflates each in-sample rung for the
  lookbacks that rung tried, but it cannot count the strategies, features and
  ideas tried before this one was written down, so multiple testing is still
  lurking.
- That synthetic data tells you anything about real markets. It tells you about
  your method.

## Roadmap

What is built and what is planned, in the order it is being worked on.
Sizes are rough: S is a few hours, M a day or two, L longer.

**Done**

- Real price files: `load_csv_bars` reads a CSV with dates, adjusted closes
  and order checks.
- A generic degradation ladder: any `StrategySpec` (builder, parameter grid,
  warm-up), not only the built-in momentum strategy.
- `honest-backtest audit`: the ladder on your own strategy and data.
- Next-bar execution as its own rung (`--fill`).
- P(edge): Deflated Sharpe for in-sample rungs, Probabilistic Sharpe for the
  out-of-sample one.
- The look-ahead check that runs before every audit.
- The self-contained HTML report (`--html`).
- Multi-asset panels and books in the engine, with cross-sectional momentum
  as an example. The ladder and `audit` still run on one instrument.
- Partial fills under a volume participation cap, and passive limit orders.
- Permanent impact that decays with a half-life (`PermanentImpactSlippage`).
- Financing: margin interest, short-borrow fees, interest on cash and a
  leverage cap.
- A fast path (`--fast`) that reproduces the engine bit for bit, and a
  parallel seed sweep (`--workers`).
- `honest-backtest paper`: paper trading on a growing price file, with a
  journal and resume.
- A PyPI-ready package (`honest-backtester`, since `honest-backtest` is taken
  on PyPI), an MkDocs site and a release workflow. Nothing is published yet.
- Combinatorial purged cross-validation (`audit --cpcv`): a distribution of
  out-of-sample Sharpe across purged, embargoed folds.
- Config files (`--config`, JSON or YAML) and machine-readable results
  (`--json`).

**Planned**

| Item | What it adds | Size |
| --- | --- | --- |
| Parquet input and an optional yfinance loader | More ways in besides CSV | S |
| First PyPI release and a live docs site | `pip install honest-backtester`, once the name is settled | S |

## License

MIT; see [LICENSE](LICENSE).
