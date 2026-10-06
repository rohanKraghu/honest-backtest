# Auditing a strategy

`honest-backtest audit` runs the degradation ladder on your strategy and your
prices. Each rung removes one comfortable assumption, and every rung is scored
over the same bars, so the drop from one rung to the next is the cost of that
assumption and nothing else.

```bash
honest-backtest audit --data prices.csv --strategy my_strategy.py
```

## The rungs

| Rung | What changes |
| --- | --- |
| In-sample, no costs | The grid setting with the best Sharpe on the very window being reported: what most backtests show. |
| + slippage | Half-spread plus square-root market impact on every fill. |
| + commissions | Per-share or percent-of-notional fees. |
| + next-bar execution | Orders fill at the next bar's open (or close), not the close they were computed from. |
| + liquidity and carry | Only when asked for; see [Costs, liquidity and carry](costs.md). |
| + walk-forward OOS | Parameters re-fitted on each training window and scored only on the following, unseen window. |

The last rung is the number to quote. The report also prints buy and hold over
the same window with the same costs, an approximate t-statistic for the
out-of-sample Sharpe, and P(edge) on every rung: the Deflated Sharpe Ratio for
the in-sample fits (which penalises the number of settings tried) and the
Probabilistic Sharpe Ratio out of sample.

## The data file

A CSV with a date column and a close column, oldest row first. Open, high, low,
volume and an adjusted close are used when present. Rows out of order or
duplicated, and non-positive prices, are refused. An adjusted-close column
rescales the whole bar so a split does not look like a crash (`--no-adjust`
turns that off). A file without volume needs `--default-volume`, because volume
drives the impact model and a made-up value should be a deliberate choice.

The impact model's volatility is estimated from the first training window
only, which ends before any scored bar.

## The look-ahead check

Before the ladder runs, the audit replaces every bar after each of five cut
points with a different, plausible future and replays every setting in the
grid. If any signal at or before a cut changes, the strategy read the future.
The report then opens with a warning and the command exits with status 1, so it
can gate CI. A strategy whose signals differ between two runs on identical data
is reported as uncheckable rather than passed. `--no-leak-check` skips it; in
code, `detect_look_ahead(bars, spec)` returns the same report.

The check cannot see a strategy that keeps its own copy of the data, because
the altered future never reaches it. Read prices only through the handler.

## A page to send someone

`--html report.html` writes one self-contained file: a Sharpe waterfall from
the in-sample headline down to the out-of-sample number, every rung's equity
curve against buy and hold, and the costs paid per rung. It has no scripts and
fetches nothing, so it opens offline and can be attached as it is.

## More than one path, and results for other programs

`--cpcv 6,2` adds combinatorial purged cross-validation after the ladder, to
show how far one out-of-sample Sharpe can land from another; see
[Cross-validation paths](cross-validation.md). `--config audit.yaml` reads the
flags from a file and `--json results.json` writes every number in the report;
see [Settings files and JSON](config-and-json.md).

Every flag is listed in the [command-line reference](../reference/cli.md).
