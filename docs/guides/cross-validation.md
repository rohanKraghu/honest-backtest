# More than one out-of-sample path

Walk-forward produces one out-of-sample path, so its Sharpe is a single draw.
`--cpcv N,K` adds combinatorial purged cross-validation (López de Prado,
*Advances in Financial Machine Learning*, ch. 12) after the ladder and reports
how that Sharpe is spread across paths:

```bash
honest-backtest audit --data examples/sample_prices.csv \
    --strategy examples/sma_crossover.py --cpcv 6,2
```

```text
Combinatorial purged cross-validation (6 groups, 2 held out per split; purge 200, embargo 0 bars):
  15 splits, 5 paths over the same 2016 bars, every bar tested 5 times
  Path Sharpe  median  -0.06   quartiles -0.09 to 0.03   range -0.21 to 0.13
  Paths below zero: 3 of 5 (60%)
  Walk-forward's one path scored -0.12; the spread above is how far
  one out-of-sample path can land from another with different training data.
```

The ladder itself is unchanged by `--cpcv`; the spread is printed after it,
and the HTML report (`--html`) draws the paths as dots against the
walk-forward number.

## What it guarantees

**The window is the ladder's.** The scored window is cut into `N` contiguous
groups and every combination of `K` groups is held out once, giving `C(N, K)`
splits. Every rung and every path covers the same bars, and none of them the
window the cost model was calibrated on.

**Training never sees a test bar.** The training set is every other bar, less
a purge on both sides of each test block and an embargo after it. Training
segments are replayed one at a time, each warming up on its own first bars, so
no test, purged or embargoed bar reaches a training run even as indicator
history. The setting with the best pooled Sharpe over the segments wins, as in
every other fit.

**Testing warms up on the past only.** A held-out group is scored with the
bars just before it replayed unscored, exactly as a walk-forward fold is: they
are strictly earlier, a live trader would have had them, and nothing after the
group is replayed.

**Paths rebuild the whole window.** Every bar is tested `C(N-1, K-1)` times,
and the tests are assigned to that many paths, each a complete walk through the
window stitched on returns.

**Costs match the walk-forward rung.** Slippage, commission, next-bar fills,
and any [liquidity and carry](costs.md) settings apply to every CPCV run
exactly as they do to the walk-forward rung, and `--fast` speeds it up the same
way without changing a number.

## Purge and embargo

| Flag | Default | What it removes from training |
| --- | --- | --- |
| `--cpcv-purge BARS` | the strategy's warm-up | Bars on each side of a test block, since a lookback that long can straddle the boundary. |
| `--cpcv-embargo BARS` | 0 | Extra bars after a test block, for effects that outlast the lookback. |

More groups give more paths but shorter, more heavily purged training
segments. A purge that leaves no segment long enough to warm up is refused
rather than fitted on noise.

## How to read it

The paths are not independent draws: they share bars and their training sets
overlap heavily, so the spread is a picture of how much one out-of-sample
number depends on which data it was trained on, not a confidence interval.
Unlike walk-forward, CPCV trains on bars after the ones it tests. The purge and
embargo deal with information leaking across a boundary, not with a market
that changes over time, which is why walk-forward stays the headline and this
is an extra.

## In code

```python
from honest_backtest import CombinatorialPurgedSplitter
from honest_backtest.audit import AuditConfig, run_audit

result = run_audit(bars, spec, AuditConfig(cpcv=(6, 2)))
print(result.cpcv.median, result.cpcv.share_below_zero)
```

`CombinatorialPurgedSplitter` produces the splits and paths on its own, and
`run_cpcv` runs any `StrategySpec` through them. Both are in the
[API reference](../reference/api.md#cross-validation).
