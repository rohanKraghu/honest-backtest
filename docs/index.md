# honest-backtest

An event-driven backtester built to demonstrate a negative result: a strategy
that looks outstanding under the assumptions most backtests quietly make is
worthless once those assumptions are removed one at a time.

The headline output is not a return figure. It is the rate at which a return
figure decays when you stop lying to yourself.

```text
#  Stage                 Sharpe  P(edge)
1  Naive backtest          6.94  >99.9%
2  + point-in-time data    0.43   77.8%
3  + slippage              0.27   60.9%
4  + commissions           0.26   59.7%
5  + walk-forward OOS     -0.19   29.8%   <- the only number worth quoting
```

Same strategy, same price path, every stage scored over the same eight years.
Look-ahead bias alone accounts for most of the apparent edge, and the honest
out-of-sample answer is that this strategy does not work. The full write-up of
that study is in the
[README](https://github.com/rohanKraghu/honest-backtest#readme).

## What you can do with it

- **Audit your own strategy** on your own prices: the same ladder, from an
  in-sample fit down to walk-forward out of sample, with spread, impact and
  commission calibrated before the scored window. See
  [Auditing a strategy](guides/audit.md).
- **Find look-ahead bugs** automatically. The audit replays your strategy
  against altered futures and fails if any past signal moves.
- **Model the frictions** most backtests leave out: permanent impact, volume
  limits, limit orders that do not fill, financing and short fees. See
  [Costs, liquidity and carry](guides/costs.md).
- **Paper trade** the same strategy object, unchanged, on bars as they arrive.
  See [Paper trading](guides/paper-trading.md).

## Why event-driven

A vectorised backtest computes every signal at once, which is fast and is also
why a single misplaced `shift` can read tomorrow's price without anyone
noticing. Here a bar arrives, the strategy reacts, an order is sized, a fill is
priced, and only then does time move. Strategies can only read the past through
a point-in-time data handler that raises `LookAheadError` when asked for the
future. The cost is speed, which the [fast path](guides/speed.md) recovers for
parameter sweeps without giving up any of the guarantees.

```bash
pip install honest-backtester
honest-backtest audit --data prices.csv --strategy my_strategy.py
```
