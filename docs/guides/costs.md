# Costs, liquidity and carry

By default an audit charges a half-spread, square-root market impact and
commission, and fills completely at one price. Four more assumptions can be
removed, and each is off unless you ask for it:

```bash
honest-backtest audit --data prices.csv --strategy my_strategy.py \
    --permanent-impact 0.3 --impact-half-life 5 \
    --max-participation 0.05 --borrow-rate 0.06 --short-fee 0.02 --max-leverage 2
```

`--permanent-impact` replaces the slippage model on every slippage rung. The
others add one "+ liquidity and carry" rung after the execution rungs, which
walk-forward and buy and hold inherit, so the rungs before it are unchanged.

## Permanent impact

`PermanentImpactSlippage` adds a linear push per fill that decays with a
half-life and is charged to later fills in the same direction, so a strategy
that keeps buying pays more each time. It is never credited back on a
reversal: the marks come from data that never saw your trades, so crediting it
would book a gain the equity curve cannot show.

## Volume limits and partial fills

With `--max-participation 0.05` no fill may take more than 5% of the bar's
volume. The rest works on later bars until it is done or a newer order
replaces it. A newer order cancels the leftover rather than adding to it,
because the portfolio sizes every order from what it actually holds.

## Passive limit orders

With `--limit-offset-bps 5` the portfolio rebalances with limit orders 5 bps
inside the close. They rest from the next bar for `--limit-expiry` bars and
fill only on a bar that trades strictly through the limit, at the limit, since
touching it says nothing about your place in the queue. They pay no spread;
the trades they miss are the cost, and `BacktestResult.unfilled_quantity`
reports how much went unfilled.

## Financing

`--cash-rate` pays interest on positive cash, `--borrow-rate` charges it on a
margin loan, `--short-fee` charges a lending fee on short market value, and
`--max-leverage` caps gross leverage. Interest accrues once per bar on the book
held over that bar. Sharpe here is measured against zero, so a cash rate raises
it without any skill; leave it at zero when quoting Sharpe.

## In code

```python
from honest_backtest import Financing, PermanentImpactSlippage, run_backtest

result = run_backtest(
    bars,
    factory,
    slippage=PermanentImpactSlippage(bar_volatility=0.01),
    max_participation=0.05,
    limit_offset_bps=None,
    financing=Financing(borrow_rate=0.06, short_fee=0.02, max_leverage=2.0),
)
```
