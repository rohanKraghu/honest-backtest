# Python API

The most used names are importable from `honest_backtest` directly.

## Running backtests

::: honest_backtest.engine.run_backtest

::: honest_backtest.engine.BacktestResult

## Strategies

::: honest_backtest.spec.StrategySpec

::: honest_backtest.spec.param_grid

::: honest_backtest.strategy.Strategy

## Data

::: honest_backtest.data.Bar

::: honest_backtest.data.DataHandler

::: honest_backtest.csvdata.load_csv_bars

## Audits and the ladder

::: honest_backtest.audit.AuditConfig

::: honest_backtest.audit.run_audit

::: honest_backtest.experiments.LadderSettings

::: honest_backtest.experiments.run_ladder

::: honest_backtest.leaks.detect_look_ahead

## Costs

::: honest_backtest.slippage.SpreadPlusImpactSlippage

::: honest_backtest.slippage.PermanentImpactSlippage

::: honest_backtest.commission.PerShareCommission

::: honest_backtest.commission.PercentOfNotionalCommission

::: honest_backtest.financing.Financing

## Paper trading

::: honest_backtest.live.PaperTrader

::: honest_backtest.live.ReplayFeed

::: honest_backtest.live.PollingFeed

::: honest_backtest.live.CsvTailFeed

## Speed

::: honest_backtest.fast
    options:
      members: [replay_signals, simulate, supports]
