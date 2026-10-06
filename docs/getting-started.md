# Getting started

## Install

```bash
pip install honest-backtester
```

The PyPI name is `honest-backtester` because `honest-backtest` was already
taken there. Everything else uses the project's own name: you
`import honest_backtest` and run `honest-backtest`. numpy is the only
dependency, and Python 3.10 to 3.13 are supported.

To work from a clone instead:

```bash
git clone https://github.com/rohanKraghu/honest-backtest
cd honest-backtest
pip install -e ".[dev]"
pytest
```

## Reproduce the study

```bash
honest-backtest                  # the headline table and a 12-path sweep
honest-backtest --seeds 1        # the headline table only, a few seconds
honest-backtest --html study.html
```

The price series is synthetic, with a weak edge injected on purpose, so the
right answer is known: the oracle line in the report shows the best any
strategy could have done. Results are deterministic; the same seed gives the
same numbers to the last digit.

## Audit your own strategy

Write a strategy file that defines `SPEC` (see
[Writing a strategy](guides/strategies.md)), point the audit at a price file,
and read the last rung:

```bash
honest-backtest audit --data prices.csv --strategy my_strategy.py --html audit.html
```

A clone ships a synthetic sample file and an example strategy, so this works
without any data of your own:

```bash
honest-backtest audit --data examples/sample_prices.csv \
    --strategy examples/sma_crossover.py
```

## From Python

```python
from honest_backtest import load_csv_bars
from honest_backtest.audit import AuditConfig, load_spec, render_audit_report, run_audit

bars = load_csv_bars("prices.csv")
spec = load_spec("my_strategy.py")
result = run_audit(bars, spec, AuditConfig())
print(render_audit_report(result))
```
