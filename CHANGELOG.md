# Changelog

The package's name on PyPI is `honest-backtester` (the name
`honest-backtest` was already taken there). The import name is
`honest_backtest` and the command is `honest-backtest`.

## 0.1.0 (unreleased)

The first release.

### The study and the audit

- The five-stage degradation study on synthetic data, with an oracle ceiling
  and a robustness sweep over independent price paths.
- `honest-backtest audit`: the same ladder on your own strategy and price file,
  with the impact model calibrated before the scored window.
- P(edge) on every rung: the Deflated Sharpe Ratio for in-sample fits, the
  Probabilistic Sharpe Ratio out of sample.
- Next-bar fills (`--fill`), with the timing chosen from whether the file has
  real opens.
- A look-ahead check that replays the strategy against altered futures and
  exits with status 1 on a leak.
- A self-contained HTML report (`--html`) with a Sharpe waterfall, equity
  curves and costs per rung.

### Validation and automation

- Combinatorial purged cross-validation (`--cpcv N,K`, with `--cpcv-purge`
  and `--cpcv-embargo`): the spread of out-of-sample Sharpe across paths,
  beside the walk-forward number, in text, HTML and JSON.
- `--config FILE` for the study and the audit: every flag from a JSON or YAML
  file, checked exactly like the flag, with the command line taking priority.
  YAML needs the `yaml` extra.
- `--json PATH`: every number in the report as strict JSON.

### Execution realism

- Decaying permanent impact, a volume participation cap with partial fills,
  passive limit orders, and financing (cash and margin interest, short fees,
  a leverage cap), each off unless asked for.
- Multi-asset panels on one shared cursor, a multi-asset book and a
  cross-sectional momentum example.

### Speed

- A fast path (`--fast`) that replays each setting's signals once and
  reproduces the engine's results bit for bit, and a parallel seed sweep
  (`--workers`).

### Paper trading

- `honest-backtest paper` and `PaperTrader`: the backtested strategy,
  unchanged, on bars as they arrive from a growing CSV file, a polling
  function or a replay, with a per-bar journal and exact resume.

### Packaging

- numpy is the only dependency (any version from 1.24); tested on Python
  3.10 to 3.13.
- Type information is shipped (`py.typed`).
