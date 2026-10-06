# Settings files and JSON results

Every flag of the study and of `audit` can live in a file, and every number in
the report can be written out for another program:

```bash
honest-backtest audit --config examples/audit.yaml --json audit.json
honest-backtest --config examples/study.json --seeds 1
```

## Config files

A config file is a flat mapping whose keys are the command's long option
names, with dashes or underscores:

```yaml
--8<-- "examples/audit.yaml"
```

- **Checked like flags.** Each value goes through the same conversion and
  checks as the flag would, so an unknown key or a value the flag would refuse
  is an error, not a silent default.
- **Flags win.** Anything given on the command line overrides the file, so
  `honest-backtest audit --config examples/audit.yaml --fill close` reruns
  that audit without the next-bar rung.
- **Paths are relative to the file.** `data: sample_prices.csv` in
  `examples/audit.yaml` means the file next to it, wherever the command is run
  from.
- **JSON needs nothing; YAML needs PyYAML.** Install it with
  `pip install "honest-backtester[yaml]"`. Without it a `.yaml` file is refused
  with that instruction.

## JSON results

`--json PATH` writes strict JSON: undefined values are `null`, never `NaN`, so
any parser reads it. The document carries a `schema_version` and a `kind`
(`"study"` or `"audit"`), then:

| Key | What it holds |
| --- | --- |
| `settings` | Windows, account and fill timing; for an audit also liquidity and carry and whether the fast path was on, for the study the synthetic path's parameters. |
| `costs` | The slippage model (with its calibrated volatility, and permanent impact when on, for an audit) and the commission model. |
| `ladder` | Every rung's metrics, costs paid, P(edge) and chosen parameters, the walk-forward folds, and buy and hold. |
| `strategy`, `data` | The strategy's grid and warm-up; the file's symbol, dates and scored window (audit). |
| `leak_check`, `honest_t_stat` | What the look-ahead check found, and the out-of-sample t-statistic (audit). |
| `cpcv` | Groups, splits, paths and the path Sharpe distribution, when `--cpcv` was given (audit). |
| `oracle_sharpe`, `sweep` | The oracle ceiling and the seed sweep (study). |

The tests read each document back and find every number in the printed
report, and check that a config file and the same flags write
byte-identical JSON.

In code, `honest_backtest.export.audit_dict(result)` and
`study_dict(result, sweep)` build the same documents, and `to_json` serialises
them.
