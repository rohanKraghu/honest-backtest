r"""``honest-backtest paper``: paper trade a strategy on a price file as it grows.

The file is read the way ``audit`` reads it. Rows already in it warm the
strategy up (and calibrate the impact model); rows appended later, by a
scheduled download for instance, are traded as they arrive::

    honest-backtest paper --data prices.csv --strategy my_strategy.py \\
        --params fast=20,slow=100 --journal paper.jsonl

``--live-from DATE`` treats rows from that date on as live, which replays a
stretch of history as if it were arriving now, and ``--once`` stops when the
file has no new rows instead of waiting for more. ``--resume`` continues a
journal after a restart.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from .audit import AuditConfig, load_spec
from .csvdata import has_real_opens, load_csv_bars
from .data import Bar
from .execution import FILL_TIMINGS
from .experiments import LadderSettings, fit_in_sample
from .live import CsvTailFeed, PaperState, PaperTrader, read_journal
from .spec import Params, StrategySpec, format_params


def build_paper_parser() -> argparse.ArgumentParser:
    """Build the parser for ``honest-backtest paper``."""
    parser = argparse.ArgumentParser(
        prog="honest-backtest paper",
        description=(
            "Paper trade a strategy on a price file, filling each order with the "
            "audit's cost model as new rows arrive."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, required=True, help="CSV file to follow")
    parser.add_argument(
        "--strategy", required=True, help="strategy file, file.py:NAME, or 'momentum'"
    )
    parser.add_argument("--symbol", default=None, help="instrument name")
    parser.add_argument(
        "--params",
        default=None,
        help="the setting to trade, e.g. fast=20,slow=100; without it the setting "
        "with the best in-sample Sharpe on the history is used",
    )
    parser.add_argument(
        "--live-from",
        default=None,
        metavar="DATE",
        help="treat rows from this date on as live (default: only rows added later)",
    )
    parser.add_argument(
        "--journal", type=Path, default=None, help="append one JSON line per bar here"
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue the book saved in --journal"
    )
    parser.add_argument(
        "--poll", type=float, default=60.0, help="seconds between checks of the file"
    )
    parser.add_argument(
        "--once", action="store_true", help="stop when there are no new rows"
    )
    parser.add_argument(
        "--max-bars", type=int, default=None, help="stop after this many live bars"
    )
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="cash")
    parser.add_argument(
        "--rebalance-band", type=float, default=0.05, help="no-trade band, of equity"
    )
    parser.add_argument(
        "--fill",
        choices=("auto", *FILL_TIMINGS),
        default="auto",
        help="when orders fill; auto means next_open with real opens, else next_close",
    )
    parser.add_argument("--half-spread-bps", type=float, default=2.0, help="half-spread")
    parser.add_argument(
        "--impact", type=float, default=0.6, help="square-root impact coefficient"
    )
    parser.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        help="commission as bps of notional (overrides --commission-per-share)",
    )
    parser.add_argument(
        "--commission-per-share", type=float, default=0.005, help="per-share fee"
    )
    parser.add_argument(
        "--default-volume", type=float, default=None, help="volume if the file has none"
    )
    parser.add_argument(
        "--adjusted",
        action="store_true",
        help="scale prices by an adjusted-close column; a later adjustment then "
        "rewrites traded rows and stops the run",
    )
    parser.add_argument("--bars-per-year", type=int, default=252, help="annualisation")
    return parser


def parse_params(text: str, spec: StrategySpec) -> Params:
    """Parse ``name=value,...`` into a setting with the grid's parameter names."""
    params: dict[str, Any] = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        name, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"expected name=value, got {item!r}")
        value: Any = raw.strip()
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                continue
        params[name.strip()] = value
    expected = set(spec.grid[0])
    if set(params) != expected:
        raise ValueError(
            f"{spec.name!r} takes {', '.join(sorted(expected)) or 'no parameters'}; "
            f"got {', '.join(sorted(params)) or 'none'}"
        )
    return params


def _parse_date(text: str) -> datetime:
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"--live-from: not a date: {text!r}") from exc


def _describe(state: PaperState) -> str:
    when = (
        state.bar.time.date().isoformat() if state.bar.time else str(state.bar.timestamp)
    )
    line = (
        f"{when}  close {state.bar.close:,.4f}  position {state.position:,.2f}  "
        f"equity {state.equity:,.2f}"
    )
    for fill in state.fills:
        line += f"  {fill.direction} {fill.quantity:,.2f} @ {fill.fill_price:,.4f}"
    return line


def paper_main(argv: list[str]) -> int:
    """Run ``honest-backtest paper`` until the feed stops or Ctrl-C."""
    args = build_paper_parser().parse_args(argv)
    journal: Path | None = args.journal
    if args.resume and journal is None:
        raise SystemExit("--resume needs --journal")
    if journal is not None and journal.exists() and not args.resume:
        raise SystemExit(
            f"{journal} already exists; pass --resume to continue it or choose "
            "another path"
        )

    bars = load_csv_bars(
        args.data,
        symbol=args.symbol,
        use_adjusted=args.adjusted,
        default_volume=args.default_volume,
    )
    symbol = bars[0].symbol
    started: dict | None = None
    if args.resume:
        journaled = read_journal(journal)  # type: ignore[arg-type]
        started = next((r for r in journaled if r.get("type") == "start"), None)
        rows = [r for r in journaled if r.get("type") == "bar"]
        if started is None or not rows:
            raise SystemExit(f"{journal} has no bars to resume from")
        last = datetime.fromisoformat(rows[-1]["time"])
        history = [b for b in bars if b.time <= last]  # type: ignore[operator]
        # Costs and the setting stay as they were fixed when the run began.
        calibration = bars[: started["history_bars"]]
    elif args.live_from is not None:
        start = _parse_date(args.live_from)
        history = [b for b in bars if b.time < start]  # type: ignore[operator]
        calibration = history
    else:
        history = calibration = bars
    if len(calibration) < 2:
        raise SystemExit(
            "need at least two rows of history before the first live bar, to "
            "calibrate the impact model and warm the strategy up"
        )

    spec = load_spec(args.strategy)
    fill = args.fill
    if fill == "auto":
        fill = "next_open" if has_real_opens(bars) else "next_close"
    costs = AuditConfig(
        settings=LadderSettings(train_size=len(calibration)),
        half_spread_bps=args.half_spread_bps,
        impact_coefficient=args.impact,
        commission_bps=args.commission_bps,
        commission_per_share=args.commission_per_share,
    )
    slippage, commission = costs.slippage(calibration), costs.commission()

    saved = started.get("meta", {}).get("params") if started else None
    if args.params is not None:
        params = parse_params(args.params, spec)
        how = "given"
        if saved is not None and params != saved:
            raise SystemExit(
                f"{journal} was started with {format_params(saved)}; resuming with "
                "a different setting would mix two strategies in one record"
            )
    elif saved is not None:
        params, how = saved, "as the journal was started"
    elif len(spec.grid) == 1:
        params, how = dict(spec.grid[0]), "the only setting"
    else:
        params = _fit(calibration, spec, fill, args, slippage, commission)
        how = "best in-sample Sharpe on the history, an optimistic choice"

    feed = CsvTailFeed(
        args.data,
        symbol=symbol,
        use_adjusted=args.adjusted,
        default_volume=args.default_volume,
        interval=args.poll,
        idle_timeout=0 if args.once else None,
        after=history[-1].time,
    )
    trader = PaperTrader(
        feed,
        spec.factory(params),
        history=history,
        symbol=symbol,
        slippage=slippage,
        commission=commission,
        initial_capital=args.capital,
        rebalance_threshold=args.rebalance_band,
        fill_timing=fill,
        bars_per_year=args.bars_per_year,
        journal=journal,
        journal_meta={"strategy": spec.name, "params": params},
        resume=args.resume,
    )
    print(f"honest-backtest paper - {spec.name} on {symbol}")
    print(f"  setting       {format_params(params)} ({how})")
    print(f"  fills         {fill}, with the audit's spread, impact and commission")
    print(f"  history       {len(history)} bars, through {_when(history[-1])}")
    print(f"  journal       {journal if journal is not None else 'none'}")
    if not args.once:
        print(f"  waiting for new rows in {args.data}; Ctrl-C stops")
    print()
    try:
        trader.run(
            max_bars=args.max_bars, on_bar=lambda s: print(_describe(s), flush=True)
        )
    except KeyboardInterrupt:
        trader.close()
        print("\nStopped.")
    result = trader.result()
    print()
    if not result.equity:
        print("No live bars yet.")
        return 0
    equity = (
        result.final_cash + result.final_position * trader.data.current_bar(symbol).close
    )
    start_equity = result.equity[0]
    change = equity / start_equity - 1 if start_equity > 0 else math.nan
    print(f"Live bars          {trader.bars_processed}")
    print(f"Fills              {result.n_trades}")
    print(f"Equity             {equity:,.2f} ({change:+.2%} over these bars)")
    print(
        f"Costs paid         slippage {result.total_slippage:,.2f}, "
        f"commission {result.total_commission:,.2f}"
    )
    return 0


def _when(bar: Bar) -> str:
    return bar.time.date().isoformat() if bar.time else f"bar {bar.timestamp}"


def _fit(history, spec, fill, args, slippage, commission) -> Params:
    """Pick the setting with the best Sharpe on the history, after costs."""
    if len(history) <= spec.warmup + 2:
        raise SystemExit(
            f"{len(history)} rows of history cannot fit {spec.name!r}, which needs "
            f"{spec.warmup} bars of warm-up; pass --params"
        )
    settings = LadderSettings(
        train_size=len(history),
        initial_capital=args.capital,
        rebalance_threshold=args.rebalance_band,
        bars_per_year=args.bars_per_year,
        fill_timing=fill,
    )
    params, _ = fit_in_sample(
        history,
        spec,
        settings,
        slippage=slippage,
        commission=commission,
        warmup=spec.warmup,
        fill_timing=fill,
    )
    return params
