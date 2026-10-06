"""Loading real price history from CSV files.

The engine never sees a file: it replays a list of :class:`~honest_backtest.data.Bar`
through the point-in-time handler, exactly as it does for synthetic data. This
module only turns a CSV into that list, so every guarantee the handler gives
holds unchanged for real data.

Parsing uses the standard library ``csv`` module rather than pandas, for the
reason the README gives: keeping a DataFrame out of the project keeps the
vectorised habits that cause look-ahead bias out of it too.

Two things are checked here because they silently corrupt a backtest if they
are wrong, and neither is visible in a results table:

* **Order.** Dates must be strictly increasing. A file sorted newest-first,
  replayed as-is, runs the strategy backwards in time.
* **Adjustment.** A raw close drops on every split and ex-dividend date. A
  strategy trading on raw closes sees a crash that never happened. When an
  adjusted-close column is present, every price on the bar is scaled by it.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

from .data import Bar

_DATE_COLUMNS = ("date", "datetime", "timestamp", "time")
_ADJ_CLOSE_COLUMNS = (
    "adj close",
    "adj_close",
    "adjclose",
    "adjusted close",
    "adjusted_close",
)


class CSVFormatError(ValueError):
    """Raised when a price file cannot be turned into a trustworthy bar series."""


def _normalise(name: str) -> str:
    return name.strip().lower()


def _find(columns: dict[str, str], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _parse_time(raw: str, line: int) -> datetime:
    text = raw.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.combine(date.fromisoformat(text), datetime.min.time())
        except ValueError as exc:
            raise CSVFormatError(
                f"line {line}: cannot parse date {raw!r}; use ISO 8601 (YYYY-MM-DD)"
            ) from exc
    return parsed


def _parse_price(raw: str, column: str, line: int) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise CSVFormatError(f"line {line}: {column} is not a number: {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise CSVFormatError(f"line {line}: {column} must be positive, got {raw!r}")
    return value


def load_csv_bars(
    path: str | Path,
    symbol: str | None = None,
    *,
    use_adjusted: bool = True,
    default_volume: float | None = None,
) -> list[Bar]:
    """Read a daily (or any regular) OHLCV file into a list of bars.

    Column names are matched case-insensitively. A date column and a close
    column are required; open, high, low, volume and an adjusted close are
    used when present. With only closes, open/high/low are derived from
    consecutive closes exactly as :func:`~honest_backtest.data.bars_from_series`
    does for synthetic data.

    Args:
        path: CSV file to read.
        symbol: Name stamped on every bar. Defaults to the file's stem.
        use_adjusted: If the file has an adjusted-close column, scale open,
            high, low and close by ``adj_close / close`` so splits and
            dividends do not appear as price moves.
        default_volume: Volume to use when the file has no volume column.
            ``None`` means the file must contain one. Volume feeds the market
            impact model, so a made-up value should be a deliberate choice.

    Returns:
        Bars in ascending time order. ``Bar.timestamp`` is the bar's index in
        the series and ``Bar.time`` carries the parsed date.

    Raises:
        CSVFormatError: If a required column is missing, a value is not a
            positive number, or the dates are not strictly increasing.
    """
    path = Path(path)
    symbol = symbol or path.stem.upper()

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CSVFormatError(f"{path} is empty")
        columns = {_normalise(name): name for name in reader.fieldnames}

        date_col = _find(columns, _DATE_COLUMNS)
        close_col = _find(columns, ("close",))
        if date_col is None or close_col is None:
            raise CSVFormatError(
                f"{path} needs a date column and a close column; "
                f"found {reader.fieldnames}"
            )
        open_col = _find(columns, ("open",))
        high_col = _find(columns, ("high",))
        low_col = _find(columns, ("low",))
        volume_col = _find(columns, ("volume",))
        adj_col = _find(columns, _ADJ_CLOSE_COLUMNS) if use_adjusted else None

        if volume_col is None and default_volume is None:
            raise CSVFormatError(
                f"{path} has no volume column. Volume drives the market impact "
                "model; pass default_volume (or --default-volume) to assume one."
            )
        have_ohlc = open_col is not None and high_col is not None and low_col is not None

        bars: list[Bar] = []
        previous_time: datetime | None = None
        previous_close: float | None = None
        for line, row in enumerate(reader, start=2):
            when = _parse_time(row[date_col], line)
            if previous_time is not None and when <= previous_time:
                raise CSVFormatError(
                    f"line {line}: date {when.date()} is not after "
                    f"{previous_time.date()}. "
                    "Rows must be sorted oldest first with no duplicates."
                )

            close = _parse_price(row[close_col], close_col, line)
            factor = 1.0
            if adj_col is not None:
                factor = _parse_price(row[adj_col], adj_col, line) / close

            if have_ohlc:
                o = _parse_price(row[open_col], open_col, line)
                h = _parse_price(row[high_col], high_col, line)
                lo = _parse_price(row[low_col], low_col, line)
            else:
                o = previous_close / factor if previous_close is not None else close
                h, lo = max(o, close), min(o, close)

            if volume_col is not None:
                try:
                    volume = float(row[volume_col])
                except ValueError as exc:
                    raise CSVFormatError(
                        f"line {line}: volume is not a number: {row[volume_col]!r}"
                    ) from exc
                if not math.isfinite(volume) or volume < 0:
                    raise CSVFormatError(f"line {line}: volume must be non-negative")
            else:
                volume = float(default_volume)

            adj_close = close * factor
            bars.append(
                Bar(
                    timestamp=len(bars),
                    symbol=symbol,
                    open=o * factor,
                    high=h * factor,
                    low=lo * factor,
                    close=adj_close,
                    volume=volume,
                    time=when,
                )
            )
            previous_time = when
            previous_close = adj_close

    if len(bars) < 2:
        raise CSVFormatError(f"{path} has {len(bars)} rows; need at least 2")
    return bars


def has_real_opens(bars: list[Bar]) -> bool:
    """Whether any bar's open differs from the previous close.

    A close-only file gets opens derived from the previous close, and then
    filling "at the next open" is the same as filling at this close. The
    audit uses this to fall back to the next bar's close in that case.
    """
    return any(b.open != a.close for a, b in zip(bars, bars[1:], strict=False))
