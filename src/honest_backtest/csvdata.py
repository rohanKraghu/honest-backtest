"""Loading real price history from CSV and Parquet files, or from Yahoo Finance.

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
import io
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

    A ``.parquet`` or ``.pq`` file is read the same way, column for column,
    and needs the ``parquet`` extra (``pip install honest-backtester[parquet]``).

    Column names are matched case-insensitively. A date column and a close
    column are required; open, high, low, volume and an adjusted close are
    used when present. With only closes, open/high/low are derived from
    consecutive closes exactly as :func:`~honest_backtest.data.bars_from_series`
    does for synthetic data.

    Args:
        path: CSV or Parquet file to read.
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
    if path.suffix.lower() in (".parquet", ".pq"):
        handle = _parquet_as_csv(path)
    else:
        handle = path.open(newline="", encoding="utf-8-sig")
    with handle:
        bars = read_csv_rows(
            handle,
            symbol,
            source=str(path),
            use_adjusted=use_adjusted,
            default_volume=default_volume,
        )
    if len(bars) < 2:
        raise CSVFormatError(f"{path} has {len(bars)} rows; need at least 2")
    return bars


def _parquet_as_csv(path: Path) -> io.StringIO:
    """Turn a Parquet file into CSV text, so it goes through the same checks."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "reading Parquet needs pyarrow: pip install honest-backtester[parquet]"
        ) from exc
    rows = pq.read_table(path).to_pylist()
    text = io.StringIO()
    if rows:
        writer = csv.DictWriter(text, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {k: "" if v is None else str(v) for k, v in row.items()} for row in rows
        )
    text.seek(0)
    return text


def load_yfinance_bars(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    *,
    use_adjusted: bool = True,
) -> list[Bar]:
    """Download daily bars for one ticker from Yahoo Finance.

    Needs the ``yfinance`` extra (``pip install honest-backtester[yfinance]``)
    and a network connection. The download goes through the same parser and
    checks as :func:`load_csv_bars`, and dates lose their time zone so they
    line up with bars read from files.

    Args:
        ticker: Yahoo symbol, for example ``"SPY"``.
        start: First date, ``YYYY-MM-DD``. ``None`` means as far back as Yahoo goes.
        end: Day after the last date, ``YYYY-MM-DD``. ``None`` means today.
        use_adjusted: As for :func:`load_csv_bars`.

    Returns:
        Bars in ascending time order.

    Raises:
        CSVFormatError: If Yahoo returns fewer than 2 rows or bad values.
    """
    try:
        import yfinance
    except ImportError as exc:
        raise ImportError(
            "downloading from Yahoo needs yfinance: "
            "pip install honest-backtester[yfinance]"
        ) from exc
    kwargs = {"start": start, "end": end} if start or end else {"period": "max"}
    frame = yfinance.Ticker(ticker).history(auto_adjust=False, **kwargs)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    source = f"Yahoo Finance data for {ticker}"
    bars = read_csv_rows(
        io.StringIO(frame.to_csv()),
        ticker.upper(),
        source=source,
        use_adjusted=use_adjusted,
    )
    if len(bars) < 2:
        raise CSVFormatError(f"{source} has {len(bars)} rows; need at least 2")
    return bars


def read_csv_rows(
    handle: Iterable[str],
    symbol: str,
    *,
    source: str = "the file",
    use_adjusted: bool = True,
    default_volume: float | None = None,
) -> list[Bar]:
    """Parse CSV text into bars, with every check :func:`load_csv_bars` makes.

    The building block of :func:`load_csv_bars`, also used to read a file
    that is still being written (see :class:`~honest_backtest.live.CsvTailFeed`).

    Args:
        handle: Lines of CSV text, header first.
        symbol: Name stamped on every bar.
        source: How to name the input in error messages.
        use_adjusted: As for :func:`load_csv_bars`.
        default_volume: As for :func:`load_csv_bars`.

    Returns:
        Bars in ascending time order, possibly none.
    """
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise CSVFormatError(f"{source} is empty")
    columns = {_normalise(name): name for name in reader.fieldnames}

    date_col = _find(columns, _DATE_COLUMNS)
    close_col = _find(columns, ("close",))
    if date_col is None or close_col is None:
        raise CSVFormatError(
            f"{source} needs a date column and a close column; found {reader.fieldnames}"
        )
    open_col = _find(columns, ("open",))
    high_col = _find(columns, ("high",))
    low_col = _find(columns, ("low",))
    volume_col = _find(columns, ("volume",))
    adj_col = _find(columns, _ADJ_CLOSE_COLUMNS) if use_adjusted else None

    if volume_col is None and default_volume is None:
        raise CSVFormatError(
            f"{source} has no volume column. Volume drives the market impact "
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
    return bars


def has_real_opens(bars: list[Bar]) -> bool:
    """Whether any bar's open differs from the previous close.

    A close-only file gets opens derived from the previous close, and then
    filling "at the next open" is the same as filling at this close. The
    audit uses this to fall back to the next bar's close in that case.
    """
    return any(b.open != a.close for a, b in zip(bars, bars[1:], strict=False))
