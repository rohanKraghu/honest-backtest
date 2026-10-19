"""Parquet files and Yahoo downloads must give exactly the bars a CSV gives."""

from __future__ import annotations

import sys
import types
from datetime import datetime

import pytest

from honest_backtest.csvdata import CSVFormatError, load_csv_bars, load_yfinance_bars

CSV = """Date,Open,High,Low,Close,Adj Close,Volume
2024-01-02,100,102,99,101,50.5,1000
2024-01-03,101,103,100,102,51.0,1100
2024-01-04,102,104,101,103,51.5,1200
"""


def _csv_bars(tmp_path):
    path = tmp_path / "spy.csv"
    path.write_text(CSV)
    return load_csv_bars(path)


def test_parquet_gives_the_same_bars_as_csv(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(
        {
            "Date": [datetime(2024, 1, d) for d in (2, 3, 4)],
            "Open": [100.0, 101.0, 102.0],
            "High": [102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [101.0, 102.0, 103.0],
            "Adj Close": [50.5, 51.0, 51.5],
            "Volume": [1000, 1100, 1200],
        }
    )
    path = tmp_path / "spy.parquet"
    pq.write_table(table, path)
    assert load_csv_bars(path) == _csv_bars(tmp_path)


def test_parquet_gets_the_same_checks(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "spy.parquet"
    pq.write_table(
        pa.table(
            {"Date": ["2024-01-03", "2024-01-02"], "Close": [1.0, 2.0], "Volume": [1, 1]}
        ),
        path,
    )
    with pytest.raises(CSVFormatError, match="oldest first"):
        load_csv_bars(path)


def test_yfinance_download_gives_the_same_bars_as_csv(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    frame = pd.read_csv(pd.io.common.StringIO(CSV), index_col="Date", parse_dates=True)
    frame.index = frame.index.tz_localize("America/New_York")
    calls = []

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            calls.append((self.symbol, kwargs))
            return frame.copy()

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=Ticker))
    bars = load_yfinance_bars("spy", start="2024-01-01", end="2024-01-05")
    assert bars == _csv_bars(tmp_path)
    assert calls == [
        ("spy", {"auto_adjust": False, "start": "2024-01-01", "end": "2024-01-05"})
    ]
