"""Real price files must load into bars without silently bending time or price."""

from __future__ import annotations

from datetime import datetime

import pytest

from honest_backtest.csvdata import CSVFormatError, load_csv_bars
from honest_backtest.engine import run_backtest
from honest_backtest.strategy import BuyAndHoldStrategy


def _write(tmp_path, text, name="spy.csv"):
    path = tmp_path / name
    path.write_text(text.strip() + "\n")
    return path


OHLCV = """
Date,Open,High,Low,Close,Adj Close,Volume
2024-01-02,100,102,99,101,50.5,1000
2024-01-03,101,103,100,102,51.0,1100
2024-01-04,102,104,101,103,51.5,1200
"""


def test_ohlcv_file_loads_with_dates_and_index_timestamps(tmp_path):
    bars = load_csv_bars(_write(tmp_path, OHLCV), use_adjusted=False)
    assert [b.timestamp for b in bars] == [0, 1, 2]
    assert bars[0].time == datetime(2024, 1, 2)
    assert bars[0].symbol == "SPY"
    assert (bars[0].open, bars[0].high, bars[0].low, bars[0].close) == (100, 102, 99, 101)
    assert bars[2].volume == 1200


def test_adjusted_close_rescales_the_whole_bar(tmp_path):
    """A 2:1 split must not look like a 50% crash, so every price is scaled."""
    bars = load_csv_bars(_write(tmp_path, OHLCV))
    assert bars[0].close == pytest.approx(50.5)
    assert bars[0].open == pytest.approx(50.0)
    assert bars[0].high == pytest.approx(51.0)
    assert bars[0].low == pytest.approx(49.5)


def test_column_names_are_case_insensitive_and_close_only_works(tmp_path):
    text = "DATE,CLOSE,VOLUME\n2024-01-02,10,5\n2024-01-03,11,5\n2024-01-04,9,5"
    bars = load_csv_bars(_write(tmp_path, text))
    assert [b.close for b in bars] == [10, 11, 9]
    # Derived exactly as for synthetic bars: open is the previous close.
    assert bars[1].open == 10
    assert (bars[2].high, bars[2].low) == (11, 9)


def test_dates_out_of_order_are_rejected(tmp_path):
    text = "date,close,volume\n2024-01-03,10,5\n2024-01-02,11,5"
    with pytest.raises(CSVFormatError, match="oldest first"):
        load_csv_bars(_write(tmp_path, text))


def test_duplicate_dates_are_rejected(tmp_path):
    text = "date,close,volume\n2024-01-02,10,5\n2024-01-02,11,5"
    with pytest.raises(CSVFormatError, match="not after"):
        load_csv_bars(_write(tmp_path, text))


@pytest.mark.parametrize("bad", ["0", "-3", "nan", "abc", ""])
def test_non_positive_or_missing_prices_are_rejected(tmp_path, bad):
    text = f"date,close,volume\n2024-01-02,10,5\n2024-01-03,{bad},5"
    with pytest.raises(CSVFormatError, match="line 3"):
        load_csv_bars(_write(tmp_path, text))


def test_missing_volume_needs_an_explicit_assumption(tmp_path):
    text = "date,close\n2024-01-02,10\n2024-01-03,11"
    with pytest.raises(CSVFormatError, match="default_volume"):
        load_csv_bars(_write(tmp_path, text))
    bars = load_csv_bars(_write(tmp_path, text), default_volume=5e5)
    assert all(b.volume == 5e5 for b in bars)


def test_missing_required_columns_are_named(tmp_path):
    with pytest.raises(CSVFormatError, match="date column and a close column"):
        load_csv_bars(_write(tmp_path, "day,price\n1,2\n2,3"))


def test_loaded_bars_run_through_the_engine(tmp_path):
    rows = "\n".join(f"2024-01-{d:02d},{100 + d},1000000" for d in range(1, 31))
    bars = load_csv_bars(_write(tmp_path, "date,close,volume\n" + rows))
    result = run_backtest(
        bars,
        lambda events, data: BuyAndHoldStrategy(events, data, symbol="SPY"),
        symbol="SPY",
    )
    assert len(result.equity) == 30
    assert result.equity[-1] > result.equity[0]
