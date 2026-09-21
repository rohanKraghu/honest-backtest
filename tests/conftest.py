"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from honest_backtest.data import Bar, bars_from_series
from honest_backtest.synthetic import SyntheticConfig, SyntheticSeries, generate_price_series


@pytest.fixture
def short_bars() -> list[Bar]:
    """A tiny, hand-checkable bar series."""
    return bars_from_series([100.0, 101.0, 102.0, 99.0, 103.0, 104.0, 98.0])


@pytest.fixture
def series() -> SyntheticSeries:
    """A small synthetic series, fixed seed."""
    return generate_price_series(SyntheticConfig(n_bars=400), seed=12345)


@pytest.fixture
def bars(series: SyntheticSeries) -> list[Bar]:
    """Bars from the small synthetic series."""
    return series.to_bars()
