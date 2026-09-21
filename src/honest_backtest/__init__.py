"""honest-backtest: an event-driven backtester built to show why backtests lie.

The public surface is small on purpose. The interesting modules are
:mod:`honest_backtest.data` (point-in-time discipline),
:mod:`honest_backtest.engine` (the event loop) and
:mod:`honest_backtest.experiments` (the degradation study).
"""

from __future__ import annotations

__version__ = "0.1.0"

from .commission import (
    CommissionModel,
    PercentOfNotionalCommission,
    PerShareCommission,
    ZeroCommission,
)
from .data import (
    Bar,
    DataHandler,
    HistoricBarDataHandler,
    LookAheadDataHandler,
    LookAheadError,
    bars_from_series,
)
from .engine import Backtest, BacktestResult, run_backtest
from .events import (
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from .execution import SimulatedExecutionHandler
from .metrics import PerformanceMetrics, compute_metrics
from .portfolio import Portfolio, PortfolioSnapshot
from .slippage import (
    FixedBpsSlippage,
    SlippageModel,
    SpreadPlusImpactSlippage,
    ZeroSlippage,
)
from .strategy import (
    BuyAndHoldStrategy,
    LookAheadMomentumStrategy,
    Strategy,
    TimeSeriesMomentumStrategy,
)
from .synthetic import SyntheticConfig, SyntheticSeries, generate_price_series
from .walkforward import Fold, WalkForwardSplitter

__all__ = [
    "Backtest",
    "BacktestResult",
    "Bar",
    "BuyAndHoldStrategy",
    "CommissionModel",
    "DataHandler",
    "EventType",
    "FillEvent",
    "FixedBpsSlippage",
    "Fold",
    "HistoricBarDataHandler",
    "LookAheadDataHandler",
    "LookAheadError",
    "LookAheadMomentumStrategy",
    "MarketEvent",
    "OrderEvent",
    "PerShareCommission",
    "PercentOfNotionalCommission",
    "PerformanceMetrics",
    "Portfolio",
    "PortfolioSnapshot",
    "SignalEvent",
    "SimulatedExecutionHandler",
    "SlippageModel",
    "SpreadPlusImpactSlippage",
    "Strategy",
    "SyntheticConfig",
    "SyntheticSeries",
    "TimeSeriesMomentumStrategy",
    "WalkForwardSplitter",
    "ZeroCommission",
    "ZeroSlippage",
    "bars_from_series",
    "compute_metrics",
    "generate_price_series",
    "run_backtest",
    "__version__",
]
