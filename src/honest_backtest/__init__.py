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
from .csvdata import CSVFormatError, load_csv_bars
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
from .financing import Financing
from .leaks import LeakReport, detect_look_ahead
from .live import (
    CsvTailFeed,
    Feed,
    FeedError,
    LiveDataHandler,
    PaperState,
    PaperTrader,
    PollingFeed,
    ReplayFeed,
    read_journal,
)
from .metrics import PerformanceMetrics, compute_metrics
from .multiasset import (
    CrossSectionalMomentumStrategy,
    MultiAssetDataHandler,
    MultiAssetPortfolio,
    align_panel,
    load_csv_panel,
)
from .portfolio import Portfolio, PortfolioSnapshot
from .slippage import (
    FixedBpsSlippage,
    PermanentImpactSlippage,
    SlippageModel,
    SpreadPlusImpactSlippage,
    ZeroSlippage,
)
from .spec import StrategySpec, param_grid
from .strategy import (
    BuyAndHoldStrategy,
    LookAheadMomentumStrategy,
    Strategy,
    TimeSeriesMomentumStrategy,
)
from .synthetic import SyntheticConfig, SyntheticSeries, generate_price_series
from .walkforward import (
    CombinatorialPurgedSplitter,
    CombinatorialSplit,
    Fold,
    WalkForwardSplitter,
)

__all__ = [
    "Backtest",
    "BacktestResult",
    "Bar",
    "BuyAndHoldStrategy",
    "CSVFormatError",
    "CombinatorialPurgedSplitter",
    "CombinatorialSplit",
    "CsvTailFeed",
    "CrossSectionalMomentumStrategy",
    "CommissionModel",
    "DataHandler",
    "EventType",
    "Feed",
    "FeedError",
    "FillEvent",
    "Financing",
    "FixedBpsSlippage",
    "Fold",
    "HistoricBarDataHandler",
    "LeakReport",
    "LiveDataHandler",
    "LookAheadDataHandler",
    "LookAheadError",
    "LookAheadMomentumStrategy",
    "MarketEvent",
    "MultiAssetDataHandler",
    "MultiAssetPortfolio",
    "OrderEvent",
    "PaperState",
    "PaperTrader",
    "PerShareCommission",
    "PercentOfNotionalCommission",
    "PermanentImpactSlippage",
    "PerformanceMetrics",
    "PollingFeed",
    "Portfolio",
    "PortfolioSnapshot",
    "ReplayFeed",
    "SignalEvent",
    "SimulatedExecutionHandler",
    "SlippageModel",
    "SpreadPlusImpactSlippage",
    "Strategy",
    "StrategySpec",
    "SyntheticConfig",
    "SyntheticSeries",
    "TimeSeriesMomentumStrategy",
    "WalkForwardSplitter",
    "ZeroCommission",
    "ZeroSlippage",
    "align_panel",
    "bars_from_series",
    "compute_metrics",
    "detect_look_ahead",
    "generate_price_series",
    "load_csv_bars",
    "load_csv_panel",
    "param_grid",
    "read_journal",
    "run_backtest",
    "__version__",
]
