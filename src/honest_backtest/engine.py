"""The event loop.

This is the architectural core. A vectorised backtest computes signals for
every bar at once and then multiplies by returns; that is fast, and it is
also why look-ahead bias is so easy to introduce and so hard to see. Here
nothing is computed "for every bar at once". A bar arrives, and the
components react in causal order::

    MarketEvent -> (strategy) -> SignalEvent
                -> (portfolio) -> OrderEvent
                -> (execution) -> FillEvent
                -> (portfolio)

The inner ``while`` loop drains the queue completely before the next bar is
fetched, so an event can spawn further events within the same bar without
the engine advancing time underneath it.

The cost is speed: about 26 ms per thousand bars here, some two orders of
magnitude slower than the vectorised equivalent. That trade is worth making
because the same strategy object, unchanged, could be driven by a live feed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Callable

import numpy as np

from .commission import CommissionModel, ZeroCommission
from .data import Bar, DataHandler, HistoricBarDataHandler, LookAheadDataHandler
from .events import EventType, FillEvent, MarketEvent, OrderEvent, SignalEvent
from .execution import SimulatedExecutionHandler
from .metrics import PerformanceMetrics, compute_metrics, simple_returns
from .portfolio import Portfolio, PortfolioSnapshot
from .slippage import SlippageModel, ZeroSlippage
from .strategy import Strategy

StrategyFactory = Callable[[Queue, DataHandler], Strategy]


@dataclass
class BacktestResult:
    """Everything one run produced.

    Attributes:
        timestamps: Bar index at each equity point.
        equity: Equity curve, one point per bar.
        positions: Position held into each bar.
        initial_capital: Starting cash.
        warmup: Index of the first scored bar. Bars before it are simulated
            but not scored -- walk-forward folds need history before the test
            window so the strategy's trailing statistics are warm. The bar at
            ``warmup - 1`` supplies the base level for the first return.
        bars_per_year: Annualisation factor.
        total_commission: Commission paid across the run.
        total_slippage: Implicit cost paid across the run.
        traded_notional: Gross notional traded.
        n_trades: Number of fills.
        event_counts: Events processed, by type. Proof that the loop ran.
        used_look_ahead: Whether the strategy knowingly read future data.
        fills: The trade blotter, in execution order.
        snapshots: The full book state at each bar, for auditing the
            accounting identity. Note that a snapshot is taken when the bar
            arrives, i.e. *before* that bar's own fills; the last snapshot
            therefore predates the last fill. That ordering is deliberate --
            it is what makes costs land in the following period's return --
            and ``final_cash`` / ``final_position`` give the true end state.
        final_cash: Cash after every fill, including the last bar's.
        final_position: Units held after every fill.
    """

    timestamps: list[int]
    equity: list[float]
    positions: list[float]
    initial_capital: float
    warmup: int = 0
    bars_per_year: int = 252
    total_commission: float = 0.0
    total_slippage: float = 0.0
    traded_notional: float = 0.0
    n_trades: int = 0
    event_counts: dict[str, int] = field(default_factory=dict)
    used_look_ahead: bool = False
    fills: list[FillEvent] = field(default_factory=list)
    snapshots: list[PortfolioSnapshot] = field(default_factory=list)
    final_cash: float = 0.0
    final_position: float = 0.0

    @property
    def scored_equity(self) -> np.ndarray:
        """The part of the equity curve that counts towards performance.

        The slice starts one bar *before* the scored window so that the first
        scored return is the one realised on the first scored bar. Without
        that, every walk-forward fold would silently discard its opening bar
        and the out-of-sample window would not line up with the in-sample one
        it is being compared against.
        """
        start = max(0, self.warmup - 1)
        return np.asarray(self.equity[start:], dtype=float)

    def returns(self) -> np.ndarray:
        """Per-bar simple returns over the scored window."""
        return simple_returns(self.scored_equity)

    def metrics(self) -> PerformanceMetrics:
        """Performance statistics over the scored window."""
        return compute_metrics(
            self.scored_equity,
            bars_per_year=self.bars_per_year,
            n_trades=self.n_trades,
            commission_cost=self.total_commission,
            slippage_cost=self.total_slippage,
            traded_notional=self.traded_notional,
        )


class Backtest:
    """Drives the event loop over a data handler until the data runs out.

    Args:
        data: The market data handler.
        strategy: Signal generator.
        portfolio: Book keeper and order sizer.
        execution: Fill simulator.
        events: The shared event queue.
        warmup: Equity points to exclude from scoring.
        bars_per_year: Annualisation factor.
    """

    def __init__(
        self,
        data: DataHandler,
        strategy: Strategy,
        portfolio: Portfolio,
        execution: SimulatedExecutionHandler,
        events: Queue,
        warmup: int = 0,
        bars_per_year: int = 252,
    ) -> None:
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution
        self.events = events
        self.warmup = int(warmup)
        self.bars_per_year = int(bars_per_year)
        self.event_counts: dict[str, int] = {t.value: 0 for t in EventType}

    def run(self) -> BacktestResult:
        """Run to completion and return the result.

        The outer loop advances market time exactly once per iteration. The
        inner loop drains every event that the new bar caused, so time cannot
        advance in the middle of a decision.
        """
        while self.data.continue_backtest:
            self.data.update_bars()
            if not self.data.continue_backtest:
                break

            while True:
                try:
                    event = self.events.get(block=False)
                except Empty:
                    break

                self.event_counts[event.type.value] += 1

                if isinstance(event, MarketEvent):
                    self.portfolio.on_market(event)
                    self.strategy.calculate_signals(event)
                elif isinstance(event, SignalEvent):
                    self.portfolio.on_signal(event)
                elif isinstance(event, OrderEvent):
                    self.execution.execute_order(event)
                elif isinstance(event, FillEvent):
                    self.portfolio.on_fill(event)
                else:  # pragma: no cover - defensive
                    raise TypeError(f"unhandled event type: {event!r}")

        return BacktestResult(
            timestamps=self.portfolio.timestamps,
            equity=self.portfolio.equity_curve,
            positions=self.portfolio.positions,
            initial_capital=self.portfolio.initial_capital,
            warmup=self.warmup,
            bars_per_year=self.bars_per_year,
            total_commission=self.portfolio.total_commission,
            total_slippage=self.portfolio.total_slippage,
            traded_notional=self.portfolio.traded_notional,
            n_trades=self.portfolio.n_trades,
            event_counts=dict(self.event_counts),
            used_look_ahead=self.strategy.uses_look_ahead,
            fills=list(self.portfolio.fills),
            snapshots=list(self.portfolio.history),
            final_cash=self.portfolio.cash,
            final_position=self.portfolio.position,
        )


def run_backtest(
    bars: list[Bar],
    strategy_factory: StrategyFactory,
    *,
    slippage: SlippageModel | None = None,
    commission: CommissionModel | None = None,
    initial_capital: float = 1_000_000.0,
    rebalance_threshold: float = 0.05,
    warmup: int = 0,
    bars_per_year: int = 252,
    allow_look_ahead: bool = False,
    symbol: str = "SYNTH",
) -> BacktestResult:
    """Wire up one backtest and run it.

    Args:
        bars: Bars to replay.
        strategy_factory: Callable ``(queue, data_handler) -> Strategy``. A
            factory rather than an instance, because the strategy needs the
            queue and handler that this particular run creates, and because
            the walk-forward study must build a fresh, stateless strategy per
            fold.
        slippage: Slippage model; ``None`` means zero slippage.
        commission: Commission model; ``None`` means zero commission.
        initial_capital: Starting cash.
        rebalance_threshold: No-trade band as a fraction of equity.
        warmup: Leading equity points excluded from scoring.
        bars_per_year: Annualisation factor.
        allow_look_ahead: If ``True``, use the deliberately broken
            :class:`~honest_backtest.data.LookAheadDataHandler`. Only stage 1
            of the degradation study sets this.
        symbol: Instrument name.

    Returns:
        The :class:`BacktestResult`.
    """
    events: Queue = Queue()
    handler_cls = LookAheadDataHandler if allow_look_ahead else HistoricBarDataHandler
    data = handler_cls(events, bars, symbol=symbol)
    strategy = strategy_factory(events, data)
    portfolio = Portfolio(
        events=events,
        data=data,
        symbol=symbol,
        initial_capital=initial_capital,
        rebalance_threshold=rebalance_threshold,
    )
    execution = SimulatedExecutionHandler(
        events=events,
        data=data,
        slippage=slippage or ZeroSlippage(),
        commission=commission or ZeroCommission(),
    )
    return Backtest(
        data=data,
        strategy=strategy,
        portfolio=portfolio,
        execution=execution,
        events=events,
        warmup=warmup,
        bars_per_year=bars_per_year,
    ).run()
