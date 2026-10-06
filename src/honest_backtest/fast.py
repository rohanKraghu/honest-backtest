"""A fast path that gives the event engine's answer, bit for bit, in less time.

Most of a ladder's run time goes on work it repeats. The rungs from
"+ point-in-time data" to "+ next-bar execution" replay the same strategy
with the same parameters over the same bars, and only the cost model
changes, yet each run rebuilds the strategy, re-derives every signal and
pushes every event through a thread-safe queue. A strategy's signals cannot
depend on costs (it sees only the data handler), so they can be recorded
once per setting and reused.

This module does exactly that and nothing cleverer:

* :func:`replay_signals` drives the *real* strategy object bar by bar
  through the real data handler, with a plain list in place of the queue.
  No signal is re-implemented, so a strategy cannot behave differently here
  than in the engine, and point-in-time access is enforced the same way.
* :func:`simulate` replays those signals through the portfolio's sizing
  rules and the very same slippage and commission objects, in the order the
  engine would call them, so every float comes out identical.

That equality is not hoped for; ``tests/test_fast.py`` asserts it on equity,
fills and costs for every fill timing and cost model. Liquidity caps, limit
orders and financing are not reproduced here; :func:`supports` says when the
fast path applies, and the ladder falls back to the engine otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .commission import CommissionModel, ZeroCommission
from .data import Bar, HistoricBarDataHandler, LookAheadDataHandler
from .engine import BacktestResult
from .events import FillEvent, MarketEvent, OrderEvent, SignalEvent
from .execution import FILL_TIMINGS
from .portfolio import PortfolioSnapshot
from .slippage import SlippageModel, ZeroSlippage
from .spec import Params, StrategySpec

#: ``(bar timestamp, target weights emitted on that bar, in order)``.
Signals = dict[int, list[float]]


class _ListQueue:
    """The part of :class:`queue.Queue` a strategy uses, without the locks."""

    def __init__(self) -> None:
        self.items: list = []

    def put(self, item, block: bool = True, timeout: float | None = None) -> None:
        self.items.append(item)


def replay_signals(
    bars: Sequence[Bar],
    spec: StrategySpec,
    params: Params,
    *,
    look_ahead: bool = False,
) -> tuple[Signals, bool]:
    """Run only the strategy over ``bars`` and record what it asked for.

    Returns:
        ``(signals, used_look_ahead)``: target weights by bar timestamp, and
        the strategy's own declaration of whether it read the future.
    """
    queue = _ListQueue()
    handler_cls = LookAheadDataHandler if look_ahead else HistoricBarDataHandler
    data = handler_cls(queue, bars, symbol=bars[0].symbol)  # type: ignore[arg-type]
    strategy = spec.factory(params, look_ahead=look_ahead)(queue, data)  # type: ignore[arg-type]
    signals: Signals = {}
    while data.continue_backtest:
        data.update_bars()
        while queue.items:
            event = queue.items.pop(0)
            if isinstance(event, MarketEvent):
                strategy.calculate_signals(event)
            elif isinstance(event, SignalEvent):
                signals.setdefault(event.timestamp, []).append(event.target_weight)
    return signals, strategy.uses_look_ahead


def supports(
    *,
    fill_timing: str = "close",
    max_participation: float | None = None,
    limit_offset_bps: float | None = None,
    limit_expiry_bars: int = 1,
    financing=None,
) -> bool:
    """Whether :func:`simulate` reproduces the engine for these settings."""
    return (
        fill_timing in FILL_TIMINGS
        and max_participation is None
        and limit_offset_bps is None
        and financing is None
    )


def simulate(
    bars: Sequence[Bar],
    signals: Signals,
    *,
    slippage: SlippageModel | None = None,
    commission: CommissionModel | None = None,
    initial_capital: float = 1_000_000.0,
    rebalance_threshold: float = 0.05,
    warmup: int = 0,
    bars_per_year: int = 252,
    fill_timing: str = "close",
    used_look_ahead: bool = False,
) -> BacktestResult:
    """Turn recorded signals into the result :func:`run_backtest` would return.

    Mirrors, step for step, what the engine does on each bar: fill orders
    held from the previous bar (next-bar timings), mark the book, size an
    order for each signal against the book as it stood before this bar's
    fills, then fill those orders (at this close, or hold them).
    """
    if fill_timing not in FILL_TIMINGS:
        raise ValueError(f"fill_timing must be one of {FILL_TIMINGS}")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if slippage is not None:
        slippage.reset()
    slip = slippage or ZeroSlippage()
    comm = commission or ZeroCommission()
    symbol = bars[0].symbol if bars else "SYNTH"

    cash = float(initial_capital)
    position = 0.0
    snapshots: list[PortfolioSnapshot] = []
    fills: list[FillEvent] = []
    totals = {"commission": 0.0, "slippage": 0.0, "notional": 0.0}
    held: OrderEvent | None = None
    cancelled = 0.0

    def fill(order: OrderEvent, bar: Bar, timestamp: int) -> None:
        nonlocal cash, position
        price = slip.fill_price(order, bar)
        event = FillEvent(
            symbol=order.symbol,
            timestamp=timestamp,
            quantity=order.quantity,
            direction=order.direction,
            fill_price=price,
            commission=comm.calculate(order, price),
            reference_price=bar.close,
        )
        cash -= event.signed_quantity * event.fill_price
        cash -= event.commission
        position += event.signed_quantity
        fills.append(event)
        totals["commission"] += event.commission
        totals["slippage"] += event.slippage_cost
        totals["notional"] += abs(event.quantity * event.fill_price)

    for bar in bars:
        if held is not None:
            ref = replace(bar, close=bar.open) if fill_timing == "next_open" else bar
            fill(held, ref, ref.timestamp)
            held = None
        price = bar.close
        snapshots.append(
            PortfolioSnapshot(
                bar.timestamp, cash, position, price, cash + position * price
            )
        )
        orders: list[OrderEvent] = []
        for weight in signals.get(bar.timestamp, ()):
            if price <= 0:
                continue
            equity = cash + position * price
            if equity <= 0:
                continue
            delta = weight * equity / price - position
            if delta == 0.0 or abs(delta * price) < max(
                rebalance_threshold * equity, 1e-9
            ):
                continue
            orders.append(
                OrderEvent(
                    symbol=symbol,
                    timestamp=bar.timestamp,
                    quantity=abs(delta),
                    direction="BUY" if delta > 0 else "SELL",
                )
            )
        if fill_timing == "close":
            for order in orders:
                fill(order, bar, order.timestamp)
        elif orders:
            # A newer order for the instrument cancels an older one.
            cancelled += sum(o.quantity for o in orders[:-1])
            held = orders[-1]

    return BacktestResult(
        timestamps=[s.timestamp for s in snapshots],
        equity=[s.equity for s in snapshots],
        positions=[s.position for s in snapshots],
        initial_capital=float(initial_capital),
        warmup=warmup,
        bars_per_year=bars_per_year,
        total_commission=totals["commission"],
        total_slippage=totals["slippage"],
        traded_notional=totals["notional"],
        n_trades=len(fills),
        event_counts={},
        used_look_ahead=used_look_ahead,
        fills=fills,
        snapshots=snapshots,
        final_cash=cash,
        final_position=position,
        unfilled_quantity=cancelled,
    )


class SignalCache:
    """Signals already replayed, keyed by bars, parameters and leakiness.

    Keys use the identity of the bar list, so a cache must not outlive the
    lists it was filled from; the ladder keeps one per run.
    """

    def __init__(self) -> None:
        """Start empty."""
        self._store: dict[tuple, tuple[Signals, bool]] = {}
        self._keep: list[Sequence[Bar]] = []  # pin the lists so ids stay unique
        self.hits = 0
        self.misses = 0

    def get(
        self, bars: Sequence[Bar], spec: StrategySpec, params: Params, look_ahead: bool
    ) -> tuple[Signals, bool]:
        """Return recorded signals, replaying the strategy only the first time."""
        key = (id(bars), len(bars), spec.name, tuple(params.items()), look_ahead)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        self._keep.append(bars)
        value = replay_signals(bars, spec, params, look_ahead=look_ahead)
        self._store[key] = value
        return value
