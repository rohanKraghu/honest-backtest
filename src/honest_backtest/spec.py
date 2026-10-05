"""Describing a strategy so the degradation ladder can run on it.

The ladder needs three things from a strategy, and nothing else: a way to
build a fresh instance for given parameters, the parameter grid it is allowed
to search, and how much history it needs before its first signal. A
:class:`StrategySpec` bundles exactly those, so the same study that exposes
the built-in momentum strategy can be pointed at anyone's.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from queue import Queue
from typing import Any

from .data import DataHandler
from .strategy import Strategy

#: ``build(events, data, symbol, **params) -> Strategy``.
StrategyBuilder = Callable[..., Strategy]

Params = Mapping[str, Any]


def param_grid(**axes: Sequence[Any]) -> list[dict[str, Any]]:
    """Return the cartesian product of the given parameter axes.

    ``param_grid(fast=(5, 10), slow=(50, 100))`` gives four dictionaries, in
    a fixed order. The order matters: when two settings tie on Sharpe the
    first one wins, and a fixed order keeps the result reproducible.
    """
    if not axes:
        return [{}]
    names = list(axes)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*axes.values())
    ]


@dataclass(frozen=True)
class StrategySpec:
    """Everything the ladder needs to know about one strategy.

    Attributes:
        name: Label for reports.
        build: ``build(events, data, symbol, **params)`` returning a fresh,
            stateless :class:`~honest_backtest.strategy.Strategy`. Called once
            per backtest, so no state leaks between runs or folds.
        grid: Parameter settings the in-sample fit may choose between. Every
            setting tried counts as a test, so a bigger grid fits more noise.
        warmup: Bars of history the strategy needs before it can trade.
            Walk-forward folds replay this many bars before each test window.
        build_look_ahead: Optional leaky twin of ``build`` that reads the
            future. Only the built-in demonstration has one; when present the
            ladder gains a first "naive backtest" rung that shows the damage.
    """

    name: str
    build: StrategyBuilder
    grid: Sequence[Params]
    warmup: int = 0
    build_look_ahead: StrategyBuilder | None = None

    def __post_init__(self) -> None:
        """Reject a spec the ladder cannot run."""
        if not self.grid:
            raise ValueError(f"strategy {self.name!r} has an empty parameter grid")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")

    def factory(
        self, params: Params, *, look_ahead: bool = False
    ) -> Callable[[Queue, DataHandler], Strategy]:
        """Return a ``(events, data) -> Strategy`` factory for one setting."""
        builder = self.build_look_ahead if look_ahead else self.build
        if builder is None:
            raise ValueError(f"strategy {self.name!r} has no look-ahead variant")

        def make(events: Queue, data: DataHandler) -> Strategy:
            return builder(events, data, data.symbols[0], **params)

        return make


def format_params(params: Params) -> str:
    """Render one parameter setting compactly: a bare value if there is one key."""
    if len(params) == 1:
        return str(next(iter(params.values())))
    return ",".join(f"{k}={v}" for k, v in params.items())
