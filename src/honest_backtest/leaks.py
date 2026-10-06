"""Catching look-ahead in a strategy by changing the future and watching the past.

The point-in-time handler stops a strategy that plays by its rules from
seeing tomorrow. It cannot stop one that goes around them: reaching into the
handler's private list, standardising with statistics a helper computed over
the whole file, or caching something between runs. Code review finds some of
these. A test finds the rest.

The test is the definition of look-ahead turned into an experiment. A
decision made at bar ``t`` may depend only on bars up to ``t``, so if every
bar after ``t`` is replaced by a different, equally plausible future, every
signal up to ``t`` must come out exactly the same. :func:`detect_look_ahead`
does that at several cut points and reports the first signal that moved.

What it cannot see is a strategy that keeps its own copy of the data (say,
by reading the same CSV itself), because the perturbed future never reaches
that copy. The point-in-time handler is the only door the check can watch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from queue import Empty, Queue

import numpy as np

from .data import Bar, HistoricBarDataHandler, LookAheadDataHandler, LookAheadError
from .events import MarketEvent, SignalEvent
from .spec import Params, StrategySpec, format_params

#: One emitted signal: ``(bar timestamp, target weight)``.
Signal = tuple[int, float]


def record_signals(
    bars: Sequence[Bar],
    spec: StrategySpec,
    params: Params,
    *,
    look_ahead: bool = False,
) -> list[Signal]:
    """Replay ``bars`` through one strategy and return every signal it emits.

    Only the strategy and the data handler run. Fills, costs and the
    portfolio cannot change what a strategy decides (it sees nothing but the
    handler), so leaving them out makes each replay cheap.
    """
    events: Queue = Queue()
    handler_cls = LookAheadDataHandler if look_ahead else HistoricBarDataHandler
    symbol = bars[0].symbol if bars else "SYNTH"
    data = handler_cls(events, bars, symbol=symbol)
    strategy = spec.factory(params, look_ahead=look_ahead)(events, data)
    signals: list[Signal] = []
    while data.continue_backtest:
        data.update_bars()
        while True:
            try:
                event = events.get(block=False)
            except Empty:
                break
            if isinstance(event, MarketEvent):
                strategy.calculate_signals(event)
            elif isinstance(event, SignalEvent):
                signals.append((event.timestamp, event.target_weight))
    return signals


def perturb_future(bars: Sequence[Bar], cut: int, seed: int) -> list[Bar]:
    """Return ``bars`` with every bar after index ``cut`` replaced.

    The new future starts from the real close at ``cut`` and follows an
    independent random walk with the series' own volatility, with fresh
    volumes too, so it is plausible but shares nothing with the real one.
    Timestamps and dates are kept so only prices and volumes differ.
    """
    bars = list(bars)
    tail = len(bars) - cut - 1
    if tail <= 0:
        return bars
    closes = np.array([b.close for b in bars], dtype=float)
    vol = float(np.diff(np.log(closes)).std()) if closes.size > 2 else 0.01
    vol = vol if math.isfinite(vol) and vol > 0 else 0.01
    rng = np.random.default_rng(seed)
    # A drift of several sigma per bar guarantees the future differs from
    # the real one by far more than rounding, whatever the real path did.
    steps = rng.normal(3.0 * vol, 2.0 * vol, tail)
    path = bars[cut].close * np.exp(np.cumsum(steps))
    volume_scale = rng.uniform(0.3, 3.0, tail)
    out = bars[: cut + 1]
    previous = bars[cut].close
    for i, close in enumerate(path):
        old = bars[cut + 1 + i]
        out.append(
            replace(
                old,
                open=previous,
                high=max(previous, float(close)),
                low=min(previous, float(close)),
                close=float(close),
                volume=old.volume * float(volume_scale[i]) + 1.0,
            )
        )
        previous = float(close)
    return out


@dataclass(frozen=True)
class LeakFinding:
    """One place where changing the future changed a past decision.

    Attributes:
        params: The parameter setting that leaked.
        cut: Index of the last unchanged bar.
        timestamp: Timestamp of the first signal that moved, or ``None`` if
            the replay with a different future raised instead.
        original: The signal ``(timestamp, weight)`` on the real data, if any.
        perturbed: The signal at the same position with a different future.
        detail: A human-readable explanation.
    """

    params: Params
    cut: int
    timestamp: int | None
    original: Signal | None
    perturbed: Signal | None
    detail: str


@dataclass(frozen=True)
class LeakReport:
    """The result of :func:`detect_look_ahead`.

    Attributes:
        cuts: Bar indices where the future was replaced.
        settings_checked: Parameter settings replayed.
        findings: Every divergence found; empty means no leak was detected.
        nondeterministic: Settings whose signals differed between two runs
            on identical data. Those cannot be checked this way, because a
            divergence would prove nothing.
    """

    cuts: tuple[int, ...]
    settings_checked: tuple[Params, ...]
    findings: tuple[LeakFinding, ...] = ()
    nondeterministic: tuple[Params, ...] = field(default=())

    @property
    def passed(self) -> bool:
        """No leak found and every setting was checkable."""
        return not self.findings and not self.nondeterministic

    def summary(self) -> str:
        """One line for a report header."""
        scope = f"{len(self.cuts)} cut points x {len(self.settings_checked)} settings"
        if self.findings:
            first = self.findings[0]
            return f"LEAK FOUND ({scope}): {first.detail}"
        if self.nondeterministic:
            return (
                f"not checkable ({scope}): signals differ between identical runs "
                f"for {format_params(self.nondeterministic[0])}"
            )
        return f"passed ({scope}): no signal depended on a later bar"


def _first_divergence(
    original: list[Signal], perturbed: list[Signal], cut_timestamp: int
) -> tuple[Signal | None, Signal | None] | None:
    before = [s for s in original if s[0] <= cut_timestamp]
    after = [s for s in perturbed if s[0] <= cut_timestamp]
    for a, b in zip(before, after, strict=False):
        if a[0] != b[0] or not math.isclose(a[1], b[1], rel_tol=1e-12, abs_tol=1e-12):
            return a, b
    if len(before) != len(after):
        n = min(len(before), len(after))
        return (
            before[n] if n < len(before) else None,
            after[n] if n < len(after) else None,
        )
    return None


def default_cuts(n_bars: int, warmup: int, n_cuts: int) -> tuple[int, ...]:
    """Evenly spaced cut points after the warm-up and before the last bar."""
    lo = min(max(warmup, 1), n_bars - 2)
    hi = n_bars - 2
    if hi < lo or n_cuts <= 0:
        return ()
    cuts = np.linspace(lo, hi, num=n_cuts + 2)[1:-1] if hi > lo else np.array([lo])
    return tuple(sorted({int(round(c)) for c in cuts}))


def detect_look_ahead(
    bars: Sequence[Bar],
    spec: StrategySpec,
    *,
    settings: Sequence[Params] | None = None,
    n_cuts: int = 5,
    seed: int = 0,
    look_ahead: bool = False,
) -> LeakReport:
    """Check that no signal of ``spec`` depends on a bar after it.

    Args:
        bars: The data to check on; the audit passes the user's file.
        spec: The strategy.
        settings: Parameter settings to check; every setting in the grid by
            default, since a leak can hide in one branch of the code.
        n_cuts: How many cut points to try per setting.
        seed: Seed for the replacement futures.
        look_ahead: Build the spec's deliberately leaky twin and give it the
            leaky handler. Only useful to demonstrate the check working.

    Returns:
        A :class:`LeakReport`; ``report.passed`` is the verdict.
    """
    bars = list(bars)
    settings = tuple(settings if settings is not None else spec.grid)
    cuts = default_cuts(len(bars), spec.warmup, n_cuts)
    findings: list[LeakFinding] = []
    nondeterministic: list[Params] = []

    for params in settings:
        try:
            original = record_signals(bars, spec, params, look_ahead=look_ahead)
        except LookAheadError as exc:
            # The handler caught the strategy asking for the future outright.
            findings.append(
                LeakFinding(
                    params, -1, None, None, None, f"the data handler refused: {exc}"
                )
            )
            continue
        if record_signals(bars, spec, params, look_ahead=look_ahead) != original:
            nondeterministic.append(params)
            continue
        for k, cut in enumerate(cuts):
            changed = perturb_future(bars, cut, seed=seed * 1000 + k)
            try:
                perturbed = record_signals(changed, spec, params, look_ahead=look_ahead)
            except LookAheadError as exc:  # pragma: no cover - raised above first
                findings.append(
                    LeakFinding(
                        params, cut, None, None, None, f"the data handler refused: {exc}"
                    )
                )
                break
            diverged = _first_divergence(original, perturbed, bars[cut].timestamp)
            if diverged is None:
                continue
            a, b = diverged
            ts = (a or b)[0]  # type: ignore[index]
            findings.append(
                LeakFinding(
                    params,
                    cut,
                    ts,
                    a,
                    b,
                    f"with {format_params(params)}, the signal at bar {ts} changed "
                    f"when only bars after {bars[cut].timestamp} were altered "
                    f"({_fmt_signal(a)} became {_fmt_signal(b)})",
                )
            )
            break  # one finding per setting is enough to fail it

    return LeakReport(
        cuts=cuts,
        settings_checked=settings,
        findings=tuple(findings),
        nondeterministic=tuple(nondeterministic),
    )


def _fmt_signal(signal: Signal | None) -> str:
    return "no signal" if signal is None else f"weight {signal[1]:+.3f}"
