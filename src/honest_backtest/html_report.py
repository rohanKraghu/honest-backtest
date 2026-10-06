"""A one-file HTML report of a ladder: where the Sharpe went, and where the money went.

The text report is the record; this is the page you send someone. It has
three charts, each answering one question a reader of the table would ask:

* **Sharpe waterfall.** How much of the headline did each removed
  assumption take away? The bars start at the in-sample headline, float down
  (or up) by each rung's change, and end at the out-of-sample number.
* **Equity curves.** What did each rung's account actually do over the
  scored window? Log scale, because a leaky backtest compounds to absurd
  levels and would flatten every honest curve on a linear axis.
* **Cost attribution.** How much went to slippage and how much to
  commission, rung by rung.

The page is self-contained: inline SVG drawn from numpy arrays, inline CSS,
no JavaScript and nothing fetched, so it opens offline and can be attached
to an email or a pull request as it is.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from html import escape

import numpy as np

from .experiments import LadderResult, StudyResult
from .report import _fmt_params, _fmt_probability

#: Series colours, by rung. The last rung (out of sample) always gets the
#: emphasis colour, whatever its index.
_N_COLOURS = 6

_CSS = """
:root {
  --bg: #fbfaf7; --fg: #1d1d1b; --muted: #6b6a66; --rule: #e4e1da;
  --panel: #ffffff; --up: #2f7d4f; --down: #b5452f; --total: #3d5a80;
  --honest: #c2410c; --ref: #8a8780;
  --c1: #7b6fa8; --c2: #3d8fa8; --c3: #4f9a6a; --c4: #b48a2c; --c5: #8a6b5a;
  --c6: #6b8aa3;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161614; --fg: #ecebe6; --muted: #a09e97; --rule: #33322e;
    --panel: #1f1f1c; --up: #5fbf86; --down: #e07a5f; --total: #8fb3de;
    --honest: #fb923c; --ref: #8f8c84;
    --c1: #ab9fd8; --c2: #6fbfd8; --c3: #7fca9a; --c4: #e4ba5c; --c5: #c09b8a;
    --c6: #9bbad3;
  }
}
:root[data-theme="dark"] {
  --bg: #161614; --fg: #ecebe6; --muted: #a09e97; --rule: #33322e;
  --panel: #1f1f1c; --up: #5fbf86; --down: #e07a5f; --total: #8fb3de;
  --honest: #fb923c; --ref: #8f8c84;
  --c1: #ab9fd8; --c2: #6fbfd8; --c3: #7fca9a; --c4: #e4ba5c; --c5: #c09b8a;
  --c6: #9bbad3;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 17px; margin: 40px 0 4px; }
.sub, .note { color: var(--muted); }
.note { font-size: 13px; margin: 4px 0 12px; }
.warn {
  border: 1px solid var(--down); color: var(--down); border-radius: 8px;
  padding: 12px 16px; margin: 16px 0; font-weight: 600;
}
.headline { display: flex; flex-wrap: wrap; gap: 12px; margin: 24px 0 8px; }
.stat {
  flex: 1 1 180px; background: var(--panel); border: 1px solid var(--rule);
  border-radius: 10px; padding: 12px 16px;
}
.stat .k { color: var(--muted); font-size: 13px; }
.stat .v { font-size: 26px; font-variant-numeric: tabular-nums; font-weight: 600; }
.stat.honest .v { color: var(--honest); }
dl.facts {
  display: grid; grid-template-columns: max-content 1fr; gap: 2px 16px;
  margin: 16px 0; font-size: 14px;
}
dl.facts dt { color: var(--muted); }
dl.facts dd { margin: 0; overflow-wrap: anywhere; }
.panel {
  background: var(--panel); border: 1px solid var(--rule); border-radius: 10px;
  padding: 12px;
}
svg { display: block; width: 100%; height: auto; }
svg text { fill: var(--fg); font: 12px system-ui, sans-serif; }
svg .muted { fill: var(--muted); }
svg .axis { stroke: var(--rule); stroke-width: 1; }
svg .zero { stroke: var(--muted); stroke-width: 1; }
svg .ref { stroke: var(--ref); stroke-width: 1.2; stroke-dasharray: 5 4; fill: none; }
svg .up { fill: var(--up); } svg .down { fill: var(--down); }
svg .total { fill: var(--total); } svg .honest { fill: var(--honest); }
svg .slip { fill: var(--down); } svg .comm { fill: var(--c4); }
svg .line { fill: none; stroke-width: 1.6; }
svg .line.honest { stroke: var(--honest); stroke-width: 2.4; fill: none; }
svg .s1 { stroke: var(--c1); } svg .s2 { stroke: var(--c2); }
svg .s3 { stroke: var(--c3); } svg .s4 { stroke: var(--c4); }
svg .s5 { stroke: var(--c5); } svg .s6 { stroke: var(--c6); }
.legend {
  display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; margin: 8px 4px 0;
}
.legend span::before {
  content: ""; display: inline-block; width: 14px; height: 3px; margin-right: 6px;
  vertical-align: middle; background: currentColor;
}
.legend .ref { color: var(--ref); }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--rule); }
th { color: var(--muted); font-weight: 500; }
td:nth-child(2) { white-space: nowrap; }
td.n { text-align: right; font-variant-numeric: tabular-nums; }
tr.honest td { font-weight: 600; }
tr.honest td:nth-child(3) { color: var(--honest); }
ul.notes { padding-left: 20px; }
footer { margin-top: 48px; color: var(--muted); font-size: 13px; }
"""


@dataclass(frozen=True)
class ReportContext:
    """Everything around the ladder that the page should say.

    Attributes:
        title: Page heading.
        subtitle: One line under the heading.
        facts: ``(label, value)`` pairs describing the run.
        references: Named Sharpe levels drawn on the waterfall, such as
            buy-and-hold or the oracle.
        notes: Paragraphs for the "Reading it" section.
        warning: Shown prominently at the top when set.
        param_header: What the chosen-parameters column is called.
        dates: Optional date label for each scored bar, for the x axis.
    """

    title: str
    subtitle: str = ""
    facts: Sequence[tuple[str, str]] = ()
    references: Sequence[tuple[str, float]] = ()
    notes: Sequence[str] = ()
    warning: str | None = None
    param_header: str = "Params"
    dates: Sequence[str] | None = field(default=None)


def _nice_step(span: float, target: int = 5) -> float:
    raw = span / max(target, 1)
    if raw <= 0 or not math.isfinite(raw):
        return 1.0
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _fmt_num(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 100 else f"{v:,.0f}"


def sharpe_waterfall_svg(
    ladder: LadderResult, references: Sequence[tuple[str, float]]
) -> str:
    """Draw the headline Sharpe, each rung's change, and the honest total."""
    sharpes = [s.metrics.sharpe for s in ladder.stages]
    columns: list[tuple[str, float, float, str]] = []  # label, from, to, class
    columns.append((ladder.stages[0].name, 0.0, sharpes[0], "total"))
    for prev, stage in zip(sharpes, ladder.stages[1:], strict=False):
        cur = stage.metrics.sharpe
        columns.append((stage.name, prev, cur, "down" if cur < prev else "up"))
    columns.append(("Out of sample", 0.0, sharpes[-1], "honest"))

    values = [0.0, *sharpes, *(v for _, v in references)]
    lo, hi = min(values), max(values)
    pad = 0.08 * (hi - lo or 1.0)
    lo, hi = lo - pad, hi + pad
    width, height = 900, 360
    left, right, top, bottom = 56, 16, 24, 74
    plot_w, plot_h = width - left - right, height - top - bottom

    def y(v: float) -> float:
        return top + (hi - v) / (hi - lo) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Sharpe waterfall">'
    ]
    step = _nice_step(hi - lo)
    tick = math.ceil(lo / step) * step
    while tick <= hi + 1e-12:
        parts.append(
            f'<line class="axis" x1="{left}" x2="{width - right}" '
            f'y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>'
            f'<text class="muted" x="{left - 8}" y="{y(tick) + 4:.1f}" '
            f'text-anchor="end">{_fmt_num(tick)}</text>'
        )
        tick += step
    parts.append(
        f'<line class="zero" x1="{left}" x2="{width - right}" '
        f'y1="{y(0):.1f}" y2="{y(0):.1f}"/>'
    )

    slot = plot_w / len(columns)
    bar_w = slot * 0.62
    for i, (label, a, b, cls) in enumerate(columns):
        x = left + i * slot + (slot - bar_w) / 2
        y0, y1 = sorted((y(a), y(b)))
        h = max(y1 - y0, 1.0)
        parts.append(
            f'<rect class="{cls}" x="{x:.1f}" y="{y0:.1f}" width="{bar_w:.1f}" '
            f'height="{h:.1f}" rx="2"/>'
        )
        delta = b - a
        if cls in ("total", "honest"):
            text = f"{b:.2f}"
        else:
            text = "0.00" if abs(delta) < 0.005 else f"{delta:+.2f}"
        ty = y0 - 6 if b >= a else y1 + 14
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{ty:.1f}" text-anchor="middle">'
            f"{escape(text)}</text>"
        )
        if i + 1 < len(columns):
            nx = left + (i + 1) * slot + (slot - bar_w) / 2
            parts.append(
                f'<line class="zero" x1="{x + bar_w:.1f}" x2="{nx:.1f}" '
                f'y1="{y(b):.1f}" y2="{y(b):.1f}" stroke-dasharray="2 3"/>'
            )
        words = label.split()
        lines, current = [], ""
        for word in words:
            if len(current) + len(word) > 13 and current:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        lines.append(current)
        for k, line in enumerate(lines[:3]):
            parts.append(
                f'<text class="muted" x="{x + bar_w / 2:.1f}" '
                f'y="{height - bottom + 18 + k * 14}" text-anchor="middle">'
                f"{escape(line)}</text>"
            )
    for name, value in references:
        parts.append(
            f'<line class="ref" x1="{left}" x2="{width - right}" '
            f'y1="{y(value):.1f}" y2="{y(value):.1f}"/>'
            f'<text class="muted" x="{width - right - 4}" y="{y(value) - 5:.1f}" '
            f'text-anchor="end">{escape(name)} {value:.2f}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _downsample(n: int, max_points: int = 500) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def equity_svg(
    ladder: LadderResult, dates: Sequence[str] | None = None
) -> tuple[str, list[tuple[str, str]]]:
    """Draw every rung's growth of one unit on a log axis, with buy-and-hold.

    Returns:
        The SVG, and ``(css class, label)`` pairs for the legend.
    """
    series: list[tuple[str, str, np.ndarray]] = []
    for stage in ladder.stages:
        cls = (
            "line honest"
            if stage.honest
            else f"line s{(stage.index - 1) % _N_COLOURS + 1}"
        )
        growth = np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(stage.returns))])
        series.append((cls, f"{stage.index}. {stage.name}", growth))
    bh = np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(ladder.buy_hold_returns))])
    series.append(("ref", "Buy and hold", bh))

    logs = [np.log10(np.clip(g, 1e-9, None)) for _, _, g in series]
    lo = min(float(v.min()) for v in logs)
    hi = max(float(v.max()) for v in logs)
    if hi - lo < 0.1:
        lo, hi = lo - 0.05, hi + 0.05
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    n = max(len(v) for v in logs)
    width, height = 900, 380
    left, right, top, bottom = 64, 16, 16, 36
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(i: float) -> float:
        return left + i / max(n - 1, 1) * plot_w

    def y(v: float) -> float:
        return top + (hi - v) / (hi - lo) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Equity curves">'
    ]
    for mult in _log_ticks(lo, hi):
        lv = math.log10(mult)
        label = f"{mult:g}x" if mult < 1000 else f"{mult:,.0f}x"
        parts.append(
            f'<line class="{"zero" if mult == 1 else "axis"}" x1="{left}" '
            f'x2="{width - right}" y1="{y(lv):.1f}" y2="{y(lv):.1f}"/>'
            f'<text class="muted" x="{left - 8}" y="{y(lv) + 4:.1f}" '
            f'text-anchor="end">{escape(label)}</text>'
        )
    for k in range(5):
        i = round(k * (n - 1) / 4)
        label = (
            dates[min(i, len(dates) - 1)] if dates else f"bar {ladder.scored_start + i}"
        )
        anchor = "start" if k == 0 else "end" if k == 4 else "middle"
        parts.append(
            f'<text class="muted" x="{x(i):.1f}" y="{height - 12}" '
            f'text-anchor="{anchor}">{escape(label)}</text>'
        )
    # Honest curve last so it is drawn on top.
    order = sorted(range(len(series)), key=lambda j: "honest" in series[j][0])
    for j in order:
        cls, _, _ = series[j]
        values = logs[j]
        idx = _downsample(len(values))
        points = " ".join(f"{x(i):.1f},{y(values[i]):.1f}" for i in idx)
        parts.append(f'<polyline class="{cls}" points="{points}"/>')
    parts.append("</svg>")
    legend = [(cls.replace("line ", ""), label) for cls, label, _ in series]
    return "".join(parts), legend


def _log_ticks(lo: float, hi: float) -> list[float]:
    """Multipliers like 0.5x, 1x, 2x, 10x that fall inside ``[lo, hi]`` (log10)."""
    span = hi - lo
    if span > 2.5:
        mults = [10.0**k for k in range(math.floor(lo), math.ceil(hi) + 1)]
    else:
        base = [1, 2, 5] if span > 0.8 else [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8]
        mults = [
            b * 10.0**k
            for k in range(math.floor(lo) - 1, math.ceil(hi) + 1)
            for b in base
        ]
    return [m for m in mults if lo <= math.log10(m) <= hi]


def costs_svg(ladder: LadderResult) -> str:
    """Draw slippage and commission paid by each rung, stacked."""
    rows = [
        (s.name, s.metrics.slippage_cost, s.metrics.commission_cost)
        for s in ladder.stages
    ]
    biggest = max((a + b for _, a, b in rows), default=0.0) or 1.0
    width, row_h = 900, 34
    left, right, top = 190, 140, 8
    height = top + row_h * len(rows) + 8
    plot_w = width - left - right
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Costs by rung">'
    ]
    for i, (name, slip, comm) in enumerate(rows):
        y = top + i * row_h
        ws = slip / biggest * plot_w
        wc = comm / biggest * plot_w
        parts.append(
            f'<text x="{left - 10}" y="{y + 21}" text-anchor="end">'
            f"{i + 1}. {escape(name)}</text>"
            f'<rect class="slip" x="{left}" y="{y + 7}" width="{ws:.1f}" '
            f'height="20" rx="2"/>'
            f'<rect class="comm" x="{left + ws:.1f}" y="{y + 7}" width="{wc:.1f}" '
            f'height="20" rx="2"/>'
            f'<text class="muted" x="{left + ws + wc + 8:.1f}" y="{y + 21}">'
            f"{slip + comm:,.0f}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _table(ladder: LadderResult, param_header: str) -> str:
    head = [
        "#",
        "Stage",
        "Sharpe",
        "P(edge)",
        "Total return",
        "Ann. return",
        "Max DD",
        "Trades",
        "Turnover",
        param_header,
    ]
    rows = []
    for s in ladder.stages:
        m = s.metrics
        cells = [
            (str(s.index), "n"),
            (s.name, ""),
            (f"{m.sharpe:.2f}", "n"),
            (_fmt_probability(s.p_edge), "n"),
            (f"{m.total_return * 100:+.1f}%", "n"),
            (f"{m.annual_return * 100:+.1f}%", "n"),
            (f"{m.max_drawdown * 100:.1f}%", "n"),
            (f"{m.n_trades:,}", "n"),
            (f"{m.annual_turnover:.1f}x/yr", "n"),
            (_fmt_params(s.chosen_params), ""),
        ]
        tds = "".join(
            f'<td class="{c}">{escape(v)}</td>' if c else f"<td>{escape(v)}</td>"
            for v, c in cells
        )
        rows.append(f'<tr class="{"honest" if s.honest else ""}">{tds}</tr>')
    ths = "".join(f"<th>{escape(h)}</th>" for h in head)
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{ths}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_html(ladder: LadderResult, context: ReportContext) -> str:
    """Render a complete, self-contained HTML page for a ladder."""
    first, last = ladder.stages[0], ladder.stages[-1]
    equity, legend = equity_svg(ladder, context.dates)
    stats = [
        ("In-sample headline", f"{first.metrics.sharpe:.2f}", ""),
        *((name, f"{value:.2f}", "") for name, value in context.references),
        ("Out of sample", f"{last.metrics.sharpe:.2f}", "honest"),
        ("P(edge), out of sample", _fmt_probability(last.p_edge), "honest"),
    ]
    stat_html = "".join(
        f'<div class="stat {cls}"><div class="k">{escape(k)}</div>'
        f'<div class="v">{escape(v)}</div></div>'
        for k, v, cls in stats
    )
    facts = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in context.facts)
    legend_html = "".join(
        f'<span class="{"ref" if c == "ref" else ""}" style="color: var(--'
        f'{"honest" if c == "honest" else "ref" if c == "ref" else "c" + c[1:]})">'
        f"{escape(label)}</span>"
        for c, label in legend
    )
    notes = "".join(f"<li>{escape(n)}</li>" for n in context.notes)
    warning = (
        f'<div class="warn">{escape(context.warning)}</div>' if context.warning else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(context.title)}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<h1>{escape(context.title)}</h1>
<div class="sub">{escape(context.subtitle)}</div>
{warning}
<div class="headline">{stat_html}</div>
<dl class="facts">{facts}</dl>

<h2>Where the Sharpe went</h2>
<p class="note">Each bar is the change from removing one comfortable assumption.
Every rung is scored over the same bars.</p>
<div class="panel">{sharpe_waterfall_svg(ladder, context.references)}</div>

<h2>What each rung's account did</h2>
<p class="note">Growth of one unit over the scored window, log scale.
The dashed line is buy and hold paying the same costs.</p>
<div class="panel">{equity}<div class="legend">{legend_html}</div></div>

<h2>Where the money went</h2>
<p class="note">Slippage (spread and impact) and commission paid over the
scored window, in account currency.</p>
<div class="panel">{costs_svg(ladder)}
<div class="legend"><span style="color: var(--down)">Slippage</span>
<span style="color: var(--c4)">Commission</span></div></div>

<h2>The ladder</h2>
{_table(ladder, context.param_header)}

<h2>Reading it</h2>
<ul class="notes">{notes}</ul>

<footer>Generated by honest-backtest. Self-contained: no scripts, nothing
fetched.</footer>
</main>
</body>
</html>
"""


def study_html(result: StudyResult) -> str:
    """The synthetic study as a page, with the oracle as a reference level."""
    cfg = result.config
    n_scored = result.scored_end - result.scored_start
    context = ReportContext(
        title="honest-backtest degradation study",
        subtitle="Synthetic prices. A demonstration of method, not a claim of profit.",
        facts=[
            ("Seed", str(cfg.seed)),
            (
                "Scored window",
                f"bars {result.scored_start} to {result.scored_end} "
                f"({n_scored / cfg.synthetic.bars_per_year:.1f} years)",
            ),
            (
                "Walk-forward folds",
                f"{len(result.folds)} (train {cfg.train_size}, test {cfg.test_size})",
            ),
            ("Lookbacks tried", ", ".join(str(v) for v in cfg.lookback_grid)),
        ],
        references=[
            ("Oracle", result.oracle_sharpe),
            ("Buy and hold", result.buy_hold_sharpe),
        ],
        notes=[
            f"The oracle knows the hidden state and pays nothing; it reaches "
            f"Sharpe {result.oracle_sharpe:.2f}. Any rung above that is reading "
            "its own answer.",
            "P(edge) is the Deflated Sharpe Ratio for in-sample rungs (it discounts "
            "for picking the best lookback) and the Probabilistic Sharpe Ratio "
            "against zero for the walk-forward rung.",
            "Only the last rung chose nothing on the bars it reports, so it is the "
            "only number worth quoting.",
        ],
        param_header="Lookback",
    )
    return render_html(result, context)
