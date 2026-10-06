"""Results as JSON, for anything that should read them instead of a person.

``--json PATH`` on ``run_experiment.py`` and on ``honest-backtest audit``
writes everything the text report shows, at full precision: the settings,
every rung's metrics and P(edge), the parameters each fit chose, the costs,
buy and hold, and, for the synthetic study, the oracle. An audit adds the
look-ahead check and, when it was run, the cross-validation distribution.

Values the text report prints as ``-`` (an undefined P(edge), say) are
``null`` here rather than ``NaN``, because ``NaN`` is not JSON and strict
parsers refuse it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass
from typing import Any

import numpy as np

from .audit import AuditResult
from .experiments import CPCVResult, LadderResult, SeedSweep, StageResult, StudyResult
from .leaks import LeakReport
from .metrics import PerformanceMetrics
from .spec import format_params

#: Bumped when a field is renamed or removed, so readers can tell.
SCHEMA_VERSION = 1


def _plain(value: Any) -> Any:
    """Convert a value to JSON-safe builtins; non-finite floats become ``None``."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    return str(value)


def _metrics(m: PerformanceMetrics) -> dict[str, Any]:
    return asdict(m)


def _stage(stage: StageResult) -> dict[str, Any]:
    return {
        "index": stage.index,
        "name": stage.name,
        "assumption_removed": stage.assumption_removed,
        "honest": stage.honest,
        "p_edge": stage.p_edge,
        "chosen_params": [dict(p) for p in stage.chosen_params],
        "chosen_params_text": [format_params(p) for p in stage.chosen_params],
        "metrics": _metrics(stage.metrics),
    }


def ladder_dict(ladder: LadderResult) -> dict[str, Any]:
    """Every rung of a ladder, with the shared window and folds."""
    return {
        "scored_start": ladder.scored_start,
        "scored_end": ladder.scored_end,
        "folds": [
            {
                "index": f.index,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "warmup": f.warmup,
            }
            for f in ladder.folds
        ],
        "stages": [_stage(s) for s in ladder.stages],
        "honest_sharpe": ladder.honest_sharpe,
        "buy_and_hold": {
            "sharpe": ladder.buy_hold_sharpe,
            "same_window_same_costs": True,
        },
        "monotone": ladder.is_monotone,
        "out_of_sample_events": dict(ladder.n_events),
    }


def sweep_dict(sweep: SeedSweep) -> dict[str, Any]:
    """The multi-seed robustness check."""
    stages = []
    for i, name in enumerate(sweep.stage_names):
        mean, sd, t = sweep.summary(i)
        stages.append(
            {"name": name, "sharpes": sweep.sharpes[i], "mean": mean, "stdev": sd, "t": t}
        )
    return {
        "seeds": sweep.seeds,
        "stages": stages,
        "monotone": sweep.monotone_flags,
        "n_monotone": sweep.n_monotone,
    }


def cpcv_dict(cpcv: CPCVResult) -> dict[str, Any]:
    """The cross-validation settings, splits and path Sharpe distribution."""
    q1, q3 = cpcv.quartiles
    return {
        "n_groups": cpcv.n_groups,
        "n_test_groups": cpcv.n_test_groups,
        "purge": cpcv.purge,
        "embargo": cpcv.embargo,
        "start": cpcv.start,
        "end": cpcv.end,
        "groups": [list(g) for g in cpcv.groups],
        "splits": [
            {
                "test_groups": list(s.test_groups),
                "train_segments": [list(t) for t in s.train_segments],
                "chosen_params": dict(p),
            }
            for s, p in zip(cpcv.splits, cpcv.chosen_params, strict=True)
        ],
        "paths": [[{"group": g, "split": s} for g, s in path] for path in cpcv.paths],
        "path_sharpes": cpcv.path_sharpes,
        "median": cpcv.median,
        "quartiles": [q1, q3],
        "min": float(np.min(cpcv.path_sharpes)),
        "max": float(np.max(cpcv.path_sharpes)),
        "n_below_zero": cpcv.n_below_zero,
        "share_below_zero": cpcv.share_below_zero,
    }


def leak_dict(report: LeakReport | None) -> dict[str, Any]:
    """The look-ahead check, or a record that it was skipped."""
    if report is None:
        return {"ran": False}
    return {
        "ran": True,
        "passed": report.passed,
        "summary": report.summary(),
        "cuts": list(report.cuts),
        "settings_checked": [dict(p) for p in report.settings_checked],
        "findings": [
            {
                "params": dict(f.params),
                "cut": f.cut,
                "timestamp": f.timestamp,
                "detail": f.detail,
            }
            for f in report.findings
        ],
        "nondeterministic": [dict(p) for p in report.nondeterministic],
    }


def study_dict(result: StudyResult, sweep: SeedSweep | None = None) -> dict[str, Any]:
    """The synthetic study for one seed, and the sweep if it was run."""
    cfg = result.config
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "study",
        "settings": {
            **{
                f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name != "synthetic"
            },
            "synthetic": asdict(cfg.synthetic),
            "fill_timing": cfg.settings().fill_timing,
        },
        "costs": {
            "slippage": {
                "model": "spread plus square-root impact",
                "half_spread_bps": cfg.half_spread_bps,
                "impact_coefficient": cfg.impact_coefficient,
            },
            "commission": {
                "model": "per share",
                "per_share": cfg.commission_per_share,
                "minimum": cfg.commission_minimum,
            },
        },
        "ladder": ladder_dict(result),
        "oracle_sharpe": result.oracle_sharpe,
        "sweep": sweep_dict(sweep) if sweep is not None else None,
    }


def audit_dict(result: AuditResult) -> dict[str, Any]:
    """An audit: data, strategy, settings, costs, leak check, ladder, CPCV."""
    cfg = result.config
    bars = result.bars

    def when(i: int) -> str | int:
        bar = bars[i]
        return bar.time.date().isoformat() if bar.time is not None else bar.timestamp

    slippage = cfg.slippage(bars)
    commission: dict[str, Any] = (
        {"model": "percent of notional", "bps": cfg.commission_bps}
        if cfg.commission_bps is not None
        else {
            "model": "per share",
            "per_share": cfg.commission_per_share,
            "minimum": cfg.commission_minimum,
        }
    )
    ladder = result.ladder
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "audit",
        "strategy": {
            "name": result.spec.name,
            "grid": [dict(p) for p in result.spec.grid],
            "warmup": result.spec.warmup,
            "has_look_ahead_twin": result.spec.build_look_ahead is not None,
        },
        "data": {
            "symbol": bars[0].symbol,
            "n_bars": len(bars),
            "first": when(0),
            "last": when(-1),
            "scored_first": when(ladder.scored_start),
            "scored_last": when(ladder.scored_end - 1),
            "years_scored": result.years_scored,
        },
        "settings": asdict(cfg.settings),
        "costs": {
            "slippage": {
                "model": "spread plus square-root impact",
                "half_spread_bps": cfg.half_spread_bps,
                "impact_coefficient": cfg.impact_coefficient,
                "bar_volatility": slippage.bar_volatility,
                "calibrated_on_first_bars": cfg.settings.train_size,
            },
            "commission": commission,
        },
        "leak_check": leak_dict(result.leaks),
        "ladder": ladder_dict(ladder),
        "honest_t_stat": result.honest_t_stat,
        "cpcv": cpcv_dict(result.cpcv) if result.cpcv is not None else None,
    }


def to_json(document: Mapping[str, Any] | Sequence[Any]) -> str:
    """Serialise a results document as strict, indented JSON."""
    return json.dumps(_plain(document), indent=2, allow_nan=False) + "\n"
