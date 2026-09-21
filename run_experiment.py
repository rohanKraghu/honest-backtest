#!/usr/bin/env python3
"""Reproduce the headline degradation table.

This is the documented entry point: it works from a fresh clone with nothing
installed beyond ``requirements.txt``. The implementation lives in
``honest_backtest.cli`` so that the installed ``honest-backtest`` console
script runs identical code.

Usage:
    python run_experiment.py                 # the headline result, ~32s
    python run_experiment.py --seeds 1       # headline table only, ~2s
    python run_experiment.py --seed 42       # a different price path
    python run_experiment.py --markdown      # emit the README table
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python run_experiment.py` from a fresh clone without installing.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from honest_backtest.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
