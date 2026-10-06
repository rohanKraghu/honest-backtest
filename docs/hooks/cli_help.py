"""Fill the command-line reference from the parsers themselves.

A page containing ``<!-- cli-help: NAME -->`` gets that command's ``--help``
text in its place at build time, so the reference cannot drift from the code.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# argparse wraps help to the terminal; fix the width so builds are identical.
os.environ["COLUMNS"] = "76"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from honest_backtest.cli import build_audit_parser, build_parser  # noqa: E402
from honest_backtest.paper_cli import build_paper_parser  # noqa: E402

_PARSERS = {
    "study": build_parser,
    "audit": build_audit_parser,
    "paper": build_paper_parser,
}
_MARKER = re.compile(r"<!-- cli-help: (\w+) -->")


def _help(match: re.Match) -> str:
    parser = _PARSERS[match.group(1)]()
    parser.prog = {
        "study": "honest-backtest",
        "audit": "honest-backtest audit",
        "paper": "honest-backtest paper",
    }[match.group(1)]
    return "```text\n" + parser.format_help().rstrip() + "\n```"


def on_page_markdown(markdown: str, **kwargs) -> str:
    """Replace each marker with the named command's help text."""
    return _MARKER.sub(_help, markdown)
