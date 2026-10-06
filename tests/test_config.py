"""Settings from a file must mean exactly what the same flags mean.

A config file is only useful as a record of a run if replaying it gives the
run back, so the central check is that a file and the equivalent flags
produce identical output. The rest pins the edges: flags beat the file,
paths are relative to the file, and a bad file fails with a reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from honest_backtest.audit import BUILT_IN_STRATEGIES
from honest_backtest.cli import build_audit_parser, build_parser, main
from honest_backtest.config import (
    ConfigError,
    config_defaults,
    load_config_file,
    parse_args_with_config,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
TINY = ["--bars", "800", "--train-size", "252", "--test-size", "252", "--seeds", "1"]
TINY_CONFIG = {"bars": 800, "train-size": 252, "test_size": 252, "seeds": 1}


def _stable(out: str) -> str:
    """Drop the wall-clock line, the only part of a run that is not reproducible."""
    return "\n".join(line for line in out.splitlines() if "Completed in" not in line)


def _write(path: Path, values: dict) -> Path:
    path.write_text(json.dumps(values))
    return path


def test_a_json_config_reproduces_the_flags_exactly(tmp_path, capsys):
    """Same settings, by file or by flag, print the same report."""
    assert main(TINY + ["--seed", "7"]) == 0
    by_flags = capsys.readouterr().out
    config = _write(tmp_path / "study.json", {**TINY_CONFIG, "seed": 7})
    assert main(["--config", str(config)]) == 0
    assert _stable(capsys.readouterr().out) == _stable(by_flags)


def test_a_yaml_config_reproduces_the_flags_exactly(tmp_path, capsys):
    """YAML is read into the same settings as JSON."""
    config = tmp_path / "study.yaml"
    config.write_text("# a tiny study\nbars: 800\ntrain-size: 252\ntest_size: 252\n")
    args = parse_args_with_config(build_parser(), ["--config", str(config)])
    flags = build_parser().parse_args(["--bars", "800", "--train-size", "252"])
    assert vars(args) | {"config": None} == vars(flags) | {"config": None}


def test_flags_override_the_file(tmp_path):
    """The command line wins; the file fills in only what it leaves out."""
    config = _write(tmp_path / "study.json", {"seed": 1, "bars": 900})
    args = parse_args_with_config(
        build_parser(), ["--config", str(config), "--seed", "5"]
    )
    assert args.seed == 5
    assert args.bars == 900


def test_yaml_without_pyyaml_asks_for_it(tmp_path, monkeypatch, capsys):
    """No silent fallback: a YAML file without PyYAML is a clear error."""
    monkeypatch.setitem(sys.modules, "yaml", None)
    config = tmp_path / "study.yaml"
    config.write_text("seed: 3\n")
    with pytest.raises(ConfigError, match="pip install pyyaml"):
        load_config_file(config)
    with pytest.raises(SystemExit):
        main(["--config", str(config)])
    assert "pip install pyyaml" in capsys.readouterr().err
    # JSON still works without it.
    assert load_config_file(_write(tmp_path / "s.json", {"seed": 3})) == {"seed": 3}


@pytest.mark.parametrize(
    "values, message",
    [
        ({"sed": 1}, "unknown config key 'sed'"),
        ({"seed": "soon"}, "--seed cannot take 'soon'"),
        ({"bars": 1.5}, "--bars cannot take 1.5"),
        ({"markdown": "yes"}, "--markdown is a switch"),
        ({"seed": True}, "--seed takes a value"),
        ({"config": "other.json"}, "unknown config key 'config'"),
    ],
)
def test_bad_values_are_refused_like_bad_flags(tmp_path, values, message):
    """A file cannot set anything the command line would refuse."""
    with pytest.raises(ConfigError, match=message):
        config_defaults(build_parser(), values, base=tmp_path)


def test_audit_choices_and_cpcv_are_checked(tmp_path):
    """Options with choices or their own parser apply them to file values too."""
    parser = build_audit_parser()
    with pytest.raises(ConfigError, match="must be one of"):
        config_defaults(parser, {"fill": "tomorrow"}, base=tmp_path)
    with pytest.raises(ConfigError, match="--cpcv cannot take"):
        config_defaults(parser, {"cpcv": [2, 2]}, base=tmp_path)
    assert config_defaults(parser, {"cpcv": [6, 2]}, base=tmp_path) == {"cpcv": (6, 2)}
    assert config_defaults(parser, {"cpcv": "6,2"}, base=tmp_path) == {"cpcv": (6, 2)}


def test_unreadable_files_are_refused(tmp_path):
    """Missing, misnamed, malformed and non-mapping files fail with a reason."""
    with pytest.raises(ConfigError, match="does not exist"):
        load_config_file(tmp_path / "missing.json")
    (tmp_path / "study.toml").write_text("seed = 1\n")
    with pytest.raises(ConfigError, match=r"\.json, \.yaml or \.yml"):
        load_config_file(tmp_path / "study.toml")
    (tmp_path / "bad.json").write_text("{seed: 1")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config_file(tmp_path / "bad.json")
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="must hold a mapping"):
        load_config_file(tmp_path / "list.yaml")
    (tmp_path / "empty.yaml").write_text("")
    assert load_config_file(tmp_path / "empty.yaml") == {}


def test_paths_in_a_config_are_relative_to_the_file(tmp_path):
    """A config keeps working from any directory; built-in names stay names."""
    sub = tmp_path / "configs"
    sub.mkdir()
    config = _write(
        sub / "audit.json",
        {"data": "prices.csv", "strategy": "mine.py:FAST", "html": "out/r.html"},
    )
    args = parse_args_with_config(build_audit_parser(), ["--config", str(config)])
    assert args.data == str(sub.resolve() / "prices.csv")
    assert args.strategy == f"{sub.resolve() / 'mine.py'}:FAST"
    assert args.html == sub.resolve() / "out" / "r.html"
    builtin = _write(sub / "b.json", {"strategy": "momentum", "data": "/abs/p.csv"})
    args = parse_args_with_config(
        build_audit_parser(), ["--config", str(builtin)], keep_names=BUILT_IN_STRATEGIES
    )
    assert (args.strategy, args.data) == ("momentum", "/abs/p.csv")


def test_the_audit_still_needs_data_and_a_strategy(tmp_path, capsys):
    """Neither the flags nor the file naming them is an error, not a crash."""
    config = _write(tmp_path / "a.json", {"strategy": "momentum"})
    with pytest.raises(SystemExit):
        main(["audit", "--config", str(config)])
    assert "--data is required" in capsys.readouterr().err


def test_the_shipped_audit_config_reproduces_its_flags(capsys):
    """``examples/audit.yaml`` is the README audit plus ``--cpcv 6,2``."""
    argv = [
        "audit",
        "--data",
        str(EXAMPLES / "sample_prices.csv"),
        "--strategy",
        str(EXAMPLES / "sma_crossover.py"),
        "--cpcv",
        "6,2",
        "--no-leak-check",
    ]
    assert main(argv) == 0
    by_flags = capsys.readouterr().out
    # A flag added on top of the file applies to both runs alike.
    argv = ["audit", "--config", str(EXAMPLES / "audit.yaml"), "--no-leak-check"]
    assert main(argv) == 0
    by_file = capsys.readouterr().out
    assert _stable(by_file) == _stable(by_flags)
    assert "Combinatorial purged cross-validation (6 groups" in by_file


def test_the_shipped_study_config_is_the_documented_default():
    """``examples/study.json`` spells out the headline run."""
    args = parse_args_with_config(
        build_parser(), ["--config", str(EXAMPLES / "study.json")]
    )
    defaults = build_parser().parse_args([])
    assert vars(args) | {"config": None} == vars(defaults)
