"""Study and audit settings from a file, so a run can be written down and rerun.

``--config FILE`` on ``run_experiment.py`` and on ``honest-backtest audit``
reads a flat mapping whose keys are the command's long option names, with
dashes or underscores (``train-size`` or ``train_size``)::

    data: sample_prices.csv
    strategy: sma_crossover.py
    train_size: 504
    cpcv: [6, 2]

Each value goes through the same conversion and checks as the flag would, so
a file cannot set anything the command line could not. Flags given on the
command line override the file. Relative paths in the file (``data``,
``strategy``, ``html``, ``json``) are resolved against the file's own
directory, so a config keeps working wherever it is run from.

JSON files need nothing beyond the standard library. YAML files need PyYAML,
which is an optional extra (``pip install "honest-backtest[yaml]"``); without
it a ``.yaml`` file is refused with a message saying so, rather than parsed
by something that only looks like a YAML reader.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

#: Options that name a file. Relative values in a config file are resolved
#: against the config file's directory.
PATH_OPTIONS = frozenset({"data", "strategy", "html", "json"})

#: Options a config file may not set: they control how the file itself is read.
_NOT_CONFIGURABLE = frozenset({"help", "config"})


class ConfigError(ValueError):
    """A config file that cannot be read or does not fit the command."""


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Read a JSON or YAML config file into a mapping.

    Args:
        path: A ``.json``, ``.yaml`` or ``.yml`` file.

    Returns:
        The top-level mapping.

    Raises:
        ConfigError: If the file is missing, has another extension, is not
            valid, is not a mapping, or is YAML and PyYAML is not installed.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file {path} does not exist")
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} is not valid JSON: {exc}") from None
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ConfigError(
                f"{path} is YAML, which needs PyYAML: pip install pyyaml "
                '(or pip install "honest-backtest[yaml]"), or write the config '
                "as JSON instead"
            ) from None
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path} is not valid YAML: {exc}") from None
    else:
        raise ConfigError(
            f"{path}: config files must end in .json, .yaml or .yml, not {suffix!r}"
        )
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"{path} must hold a mapping of option names to values, "
            f"not a {type(data).__name__}"
        )
    return dict(data)


def _option_name(action: argparse.Action) -> str:
    """The long flag of an action, for messages."""
    longs = [s for s in action.option_strings if s.startswith("--")]
    return longs[0] if longs else action.dest


def _resolve_path(value: str, base: Path, dest: str, keep: Collection[str]) -> str:
    """Resolve a relative path against ``base``, leaving built-in names alone."""
    if dest == "strategy" and value in keep:
        return value
    if dest == "strategy":
        file_part, sep, attr = value.partition(":")
    else:
        file_part, sep, attr = value, "", ""
    path = Path(file_part).expanduser()
    if not path.is_absolute():
        path = base / path
    return f"{path}{sep}{attr}"


def config_defaults(
    parser: argparse.ArgumentParser,
    values: Mapping[str, Any],
    *,
    base: Path,
    keep_names: Collection[str] = (),
) -> dict[str, Any]:
    """Turn a config mapping into parser defaults, checked like flags.

    Args:
        parser: The command's parser.
        values: The config file's mapping.
        base: Directory relative paths are resolved against.
        keep_names: Values of ``strategy`` that are built-in names rather
            than files, and so are not resolved as paths.

    Returns:
        ``{dest: value}`` ready for ``parser.set_defaults``.

    Raises:
        ConfigError: For an unknown key, or a value the flag would refuse.
    """
    actions = {
        a.dest: a
        # argparse has no public list of its actions; this one is stable.
        for a in parser._actions
        if a.dest not in _NOT_CONFIGURABLE and a.option_strings
    }
    defaults: dict[str, Any] = {}
    for key, raw in values.items():
        dest = str(key).replace("-", "_")
        action = actions.get(dest)
        if action is None:
            known = ", ".join(sorted(d.replace("_", "-") for d in actions))
            raise ConfigError(f"unknown config key {key!r}; expected one of: {known}")
        flag = _option_name(action)
        if isinstance(action, argparse._StoreTrueAction):
            if not isinstance(raw, bool):
                raise ConfigError(f"{key}: {flag} is a switch, so use true or false")
            defaults[dest] = raw
            continue
        if raw is None:
            defaults[dest] = None
            continue
        if isinstance(raw, Sequence) and not isinstance(raw, str):
            text = ",".join(str(v) for v in raw)
        elif isinstance(raw, bool) or isinstance(raw, Mapping):
            raise ConfigError(f"{key}: {flag} takes a value, not {raw!r}")
        else:
            text = str(raw)
        if dest in PATH_OPTIONS:
            text = _resolve_path(text, base, dest, keep_names)
        try:
            value = action.type(text) if action.type is not None else text
        except (argparse.ArgumentTypeError, TypeError, ValueError) as exc:
            raise ConfigError(f"{key}: {flag} cannot take {raw!r} ({exc})") from None
        if action.choices is not None and value not in action.choices:
            choices = ", ".join(map(str, action.choices))
            raise ConfigError(f"{key}: {flag} must be one of {choices}, not {raw!r}")
        defaults[dest] = value
    return defaults


def add_config_option(parser: argparse.ArgumentParser) -> None:
    """Add ``--config`` to a command's parser."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "read settings from a JSON or YAML file whose keys are this command's "
            "long option names; flags on the command line override it"
        ),
    )


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
    *,
    keep_names: Collection[str] = (),
) -> argparse.Namespace:
    """Parse ``argv``, taking defaults from ``--config`` when it is given.

    The file's values become the parser's defaults, so anything given on the
    command line still wins. A bad file is reported through ``parser.error``,
    like a bad flag.

    Args:
        parser: The command's parser, with ``--config`` added.
        argv: Command-line arguments.
        keep_names: Built-in strategy names, which are not paths.

    Returns:
        The parsed arguments.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args(list(argv))
    if known.config is not None:
        try:
            values = load_config_file(known.config)
            parser.set_defaults(
                **config_defaults(
                    parser,
                    values,
                    base=known.config.resolve().parent,
                    keep_names=keep_names,
                )
            )
        except ConfigError as exc:
            parser.error(str(exc))
    return parser.parse_args(list(argv))
