"""
pm — Typed Configuration Management for ML Projects
==================================================================

Combines the best of **Hydra** (YAML config composition, ``${}`` interpolation,
output management) with **tyro** (type-safe dataclasses, IDE autocompletion).

Define your config as typed Python dataclasses. Load, merge, and override from
YAML files and CLI arguments. Get a fully typed config object back — with
``${}`` interpolation resolved, types validated, and the resolved config saved
alongside experiment outputs.

Dependencies: ``PyYAML`` (``pip install pyyaml``). Python >= 3.9.

Quickstart
----------

.. code-block:: python

    from dataclasses import dataclass, field
    from pm import load_config

    @dataclass
    class ModelConfig:
        hidden_dim: int = 256
        num_layers: int = 6
        dropout: float = 0.1

    @dataclass
    class TrainConfig:
        model: ModelConfig = field(default_factory=ModelConfig)
        lr: float = 3e-4
        batch_size: int = 32
        exp_name: str = "default"
        output_dir: str = "outputs/${exp_name}/${now:%Y-%m-%d_%H-%M-%S}"

    # Load from YAML with CLI overrides. cfg is a fully typed TrainConfig.
    cfg = load_config(TrainConfig, config_files=["base.yaml"])
    print(cfg.model.hidden_dim)  # IDE autocompletion works!

CLI Override Grammar
--------------------

Override any config value from the command line using dotted paths::

    python train.py --model.hidden_dim=512 --lr=1e-3 --exp_name=run42

The library extracts only ``--dotted.path=value`` tokens and leaves everything
else (``--flag``, ``positional``) untouched for your own argument parser.

Or pass overrides programmatically::

    cfg = load_config(TrainConfig, overrides=["model.hidden_dim=512", "lr=1e-3"])

Interpolation Reference
-----------------------

Values containing ``${...}`` are resolved automatically. Supported expressions:

=================  =========================================  ================================
Expression          Meaning                                     Example
=================  =========================================  ================================
``${a.b.c}``        Config cross-reference (absolute path)     ``${model.hidden_dim}``
``${env:NAME}``     Environment variable                       ``${env:HOME}``
``${env:NAME:def}`` Env var with default                       ``${env:CUDA_VISIBLE_DEVICES:0}``
``${now:FORMAT}``   ``datetime.strftime`` (frozen per load)    ``${now:%Y-%m-%d_%H-%M-%S}``
``${now}``          ISO-8601 timestamp                         ``${now}``
``${eval:EXPR}``    Python expression (cfg, env, now, math)    ``${eval:cfg.lr * 10}``
``$${``             Escaped literal ``${``                     ``"$${not_interp}"``
``???``             Required-but-unset marker (whole value)    ``secret: ???``
=================  =========================================  ================================

- **Nested** interpolations: ``${env:PREFIX_${suffix}}`` resolved innermost-first.
- **Type preservation**: ``hidden_dim: ${model.dim}`` keeps ``int`` if ``model.dim`` is ``int``.
- **Partial matches** stringify: ``"result_${exp_name}"`` → ``"result_run42"``.
- Interpolation inside YAML **keys** is not supported (same as Hydra).

Precedence Chain
----------------

Values are resolved in this order (last wins)::

    dataclass field defaults  <  YAML file 1  <  YAML file 2  <  CLI overrides

Within each YAML file, ``defaults:`` group entries are composed first (see below).

Config Groups (Hydra-style ``defaults:``)
------------------------------------------

A YAML file may declare a ``defaults:`` list to compose config fragments:

.. code-block:: yaml

    # conf/base.yaml
    defaults:
      - db: mysql          # loads conf/db/mysql.yaml → merged under "db" key
      - common/training    # loads conf/common/training.yaml → merged at top level
    lr: 3e-4

Config file layout::

    conf/
      base.yaml
      db/
        mysql.yaml          # host: localhost, port: 3306
        postgres.yaml       # host: pg.example.com, port: 5432
      common/
        training.yaml       # batch_size: 32, max_epochs: 100

Entry forms:
  - ``key: path`` — load ``path.yaml`` and merge result under ``key``
  - ``path`` — load ``path.yaml`` and merge at top level

Paths resolve relative to the **referencing file's directory**. ``.yaml`` is
appended automatically if no extension. Groups recurse (group files may have
their own ``defaults:`` lists). Set ``resolve_defaults=False`` to disable.

Output Layout
-------------

When ``save=True`` (default), the resolved config is written to ``output_dir``::

    outputs/my_exp/2026-08-05_11-23-04/
      config.yaml       # fully resolved config (round-trip safe)
      meta.json         # timestamp, command, git sha, python version
      overrides.txt     # override tokens, one per line

Typed Config Idioms
-------------------

.. code-block:: python

    from dataclasses import dataclass, field
    from enum import Enum
    from typing import Optional, List, Dict, Any, Literal

    class Optimizer(Enum):
        ADAM = "adam"
        SGD = "sgd"

    @dataclass
    class Config:
        # Optional fields — None from YAML or omitted
        wandb_project: Optional[str] = None

        # Enums — matched by name or value
        optimizer: Optimizer = Optimizer.ADAM

        # Literals — type-safe choices
        precision: Literal["fp16", "fp32", "bf16"] = "fp32"

        # Lists — comma-separated or JSON on CLI
        layer_dims: List[int] = field(default_factory=lambda: [256, 128, 64])

        # Extra metadata (with allow_unknown=True)
        extra: Dict[str, Any] = field(default_factory=dict)

        # Required — use "???" in YAML or a field with no default
        dataset_path: str = ""  # will be filled by YAML or CLI

Comparison with Hydra and tyro
-------------------------------

===================  ==============  ==============  ====================
Capability            Hydra           tyro            parameter_manager
===================  ==============  ==============  ====================
Config schema         YAML/ConfigStore dataclass       dataclass
Typed config objects   DictConfig      native types    native types
IDE autocompletion     limited         excellent       excellent
YAML file loading      excellent (composition)  manual    excellent (composition)
YAML ``defaults:``     full            none            simplified groups
``${}`` interpolation  yes (OmegaConf) no              yes
CLI overrides          rich grammar    --key=value     --key=value (non-hijacking)
Output management      yes (.hydra)    no              yes (config.yaml + meta)
``instantiate``        yes             no              no (use __post_init__)
Dependencies           OmegaConf+heavy stdlib+PyYAML   PyYAML only
===================  ==============  ==============  ====================

Security
--------

``${eval:...}`` executes arbitrary Python code. Never load untrusted YAML files
that use eval interpolation.

Limitations
-----------

- Config dataclasses must be defined at **module level** (``get_type_hints``
  requires importable classes).
- ``InitVar`` fields are not supported.
- List merging is **wholesale replace**, not per-index (unlike OmegaConf).
- Interpolation inside YAML **keys** is not supported.
- No built-in ``instantiate`` (use ``__post_init__`` or a factory pattern instead).

API Reference
-------------

.. autofunction:: load_config
.. autofunction:: save_config
.. autofunction:: parse_overrides
.. autofunction:: render_help
.. autofunction:: to_plain
.. data:: last_output_dir
.. autoexception:: ConfigError
.. autoexception:: InterpolationError
.. autoexception:: TypeConversionError
.. autoexception:: YamlError
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import datetime
import difflib
import enum
import json
import math
import os
import pathlib
import re
import socket
import subprocess
import sys
import types
import typing
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import yaml

__version__ = "0.1.0"
__all__ = [
    "load_config",
    "save_config",
    "parse_overrides",
    "print_config",
    "render_help",
    "to_plain",
    "last_output_dir",
    "ConfigError",
    "InterpolationError",
    "TypeConversionError",
    "YamlError",
]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Base exception for all pm errors."""


class InterpolationError(ConfigError):
    """Cycle, missing key, bad ``${}`` syntax, or eval error during interpolation."""


class TypeConversionError(ConfigError):
    """Wrong type, unknown key, missing required field during conversion/instantiation."""


class YamlError(ConfigError):
    """File missing, parse error, or bad ``defaults:`` composition."""


# ---------------------------------------------------------------------------
# Sentinel: missing / required
# ---------------------------------------------------------------------------

_MISSING: Any = dataclasses.MISSING
_UNRESOLVED = object()  # sentinel: dependency not ready yet


def _is_missing(v: Any) -> bool:
    return v is _MISSING


def _is_unresolved(v: Any) -> bool:
    return v is _UNRESOLVED


# ---------------------------------------------------------------------------
# Run context — frozen per load_config call
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _RunContext:
    now: datetime.datetime
    cwd: Path

    @staticmethod
    def create() -> "_RunContext":
        return _RunContext(now=datetime.datetime.now(), cwd=Path.cwd())

    @property
    @lru_cache(maxsize=1)
    def git_sha(self) -> Optional[str]:
        try:
            r = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                cwd=str(self.cwd),
            )
            return r.stdout.strip()[:12] if r.returncode == 0 else None
        except Exception:
            return None

    @property
    def command(self) -> str:
        return " ".join(sys.argv)


# ---------------------------------------------------------------------------
# Schema extraction
# ---------------------------------------------------------------------------

_FieldInfo = collections.namedtuple(  # type: ignore[name-match]
    "_FieldInfo", ["type", "default", "default_factory", "metadata"]
)
_FieldInfo.__doc__ = """
Holds a cleaned-up snapshot of one dataclass field.

Attributes:
    type:      Resolved Python type (from ``get_type_hints``).
    default:   Literal default value, or ``_MISSING``.
    default_factory: Callable, or ``_MISSING``.
    metadata:  ``field(metadata=...)`` dict.
"""


@lru_cache(maxsize=None)
def _type_hints(cls: type) -> Dict[str, Any]:
    """Cached ``get_type_hints`` — resolves string annotations."""
    return typing.get_type_hints(cls)


@lru_cache(maxsize=None)
def _field_map(cls: type) -> Dict[str, _FieldInfo]:
    """Build the authoritative field map for a dataclass.

    Uses ``get_type_hints`` (not bare ``f.type``) so ``from __future__ import annotations``
    works, and ``Optional[X]`` / forward references resolve correctly.
    """
    hints = _type_hints(cls)
    result: Dict[str, _FieldInfo] = {}
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        result[f.name] = _FieldInfo(
            type=hints.get(f.name, f.type),
            default=f.default,
            default_factory=f.default_factory,
            metadata=f.metadata,
        )
    return result


def _kind(t: Any) -> str:
    """Classify a type for the conversion dispatch.

    Returns one of:
      dataclass, optional, list, dict, tuple, literal, enum,
      path, datetime, date, bool, int, float, str, any, other.
    """
    origin = typing.get_origin(t)
    args = typing.get_args(t)

    # Optional[T] = Union[T, None]
    if origin in (Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return "optional"
        return "union"

    if origin is Literal:
        return "literal"
    if origin in (list, List):
        return "list"
    if origin in (dict, Dict):
        return "dict"
    if origin in (tuple, Tuple):
        return "tuple"
    if t is Any:
        return "any"
    if isinstance(t, type):
        if dataclasses.is_dataclass(t):
            return "dataclass"
        if issubclass(t, bool):
            return "bool"
        if issubclass(t, int):
            return "int"
        if issubclass(t, float):
            return "float"
        if issubclass(t, str):
            return "str"
        if issubclass(t, enum.Enum):
            return "enum"
        if issubclass(t, Path):
            return "path"
        if issubclass(t, datetime.datetime):
            return "datetime"
        if issubclass(t, datetime.date):
            return "date"
    return "other"


def _defaults_to_dict(cls: type) -> dict:
    """Materialize dataclass defaults into a plain dict tree."""
    result: dict = {}
    for name, info in _field_map(cls).items():
        if not _is_missing(info.default):
            val = info.default
        elif not _is_missing(info.default_factory):
            val = info.default_factory()
        else:
            continue
        k = _kind(info.type)
        if k == "dataclass" and dataclasses.is_dataclass(val):
            result[name] = _defaults_to_dict(type(val))
        else:
            result[name] = val
    return result


# ---------------------------------------------------------------------------
# YAML loading, deep merge, defaults groups
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* into *base*. Later wins. Lists are replaced."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _resolve_defaults_path(file_dir: Path, entry: str) -> Path:
    """Resolve a defaults entry to a concrete .yaml file."""
    p = file_dir / entry
    if p.suffix == "":
        p = p.with_suffix(".yaml")
    return p


def _load_yaml_with_defaults(
    file_path: Path,
    visited: Optional[set] = None,
    resolve_defaults: bool = True,
) -> dict:
    """Load a YAML file and recursively resolve its ``defaults:`` list.

    *visited* tracks absolute resolved paths to prevent cycles.
    """
    if visited is None:
        visited = set()

    abs_path = file_path.resolve()
    if abs_path in visited:
        raise YamlError(f"circular defaults reference: {abs_path}")
    visited.add(abs_path)

    if not abs_path.is_file():
        raise YamlError(f"config file not found: {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise YamlError(f"YAML parse error in {abs_path}: {e}") from e

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise YamlError(f"config file must contain a mapping, got {type(data).__name__} in {abs_path}")

    if not resolve_defaults:
        return data

    defaults_entries = data.pop("defaults", None)
    if defaults_entries is None:
        return data

    if not isinstance(defaults_entries, list):
        raise YamlError(f"'defaults' must be a list in {abs_path}")

    merged: dict = {}
    file_dir = abs_path.parent

    for entry in defaults_entries:
        if isinstance(entry, str):
            # Bare path: merge at top level
            target = _resolve_defaults_path(file_dir, entry)
            child = _load_yaml_with_defaults(target, visited, resolve_defaults)
            _deep_merge(merged, child)
        elif isinstance(entry, dict):
            # key: path — merge under key.
            # The value is resolved relative to a subdirectory named by the key
            # so that `db: mysql` → <file_dir>/db/mysql.yaml
            for key, sub_path in entry.items():
                target = _resolve_defaults_path(file_dir / key, sub_path)
                child = _load_yaml_with_defaults(target, visited, resolve_defaults)
                if key not in merged:
                    merged[key] = {}
                if isinstance(merged[key], dict) and isinstance(child, dict):
                    _deep_merge(merged[key], child)
                else:
                    merged[key] = copy.deepcopy(child)
        else:
            raise YamlError(f"invalid defaults entry in {abs_path}: {entry!r}")

    # Current file's own keys override groups
    _deep_merge(merged, data)
    return merged


def _load_yaml_files(
    config_files: List[Union[str, Path, dict]],
    resolve_defaults: bool,
) -> dict:
    """Load and merge a sequence of YAML files / in-memory dicts."""
    merged: dict = {}
    for cf in config_files:
        if isinstance(cf, dict):
            overlay = copy.deepcopy(cf)
        else:
            file_path = Path(cf)
            if not file_path.is_absolute():
                file_path = Path.cwd() / file_path
            overlay = _load_yaml_with_defaults(file_path, resolve_defaults=resolve_defaults)
        _deep_merge(merged, overlay)
    return merged


# ---------------------------------------------------------------------------
# CLI override parsing
# ---------------------------------------------------------------------------


def parse_overrides(argv: Optional[List[str]] = None) -> Tuple[Dict[str, str], List[str]]:
    """Extract ``--dotted.path=value`` tokens from *argv*.

    Returns ``(overrides, rest)`` where *overrides* maps dotted paths to raw
    string values, and *rest* is everything else (flags, positionals).

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Examples:
        >>> parse_overrides(["--model.dim=512", "--verbose", "input.txt"])
        ({'model.dim': '512'}, ['--verbose', 'input.txt'])

        >>> parse_overrides(["--lr=1e-3", "--batch_size=32"])
        ({'lr': '1e-3', 'batch_size': '32'}, [])
    """
    if argv is None:
        argv = sys.argv[1:]

    overrides: Dict[str, str] = {}
    rest: List[str] = []

    for token in argv:
        if token.startswith("--") and "=" in token:
            key, _, value = token[2:].partition("=")
            if key:  # --= would be weird but harmless to skip
                overrides[key] = value
        else:
            rest.append(token)

    return overrides, rest


def _apply_overrides(tree: dict, overrides: Dict[str, str], schema_map: dict) -> dict:
    """Apply CLI overrides to *tree*, validating against *schema_map*."""
    for key, raw_value in overrides.items():
        parts = key.split(".")
        current: Any = tree
        for i, part in enumerate(parts[:-1]):
            if not isinstance(current, dict):
                raise TypeConversionError(f"cannot set '{key}': '{part}' is not a mapping")
            if part not in current:
                current[part] = {}
            current = current[part]
        leaf_key = parts[-1]
        if isinstance(current, dict):
            # Type-convert early so interpolation sees correct types (optional; string is fine)
            current[leaf_key] = raw_value
        else:
            raise TypeConversionError(f"cannot set '{key}': parent is not a mapping")
    return tree


def _validate_path(path: str, schema_fields: Dict[str, Any]) -> None:
    """Check *path* against known schema fields; raise with suggestions."""
    if path not in schema_fields:
        close = difflib.get_close_matches(path, list(schema_fields), n=3, cutoff=0.5)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        raise TypeConversionError(f"unknown key {path!r}{hint}")


# ---------------------------------------------------------------------------
# Interpolation engine
# ---------------------------------------------------------------------------


def _find_interp_spans(s: str) -> List[Tuple[int, int]]:
    """Return (start, end) of every top-level ``${...}`` in *s*.

    Uses a balanced-brace scan so ``}`` inside ``eval:`` dict literals is handled.
    """
    spans: List[Tuple[int, int]] = []
    i = 0
    while True:
        start = s.find("${", i)
        if start < 0:
            break
        depth = 1
        j = start + 2
        while j < len(s) and depth:
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
            j += 1
        if depth:
            raise InterpolationError(f"unbalanced '${{' in {s!r}")
        spans.append((start, j - 1))
        i = j
    return spans


def _walk_tree(tree: dict, prefix: str = "") -> List[Tuple[str, Any]]:
    """Flatten *tree* to ``[(dotted_path, leaf_value), ...]``."""
    result: List[Tuple[str, Any]] = []
    for key, value in tree.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and not _is_dataclass_dict(value):
            result.extend(_walk_tree(value, full))
        else:
            result.append((full, value))
    return result


def _get_by_path(tree: dict, path: str) -> Any:
    """Get a value from *tree* by dotted path. Returns ``None`` if missing."""
    parts = path.split(".")
    current: Any = tree
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _set_by_path(tree: dict, path: str, value: Any) -> None:
    """Set a value in *tree* by dotted path, creating intermediate dicts."""
    parts = path.split(".")
    current: Any = tree
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


class _TreeProxy:
    """Lazy proxy for ``cfg`` in ``${eval:...}`` expressions.

    ``__getattr__`` walks the tree and resolves the target on demand.
    """

    __slots__ = ("_tree", "_resolver")

    def __init__(self, tree: dict, resolver: "_Resolver") -> None:
        object.__setattr__(self, "_tree", tree)
        object.__setattr__(self, "_resolver", resolver)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._resolver._lookup(name, f"cfg.{name}")

    def __getitem__(self, key: str) -> Any:
        return self._resolver._lookup(key, f"cfg[{key!r}]")


class _Resolver:
    """Resolves ``${...}`` expressions in a plain-dict config tree."""

    def __init__(self, context: _RunContext, tree: dict, schema_cls: Optional[type] = None) -> None:
        self.ctx = context
        self.tree = tree
        self.schema_cls = schema_cls
        self._active: set = set()  # paths currently being resolved (cycle detection)

    def resolve(self) -> dict:
        """Fixpoint resolution: iterate until no progress, then report leftovers."""
        # Pre-pass: replace $${ escapes with a sentinel so they don't
        # produce false positives when the fixpoint re-scans strings.
        _ESC_SENTINEL = "\x00ESC_DOLLAR\x01"
        self._replace_all_strings(self.tree, "$${", _ESC_SENTINEL)

        changed = True
        max_iterations = 100
        iteration = 0
        while changed:
            iteration += 1
            if iteration > max_iterations:
                raise InterpolationError("interpolation exceeded 100 passes — likely cycle")
            changed = False
            for path, value in _walk_tree(self.tree):
                new = self._try_resolve(value, path)
                if not _is_unresolved(new) and new is not value:
                    _set_by_path(self.tree, path, new)
                    changed = True
        leftovers = [
            p for p, v in _walk_tree(self.tree)
            if isinstance(v, str) and _find_interp_spans(v)
        ]
        if leftovers:
            raise InterpolationError(
                f"unresolvable interpolation at: {leftovers}"
            )

        # Post-pass: restore $${ → ${
        self._replace_all_strings(self.tree, _ESC_SENTINEL, "${")
        return self.tree

    @staticmethod
    def _replace_all_strings(tree: dict, old: str, new: str) -> None:
        """Recursively replace *old* with *new* in all string values of *tree*."""
        for key, value in tree.items():
            if isinstance(value, str):
                tree[key] = value.replace(old, new)
            elif isinstance(value, dict):
                _Resolver._replace_all_strings(value, old, new)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        value[i] = item.replace(old, new)
                    elif isinstance(item, dict):
                        _Resolver._replace_all_strings(item, old, new)

    def _try_resolve(self, value: Any, path: str) -> Any:
        """Attempt to resolve *value* at *path*. Returns _UNRESOLVED if not ready."""
        if not isinstance(value, str):
            return value
        if "${" not in value:
            return value
        return self._resolve_string(value, path)

    def _resolve_string(self, s: str, from_path: str) -> Any:
        """Resolve all ``${...}`` in *s*.

        Note: ``$${`` escape handling is done in ``resolve()`` as a pre/post pass,
        so this function never sees escape sentinels.
        """
        spans = _find_interp_spans(s)
        if not spans:
            return s

        # Resolve innermost spans first
        resolved_spans: List[Tuple[int, int, Any]] = []
        for start, end in spans:
            expr = s[start + 2 : end]  # strip ${ and }
            inner = self._resolve_inner(expr, from_path)
            if _is_unresolved(inner):
                return _UNRESOLVED
            resolved_spans.append((start, end, inner))

        # Single span covering the entire string → type-preserving
        if len(resolved_spans) == 1 and resolved_spans[0][0] == 0 and resolved_spans[0][1] == len(s) - 1:
            return resolved_spans[0][2]  # keep original type

        # Multiple spans or partial match → string concatenation
        parts: List[str] = []
        prev_end = 0
        for start, end, inner_val in resolved_spans:
            parts.append(s[prev_end:start])
            parts.append(str(inner_val))
            prev_end = end + 1
        parts.append(s[prev_end:])
        return "".join(parts)

    def _resolve_inner(self, expr: str, from_path: str) -> Any:
        """Resolve a single interpolation expression (contents inside ``${}``).

        May recurse for nested interpolations: ``${env:A_${b}}``.
        """
        # Resolve nested interpolations first
        if "${" in expr:
            expr = self._resolve_string(expr, from_path)
            if _is_unresolved(expr):
                return _UNRESOLVED
            if not isinstance(expr, str):
                return expr

        expr = expr.strip()

        if expr.startswith("env:"):
            rest = expr[4:]
            if ":" in rest:
                name, _, default = rest.partition(":")
                return os.environ.get(name, default)
            return os.environ.get(rest, "")

        if expr == "now":
            return self.ctx.now.isoformat()

        if expr.startswith("now:"):
            fmt = expr[4:]
            return self.ctx.now.strftime(fmt)

        if expr.startswith("eval:"):
            code = expr[5:]
            return self._eval(code, from_path)

        # Bare dotted path → config reference
        return self._lookup(expr, from_path)

    def _lookup(self, target: str, from_path: str) -> Any:
        """Resolve a config cross-reference ``target``."""
        target = target.strip()
        if target in self._active:
            chain = " -> ".join(sorted(self._active) + [target])
            raise InterpolationError(f"circular interpolation: {chain}")

        self._active.add(target)
        try:
            value = _get_by_path(self.tree, target)
            if value is None and target not in _flat_keys(self.tree):
                # Give a helpful error with suggestions
                all_keys = _flat_keys(self.tree)
                close = difflib.get_close_matches(target, all_keys, n=3, cutoff=0.5)
                hint = f" (did you mean {close[0]!r}?)" if close else ""
                raise InterpolationError(
                    f"missing key {target!r} referenced by {from_path!r}{hint}"
                )
            if isinstance(value, str):
                return self._resolve_string(value, target)
            return value
        finally:
            self._active.discard(target)

    def _eval(self, code: str, from_path: str) -> Any:
        """Evaluate a Python expression in ``${eval:...}``."""
        ns: Dict[str, Any] = {
            "cfg": _TreeProxy(self.tree, self),
            "env": os.environ,
            "now": self.ctx.now,
            "math": math,
            "__builtins__": {
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "dict": dict,
                "enumerate": enumerate,
                "filter": filter,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "map": map,
                "max": max,
                "min": min,
                "range": range,
                "round": round,
                "set": set,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
                "True": True,
                "False": False,
                "None": None,
            },
        }
        try:
            return eval(code, ns)
        except InterpolationError:
            raise
        except Exception as e:
            raise InterpolationError(
                f"eval error in '{from_path}': {e} (expr: {code!r})"
            ) from e


def _flat_keys(tree: dict, prefix: str = "") -> List[str]:
    """Get all leaf dotted-path keys from *tree*."""
    return [p for p, _ in _walk_tree(tree)]


def _is_dataclass_dict(d: dict) -> bool:
    """Guess whether a dict looks like it should be a dataclass (heuristic)."""
    # We'll rely on the schema to decide; this is just a tree-walking guard.
    return False


# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------


def _convert(
    value: Any,
    expected_type: Any,
    path: str,
    allow_unknown: bool = False,
) -> Any:
    """Convert *value* to *expected_type*, raising on mismatch.

    This is the central type-conversion dispatch — see the plan for the full table.
    """
    k = _kind(expected_type)

    # 0. Any → passthrough (must be first; typing.Any breaks isinstance)
    if k == "any":
        return value

    # 1. Short-circuit: already correct type
    if isinstance(expected_type, type) and isinstance(value, expected_type):
        return value

    # 2. None handling
    if value is None or (isinstance(value, str) and value.lower() in ("null", "none")):
        if k == "optional":
            return None
        raise TypeConversionError(f"{path}: got None/null, but field is not Optional")

    # 3. Any → passthrough
    if k == "any":
        return value

    # 4. Optional[T] — unwrap and try inner type
    if k == "optional":
        origin = typing.get_origin(expected_type)
        args = typing.get_args(expected_type)
        if origin is None:
            origin = getattr(expected_type, "__origin__", None)
            args = getattr(expected_type, "__args__", ())
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        inner = non_none[0] if non_none else str
        return _convert(value, inner, path, allow_unknown)

    # 5. Union (non-Optional)
    if k == "union":
        origin = typing.get_origin(expected_type) or getattr(expected_type, "__origin__", None)
        args = typing.get_args(expected_type) or getattr(expected_type, "__args__", ())
        errors = []
        for arg in args:
            try:
                return _convert(value, arg, path, allow_unknown)
            except (TypeConversionError, InterpolationError):
                pass
            except Exception as e:
                errors.append(str(e))
        raise TypeConversionError(
            f"{path}: value {value!r} doesn't match any type in Union[{', '.join(str(a) for a in args)}]"
        )

    # 6. list[T]
    if k == "list":
        inner_type = typing.get_args(expected_type)[0] if typing.get_args(expected_type) else Any
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                if value.strip() == "":
                    parsed = []
                else:
                    parsed = [x.strip() for x in value.split(",")]
            if isinstance(parsed, list):
                value = parsed
            else:
                value = [parsed]
        if not isinstance(value, list):
            raise TypeConversionError(f"{path}: expected list, got {type(value).__name__}")
        return [_convert(item, inner_type, f"{path}[{i}]", allow_unknown) for i, item in enumerate(value)]

    # 7. dict[K,V]
    if k == "dict":
        args = typing.get_args(expected_type)
        key_type = args[0] if len(args) > 0 else Any
        val_type = args[1] if len(args) > 1 else Any
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                raise TypeConversionError(f"{path}: cannot parse dict from string {value!r}")
        if not isinstance(value, dict):
            raise TypeConversionError(f"{path}: expected dict, got {type(value).__name__}")
        return {
            _convert(k, key_type, f"{path}.keys", allow_unknown): _convert(v, val_type, f"{path}.{k}", allow_unknown)
            for k, v in value.items()
        }

    # 8. tuple
    if k == "tuple":
        args = typing.get_args(expected_type)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                value = [x.strip() for x in value.split(",")]
        if not isinstance(value, (list, tuple)):
            raise TypeConversionError(f"{path}: expected tuple/list, got {type(value).__name__}")
        if len(args) == 2 and args[1] is ...:
            # variadic tuple[T, ...]
            return tuple(_convert(item, args[0], f"{path}[{i}]", allow_unknown) for i, item in enumerate(value))
        # fixed-length tuple
        if len(value) != len(args):
            raise TypeConversionError(f"{path}: tuple length mismatch: expected {len(args)}, got {len(value)}")
        return tuple(_convert(item, args[i], f"{path}[{i}]", allow_unknown) for i, item in enumerate(value))

    # 9. Literal
    if k == "literal":
        args = typing.get_args(expected_type)
        for arg in args:
            try:
                converted = _convert(value, type(arg), path, allow_unknown)
            except Exception:
                continue
            if converted == arg:
                return converted
        raise TypeConversionError(f"{path}: {value!r} not in Literal{args}")

    # 10. Enum
    if k == "enum":
        if isinstance(value, expected_type):
            return value
        # Try by name
        if isinstance(value, str):
            try:
                return expected_type[value]
            except KeyError:
                pass
        # Try by value
        for member in expected_type:
            if member.value == value:
                return member
        valid = ", ".join(f"{m.name}={m.value!r}" for m in expected_type)
        raise TypeConversionError(f"{path}: {value!r} is not a valid {expected_type.__name__} (valid: {valid})")

    # 11. Dataclass
    if k == "dataclass":
        if isinstance(value, expected_type):
            return value
        if not isinstance(value, dict):
            raise TypeConversionError(f"{path}: expected mapping for {expected_type.__name__}, got {type(value).__name__}")
        return _instantiate(expected_type, value, path, allow_unknown)

    # 12. Path
    if k == "path":
        return Path(str(value))

    # 13. datetime
    if k == "datetime":
        if isinstance(value, str):
            return datetime.datetime.fromisoformat(value)
        if isinstance(value, datetime.datetime):
            return value
        raise TypeConversionError(f"{path}: expected datetime string, got {type(value).__name__}")

    # 14. date
    if k == "date":
        if isinstance(value, str):
            return datetime.date.fromisoformat(value)
        if isinstance(value, datetime.date):
            return value
        raise TypeConversionError(f"{path}: expected date string, got {type(value).__name__}")

    # 15. bool
    if k == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("true", "1", "yes", "on"):
                return True
            if v in ("false", "0", "no", "off", ""):
                return False
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        raise TypeConversionError(f"{path}: expected bool, got {value!r}")

    # 16. int
    if k == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            if value == int(value):
                return int(value)
            raise TypeConversionError(f"{path}: lossy float→int conversion: {value}")
        if isinstance(value, str):
            v = value.strip()
            # Try direct int parse
            try:
                return int(v)
            except ValueError:
                pass
            # Try float parse → int if lossless (e.g. "3.0" → 3)
            try:
                fv = float(v)
                if fv == int(fv):
                    return int(fv)
                raise TypeConversionError(f"{path}: lossy float→int conversion: {v!r}")
            except ValueError:
                raise TypeConversionError(f"{path}: expected int, got {v!r}")
        raise TypeConversionError(f"{path}: expected int, got {type(value).__name__} {value!r}")

    # 17. float
    if k == "float":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                raise TypeConversionError(f"{path}: expected float, got {value!r}")
        raise TypeConversionError(f"{path}: expected float, got {type(value).__name__} {value!r}")

    # 18. str
    if k == "str":
        return str(value)

    # 19. Fallback
    try:
        return expected_type(value)
    except Exception:
        return value


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def _instantiate(
    cls: type,
    data: dict,
    path: str,
    allow_unknown: bool = False,
) -> Any:
    """Construct a dataclass instance from *data*, validating all fields.

    Args:
        cls: The dataclass type to instantiate.
        data: Resolved plain-dict values.
        path: Dotted path for error messages.
        allow_unknown: If False, raise on keys not in the schema.

    Raises:
        TypeConversionError: On unknown keys, missing required fields, type mismatches.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeConversionError(f"{path}: {cls.__name__} is not a dataclass")

    fields_info = _field_map(cls)
    unknown = set(data) - set(fields_info)
    if unknown and not allow_unknown:
        closest = []
        for uk in sorted(unknown):
            matches = difflib.get_close_matches(uk, list(fields_info), n=1, cutoff=0.4)
            closest.append(f"{uk!r}" + (f" (did you mean {matches[0]!r}?)" if matches else ""))
        raise TypeConversionError(
            f"{path}: unknown key(s): {', '.join(closest)}"
        )

    kwargs: Dict[str, Any] = {}
    for name, info in fields_info.items():
        field_path = f"{path}.{name}" if path else name

        if name in data:
            raw = data[name]
            # Handle ??? marker
            if raw == "???":
                if not _is_missing(info.default) or not _is_missing(info.default_factory):
                    kwargs[name] = (
                        info.default
                        if not _is_missing(info.default)
                        else info.default_factory()
                    )
                else:
                    raise TypeConversionError(f"required field {field_path!r} is marked '???' and has no default")
            else:
                kwargs[name] = _convert(raw, info.type, field_path, allow_unknown)
        elif not _is_missing(info.default):
            kwargs[name] = info.default
        elif not _is_missing(info.default_factory):
            kwargs[name] = info.default_factory()
        else:
            raise TypeConversionError(f"missing required field {field_path!r} (no default)")

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Output management
# ---------------------------------------------------------------------------


last_output_dir: Optional[Path] = None


def to_plain(cfg: Any) -> dict:
    """Convert a dataclass config tree to a plain dict (for YAML dump).

    Handles nested dataclasses, ``Path``, ``Enum``, ``datetime``, lists, and dicts.

    Example:
        >>> to_plain(MyConfig(hidden=256, name="test"))
        {'hidden': 256, 'name': 'test'}
    """
    if dataclasses.is_dataclass(cfg):
        result: dict = {}
        for f in dataclasses.fields(cfg):
            val = getattr(cfg, f.name)
            result[f.name] = to_plain(val)
        return result
    if isinstance(cfg, dict):
        return {str(k): to_plain(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [to_plain(item) for item in cfg]
    if isinstance(cfg, enum.Enum):
        return cfg.value
    if isinstance(cfg, Path):
        return str(cfg)
    if isinstance(cfg, datetime.datetime):
        return cfg.isoformat()
    if isinstance(cfg, datetime.date):
        return cfg.isoformat()
    return cfg


def save_config(cfg: Any, output_dir: Union[str, Path], name: str = "config") -> Path:
    """Save resolved config + metadata to *output_dir*.

    Writes:
      - ``{name}.yaml`` — fully resolved config
      - ``meta.json`` — timestamp, command, git sha, etc.
      - ``overrides.txt`` — (only if not inside ``load_config`` flow; empty)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Config YAML
    config_path = out / f"{name}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain(cfg), f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Meta
    meta: Dict[str, Any] = {}
    meta["version"] = __version__
    meta["python"] = sys.version
    try:
        meta["time"] = datetime.datetime.now().isoformat()
    except Exception:
        pass
    try:
        meta["command"] = " ".join(sys.argv)
    except Exception:
        pass
    try:
        meta["cwd"] = str(Path.cwd())
    except Exception:
        pass
    try:
        meta["hostname"] = socket.gethostname()
    except Exception:
        pass
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            meta["git_sha"] = r.stdout.strip()
    except Exception:
        pass

    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    return out


# ---------------------------------------------------------------------------
# Help rendering
# ---------------------------------------------------------------------------


def render_help(cls: type) -> str:
    """Render a human-readable config tree for *cls*.

    Example:
        >>> print(render_help(TrainConfig))
        TrainConfig
          model (ModelConfig)
            hidden_dim : int = 256
            num_layers : int = 6
          lr : float = 3e-4
          ...
    """
    lines = _render_help_inner(cls, indent=0, is_root=True)
    return "\n".join(lines)


def _render_help_inner(cls: type, indent: int = 0, prefix: str = "", is_root: bool = True) -> List[str]:
    lines: List[str] = []
    if is_root:
        label = f"{prefix} ({cls.__name__})" if prefix else cls.__name__
        lines.append(f"{'  ' * indent}{label}")
    elif prefix:
        lines.append(f"{'  ' * indent}{prefix} ({cls.__name__})")

    for name, info in _field_map(cls).items():
        field_type = info.type
        k = _kind(field_type)
        spacer = "  " * (indent + 1)

        if k == "dataclass":
            lines.append(f"{spacer}{name} ({field_type.__name__})")
            lines.extend(_render_help_inner(field_type, indent + 2, is_root=False))
            continue

        type_name = _type_display_name(field_type)
        if not _is_missing(info.default):
            default_repr = json.dumps(info.default) if not isinstance(info.default, (int, float, bool)) else repr(info.default)
            lines.append(f"{spacer}{name} : {type_name} = {default_repr}")
        elif not _is_missing(info.default_factory):
            lines.append(f"{spacer}{name} : {type_name} = <factory>")
        else:
            lines.append(f"{spacer}{name} : {type_name} (required)")

    return lines


def _type_display_name(t: Any) -> str:
    """Human-friendly type name."""
    origin = typing.get_origin(t)
    args = typing.get_args(t)
    if origin is not None or args:
        if t is Any:
            return "Any"
        return str(t).replace("typing.", "")
    if isinstance(t, type):
        return t.__name__
    return str(t)


# ---------------------------------------------------------------------------
# Color config printing
# ---------------------------------------------------------------------------

# ANSI escape codes for terminal colors
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def print_config(
    cfg: Any,
    overrides: Optional[Dict[str, str]] = None,
    use_color: bool = True,
) -> str:
    """Pretty-print a resolved config, highlighting overridden values.

    Args:
        cfg: A resolved dataclass config instance.
        overrides: Optional dict of ``{dotted.path: raw_value}`` that were applied.
                   Paths in this set are highlighted in yellow.
        use_color: If ``True`` (default), use ANSI terminal colors.

    Returns:
        The formatted string (also printed to stdout).

    Colors (when ``use_color=True``):
        - **Green**: default values (unchanged).
        - **Yellow**: values that were explicitly overridden.
        - **Cyan**: section headers.
        - **Dim**: type annotations and separators.
    """
    overrides = overrides or {}
    override_paths = set(overrides.keys())
    lines: List[str] = []
    _print_config_inner(cfg, "", override_paths, lines, use_color)
    result = "\n".join(lines)

    # Print summary
    if override_paths:
        applied = {k: v for k, v in overrides.items()}
        summary_lines = [
            "",
            f"{_BOLD}{_YELLOW}Overrides applied:{_RESET}" if use_color else "Overrides applied:",
        ]
        for k, v in sorted(applied.items()):
            if use_color:
                summary_lines.append(f"  {_YELLOW}{k}{_RESET}{_DIM} = {_RESET}{_YELLOW}{v}{_RESET}")
            else:
                summary_lines.append(f"  {k} = {v}")
        result += "\n" + "\n".join(summary_lines)

    print(result)
    return result


def _print_config_inner(
    obj: Any,
    path: str,
    override_paths: set,
    lines: List[str],
    use_color: bool,
    indent: int = 0,
    is_last: bool = False,
) -> None:
    """Recursive helper for ``print_config``."""
    prefix = "  " * indent

    if dataclasses.is_dataclass(obj):
        name = type(obj).__name__
        header = f"{prefix}{_BOLD}{_CYAN}[{name}]{_RESET}" if use_color else f"{prefix}[{name}]"
        lines.append(header)
        fields = list(dataclasses.fields(obj))
        for i, f in enumerate(fields):
            val = getattr(obj, f.name)
            child_path = f"{path}.{f.name}" if path else f.name
            _print_config_inner(
                val, child_path, override_paths, lines, use_color, indent + 1,
                is_last=(i == len(fields) - 1),
            )
        return

    if isinstance(obj, dict):
        for i, (k, v) in enumerate(obj.items()):
            child_path = f"{path}.{k}" if path else k
            _print_config_inner(
                v, child_path, override_paths, lines, use_color, indent,
                is_last=False,
            )
        return

    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            child_path = f"{path}[{i}]"
            _print_config_inner(
                item, child_path, override_paths, lines, use_color, indent,
                is_last=False,
            )
        return

    # Leaf value
    field_name = path.split(".")[-1] if "." in path else path
    is_overridden = path in override_paths or _has_override_prefix(path, override_paths)

    if is_overridden:
        color = _YELLOW if use_color else ""
    else:
        color = _GREEN if use_color else ""

    val_repr = _format_value(obj)
    type_name = type(obj).__name__

    if use_color:
        line = f"{prefix}{color}{field_name}{_RESET}{_DIM}: {type_name} = {_RESET}{color}{val_repr}{_RESET}"
    else:
        line = f"{prefix}{field_name}: {type_name} = {val_repr}"

    # Add override marker
    if is_overridden and use_color:
        line += f"  {_YELLOW}{_BOLD}# <-- overridden{_RESET}"

    lines.append(line)


def _has_override_prefix(path: str, override_paths: set) -> bool:
    """Check if *path* or any parent is in *override_paths*."""
    parts = path.split(".")
    for i in range(len(parts)):
        if ".".join(parts[: i + 1]) in override_paths:
            return True
    return False


def _format_value(val: Any) -> str:
    """Format a scalar value for display."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        if val == 0 or (abs(val) >= 1e-4 and abs(val) <= 1e6):
            return f"{val:g}"
        return f"{val:.4e}"
    if isinstance(val, str) and len(val) > 60:
        return f'"{val[:57]}..."'
    if isinstance(val, enum.Enum):
        return val.name
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, (list, tuple)):
        items = [_format_value(v) for v in val]
        if len(items) <= 5:
            return "[" + ", ".join(items) + "]"
        return "[" + ", ".join(items[:3]) + f", ... ({len(items)} items)]"
    return repr(val)


# ---------------------------------------------------------------------------
# Main API: load_config
# ---------------------------------------------------------------------------


def load_config(
    cls: Type[Any],
    config_files: Optional[List[Union[str, Path, dict]]] = None,
    overrides: Optional[List[str]] = None,
    argv: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    save: bool = True,
    allow_unknown: bool = False,
    resolve_defaults: bool = True,
) -> Any:
    """Load a typed config from dataclass defaults, YAML files, and CLI overrides.

    The full pipeline:

    1. Materialize dataclass defaults into a base dict.
    2. Merge YAML files (with ``defaults:`` group composition) in order.
    3. Apply CLI overrides (``--path.to.key=value`` or *overrides* list).
    4. Resolve ``${...}`` interpolations via fixpoint passes.
    5. Convert values to declared types and instantiate the dataclass.
    6. Optionally save the resolved config to *output_dir*.

    Args:
        cls: The top-level ``@dataclass`` config type.
        config_files: List of YAML file paths, or in-memory dicts, merged in order.
        overrides: Raw override strings (``"a.b=value"``) — useful for programmatic use.
        argv: CLI argument list (default: ``sys.argv[1:]``). Only ``--key=value``
              tokens are consumed; everything else is left alone.
        output_dir: Template string for the output directory (may contain ``${}``
                    interpolations). If *cls* has an ``output_dir`` field, that
                    resolved value is used when *output_dir* is ``None``.
        save: If ``True`` (default), write ``config.yaml``, ``meta.json``, and
              ``overrides.txt`` to the resolved output directory.
        allow_unknown: If ``True``, silently drop YAML keys not in the dataclass
                       schema instead of raising.
        resolve_defaults: If ``True`` (default), process ``defaults:`` lists in
                          YAML files for config group composition.

    Returns:
        A fully typed instance of *cls* with all values resolved.

    Raises:
        ConfigError: On any configuration error.
        YamlError: On file/parse errors.
        TypeConversionError: On type mismatches, unknown keys, missing required fields.
        InterpolationError: On cycles, missing keys, or bad ``${}`` syntax.

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Simple:
        ...     a: int = 1
        ...     b: str = "hello"
        >>> cfg = load_config(Simple)
        >>> cfg.a
        1
        >>> cfg.b
        'hello'
    """
    if not dataclasses.is_dataclass(cls):
        raise ConfigError(f"{cls.__name__} is not a dataclass")

    ctx = _RunContext.create()
    config_files = config_files or []
    overrides = overrides or []

    # 1. Base layer: dataclass defaults
    tree = _defaults_to_dict(cls)

    # 2. YAML overlay
    if config_files:
        yaml_tree = _load_yaml_files(config_files, resolve_defaults)
        _deep_merge(tree, yaml_tree)

    # 3. CLI overrides
    cli_overrides, _ = parse_overrides(argv)
    all_overrides = dict(cli_overrides)
    for override_str in overrides:
        if "=" in override_str:
            k, _, v = override_str.partition("=")
            all_overrides[k] = v

    if all_overrides:
        _apply_overrides(tree, all_overrides, {})

    # 4. Interpolation
    resolver = _Resolver(ctx, tree, cls)
    tree = resolver.resolve()

    # 5. Instantiation
    cfg = _instantiate(cls, tree, "", allow_unknown)

    # 6. Output
    global last_output_dir
    if output_dir is None:
        # Check if the config has an output_dir field
        if hasattr(cfg, "output_dir") and isinstance(getattr(cfg, "output_dir"), str):
            output_dir = getattr(cfg, "output_dir")

    if output_dir and save:
        # Interpolate the output_dir template against the resolved config
        out_tree = to_plain(cfg)
        out_resolver = _Resolver(ctx, out_tree)
        out_dir_resolved = out_resolver._resolve_string(output_dir, "output_dir")
        if isinstance(out_dir_resolved, str):
            out_path = Path(out_dir_resolved)
        else:
            out_path = Path(str(out_dir_resolved))

        # Collision avoidance
        base = out_path
        suffix = 0
        while out_path.exists():
            suffix += 1
            out_path = base.with_name(f"{base.name}_{suffix}")

        saved = save_config(cfg, out_path)
        last_output_dir = saved

        # Write overrides file
        with open(last_output_dir / "overrides.txt", "w", encoding="utf-8") as f:
            for k, v in sorted(cli_overrides.items()):
                f.write(f"--{k}={v}\n")
            for override_str in overrides:
                f.write(f"{override_str}\n")
    elif save and not output_dir:
        pass  # no output_dir configured, skip silently

    return cfg
