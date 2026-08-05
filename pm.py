"""
pm — Typed Configuration Management for ML Projects
=====================================================

**Integration layer** combining **Hydra** (YAML composition, ``${}``
interpolation, CLI override grammar, output management) with **tyro**-style
typed dataclass configs.

Built on ``hydra-core`` and ``omegaconf``.  ~450 lines.

Dependencies: ``pip install hydra-core``.

Quickstart
----------

.. code-block:: python

    from dataclasses import dataclass, field
    from pm import cli, print_config

    @dataclass
    class ModelConfig:
        hidden_dim: int = 256
        num_layers: int = 6

    @dataclass
    class TrainConfig:
        model: ModelConfig = field(default_factory=ModelConfig)
        lr: float = 3e-4
        batch_size: int = 32
        exp_name: str = "default"

    @cli(config_path="conf", config_name="base")
    def main(cfg: TrainConfig) -> None:
        '''Train a model.'''
        print_config(cfg)
        print(f"Training {cfg.exp_name} with dim={cfg.model.hidden_dim}")

    if __name__ == "__main__":
        main()

Run with Hydra-compatible CLI::

    python train.py model=large optimizer.lr=0.001 exp_name=test
    python train.py model.hidden_dim=512 +optimizer.momentum=0.9
    python train.py --multirun lr=1e-4,3e-4,1e-3

Design
------

Three layers:

1. **Hydra** handles: YAML composition (``defaults:`` groups), ``${}``
   interpolation, CLI override grammar (``key=value``, ``+key``, ``~key``,
   ``--multirun`` sweeps), output management (``.hydra/`` folder).

2. **pm** converts Hydra's ``DictConfig`` → plain dict → **typed dataclass**
   instance with full Python type support (``Literal``, ``Enum``, ``Union``,
   ``Path``, ``datetime``, nested dataclasses, etc.) — going beyond
   OmegaConf's limited type vocabulary.

3. **Your code** gets a fully typed, IDE-friendly config object.

CLI Override Grammar (Hydra-native)
------------------------------------

=======  ===============================  ==============================
Syntax   Meaning                           Example
=======  ===============================  ==============================
``k=v``  Override config value             ``model.hidden_dim=512``
``+k=v`` Append new key                    ``+optimizer.momentum=0.9``
``++k=v`` Force-set (override if exists)   ``++seed=42``
``~k``   Delete a key                      ``~wandb_project``
``a,b``  Sweep (with ``--multirun``)       ``lr=1e-4,3e-4,1e-3``
=======  ===============================  ==============================

Config groups are swapped natively::

    python train.py model=large optimizer=sgd data=cifar10

Config Groups (Hydra-native)
-----------------------------

.. code-block:: yaml

    # conf/base.yaml
    defaults:
      - model: base
      - optimizer: adamw
      - data: imagenet

    exp_name: default
    seed: 42

Config file layout::

    conf/
      base.yaml
      model/
        small.yaml    # hidden_dim: 128
        base.yaml     # hidden_dim: 256
        large.yaml    # hidden_dim: 512
      optimizer/
        adamw.yaml
        sgd.yaml
      data/
        imagenet.yaml
        cifar10.yaml

Output Management (Hydra-native)
---------------------------------

Each run produces::

    outputs/<date>/<time>/
      .hydra/
        config.yaml     # fully resolved config
        hydra.yaml      # hydra configuration
        overrides.yaml  # applied overrides

API Reference
-------------

.. autofunction:: cli
.. autofunction:: load_config
.. autofunction:: print_config
.. autofunction:: to_plain
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import datetime
import difflib
import enum
import functools
import json
import math
import os
import socket
import subprocess
import sys
import types
import typing
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

__version__ = "0.2.0"
__all__ = [
    "cli",
    "load_config",
    "print_config",
    "to_plain",
    "ConfigError",
]

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Sentinel / exception
# ---------------------------------------------------------------------------

_MISSING: Any = dataclasses.MISSING


def _is_missing(v: Any) -> bool:
    return v is _MISSING


class ConfigError(Exception):
    """Raised on config schema / type errors."""


# ---------------------------------------------------------------------------
# Schema extraction (same as before — drives type conversion)
# ---------------------------------------------------------------------------

_FieldInfo = collections.namedtuple(
    "_FieldInfo", ["type", "default", "default_factory", "metadata"]
)


@lru_cache(maxsize=None)
def _type_hints(cls: type) -> Dict[str, Any]:
    return typing.get_type_hints(cls)


@lru_cache(maxsize=None)
def _field_map(cls: type) -> Dict[str, _FieldInfo]:
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
    """Classify a type for the conversion dispatch."""
    origin = typing.get_origin(t)
    args = typing.get_args(t)

    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return "optional"
        return "union"

    if t is Any:
        return "any"
    if origin is Literal:
        return "literal"
    if origin in (list, List):
        return "list"
    if origin in (dict, Dict):
        return "dict"
    if origin in (tuple, Tuple):
        return "tuple"
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


# ---------------------------------------------------------------------------
# Type conversion + instantiation
# ---------------------------------------------------------------------------


def _convert(value: Any, expected_type: Any, path: str) -> Any:
    """Convert *value* to *expected_type*, raising on mismatch."""
    k = _kind(expected_type)

    # Any → passthrough (must be first — typing.Any breaks isinstance)
    if k == "any":
        return value

    # Already correct type
    if isinstance(expected_type, type) and isinstance(value, expected_type):
        return value

    # None / null
    if value is None or (isinstance(value, str) and value.lower() in ("null", "none")):
        if k == "optional":
            return None
        raise ConfigError(f"{path}: got None/null but field is not Optional")

    # Optional[T] — unwrap
    if k == "optional":
        origin = typing.get_origin(expected_type)
        args = typing.get_args(expected_type)
        if origin is None:
            origin = getattr(expected_type, "__origin__", None)
            args = getattr(expected_type, "__args__", ())
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        return _convert(value, non_none[0] if non_none else str, path)

    # Union
    if k == "union":
        args = typing.get_args(expected_type) or getattr(expected_type, "__args__", ())
        for arg in args:
            try:
                return _convert(value, arg, path)
            except Exception:
                pass
        raise ConfigError(f"{path}: {value!r} does not match Union")

    # list[T]
    if k == "list":
        inner = typing.get_args(expected_type)[0] if typing.get_args(expected_type) else Any
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                value = [x.strip() for x in value.split(",")] if value.strip() else []
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected list, got {type(value).__name__}")
        return [_convert(item, inner, f"{path}[{i}]") for i, item in enumerate(value)]

    # dict[K,V]
    if k == "dict":
        args = typing.get_args(expected_type)
        key_type = args[0] if len(args) > 0 else Any
        val_type = args[1] if len(args) > 1 else Any
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                raise ConfigError(f"{path}: cannot parse dict from string")
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected dict, got {type(value).__name__}")
        return {_convert(k, key_type, f"{path}.keys"): _convert(v, val_type, f"{path}.{k}")
                for k, v in value.items()}

    # tuple
    if k == "tuple":
        args = typing.get_args(expected_type)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                value = [x.strip() for x in value.split(",")]
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected tuple, got {type(value).__name__}")
        if len(args) == 2 and args[1] is ...:
            return tuple(_convert(item, args[0], f"{path}[{i}]") for i, item in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{path}: tuple length mismatch: expected {len(args)}, got {len(value)}")
        return tuple(_convert(item, args[i], f"{path}[{i}]") for i, item in enumerate(value))

    # Literal
    if k == "literal":
        args = typing.get_args(expected_type)
        for arg in args:
            try:
                converted = _convert(value, type(arg), path)
            except Exception:
                continue
            if converted == arg:
                return converted
        raise ConfigError(f"{path}: {value!r} not in Literal{args}")

    # Enum
    if k == "enum":
        if isinstance(value, expected_type):
            return value
        if isinstance(value, str):
            try:
                return expected_type[value]
            except KeyError:
                pass
        for member in expected_type:
            if member.value == value:
                return member
        valid = ", ".join(f"{m.name}={m.value!r}" for m in expected_type)
        raise ConfigError(f"{path}: {value!r} is not a valid {expected_type.__name__} (valid: {valid})")

    # Dataclass
    if k == "dataclass":
        if isinstance(value, expected_type):
            return value
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected mapping for {expected_type.__name__}, got {type(value).__name__}")
        return _instantiate(expected_type, value, path)

    # Path
    if k == "path":
        return Path(str(value))

    # datetime
    if k == "datetime":
        if isinstance(value, str):
            return datetime.datetime.fromisoformat(value)
        if isinstance(value, datetime.datetime):
            return value
        raise ConfigError(f"{path}: expected datetime string")

    # date
    if k == "date":
        if isinstance(value, str):
            return datetime.date.fromisoformat(value)
        if isinstance(value, datetime.date):
            return value
        raise ConfigError(f"{path}: expected date string")

    # bool
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
        raise ConfigError(f"{path}: expected bool, got {value!r}")

    # int
    if k == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            if value == int(value):
                return int(value)
            raise ConfigError(f"{path}: lossy float→int: {value}")
        if isinstance(value, str):
            v = value.strip()
            try:
                return int(v)
            except ValueError:
                pass
            try:
                fv = float(v)
                if fv == int(fv):
                    return int(fv)
                raise ConfigError(f"{path}: lossy float→int: {v!r}")
            except ValueError:
                raise ConfigError(f"{path}: expected int, got {v!r}")
        raise ConfigError(f"{path}: expected int, got {type(value).__name__} {value!r}")

    # float
    if k == "float":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                raise ConfigError(f"{path}: expected float, got {value!r}")
        raise ConfigError(f"{path}: expected float, got {type(value).__name__} {value!r}")

    # str
    if k == "str":
        return str(value)

    # Fallback
    try:
        return expected_type(value)
    except Exception:
        return value


def _instantiate(cls: type, data: dict, path: str) -> Any:
    """Construct a typed dataclass instance from *data*, validating all fields."""
    if not dataclasses.is_dataclass(cls):
        raise ConfigError(f"{path}: {cls.__name__} is not a @dataclass")

    fields_info = _field_map(cls)
    unknown = set(data) - set(fields_info)
    if unknown:
        msgs = []
        for uk in sorted(unknown):
            matches = difflib.get_close_matches(uk, list(fields_info), n=1, cutoff=0.4)
            hint = f" (did you mean {matches[0]!r}?)" if matches else ""
            msgs.append(f"{uk!r}{hint}")
        raise ConfigError(f"{path}: unknown key(s): {', '.join(msgs)}")

    kwargs: Dict[str, Any] = {}
    for name, info in fields_info.items():
        field_path = f"{path}.{name}" if path else name
        if name in data:
            raw = data[name]
            if raw == "???":
                if not _is_missing(info.default) or not _is_missing(info.default_factory):
                    kwargs[name] = (info.default if not _is_missing(info.default)
                                    else info.default_factory())
                else:
                    raise ConfigError(f"required field {field_path!r} has no value")
            else:
                kwargs[name] = _convert(raw, info.type, field_path)
        elif not _is_missing(info.default):
            kwargs[name] = info.default
        elif not _is_missing(info.default_factory):
            kwargs[name] = info.default_factory()
        else:
            raise ConfigError(f"missing required field {field_path!r} (no default)")
    return cls(**kwargs)


def _dict_to_typed(data: dict, cls: type) -> Any:
    """Convert a plain dict (from OmegaConf) to a typed dataclass instance."""
    return _instantiate(cls, data, "")


# ---------------------------------------------------------------------------
# Hydra integration — @cli decorator
# ---------------------------------------------------------------------------


def _extract_config_type(fn: Callable[..., T]) -> Type[Any]:
    """Extract the config type from the first parameter's annotation."""
    hints = typing.get_type_hints(fn)
    params = [k for k in fn.__code__.co_varnames[:fn.__code__.co_argcount]]
    cfg_param = params[0] if params else "cfg"
    config_type = hints.get(cfg_param)
    if config_type is None or not dataclasses.is_dataclass(_unwrap_optional(config_type)):
        raise ConfigError(
            f"@cli requires the first parameter of '{fn.__name__}' "
            f"to be a typed @dataclass. Got: {config_type}"
        )
    return _unwrap_optional(config_type)


def _unwrap_optional(t: Any) -> Any:
    origin = typing.get_origin(t)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(t)
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return non_none[0]
    return t


def cli(
    config_path: str = "conf",
    config_name: str = "config",
    version_base: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: wrap a main function with Hydra CLI + typed config injection.

    The decorated function receives a **typed dataclass instance** (not a
    ``DictConfig``). Hydra handles YAML composition, ``${}`` interpolation,
    CLI override grammar, ``--multirun`` sweeps, and output management.

    Usage::

        @pm.cli(config_path="conf", config_name="base")
        def main(cfg: TrainConfig) -> None:
            print_config(cfg)
            # cfg.model.hidden_dim is fully typed — IDE autocompletion works

    Hydra-compatible CLI::

        python train.py model=large lr=0.001 exp_name=test
        python train.py --multirun lr=1e-4,3e-4,1e-3
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        config_type = _extract_config_type(fn)
        params = [k for k in fn.__code__.co_varnames[:fn.__code__.co_argcount]]
        cfg_param = params[0]

        @functools.wraps(fn)
        @hydra.main(
            config_path=os.path.abspath(config_path),
            config_name=config_name,
            version_base=version_base,
        )
        def wrapper(dict_cfg: DictConfig, *args: Any, **kwargs: Any) -> T:
            # Convert DictConfig → plain dict → typed dataclass
            plain = OmegaConf.to_container(dict_cfg, resolve=True)
            typed_cfg = _dict_to_typed(plain, config_type)
            kwargs[cfg_param] = typed_cfg
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Programmatic API (for notebooks / scripts)
# ---------------------------------------------------------------------------


def load_config(
    cls: Type[T],
    *,
    config_path: str = "conf",
    config_name: str = "config",
    overrides: Optional[List[str]] = None,
    version_base: Optional[str] = None,
) -> T:
    """Load a typed config programmatically — no decorator needed.

    Uses Hydra's ``initialize`` + ``compose`` under the hood.

    Args:
        cls: The top-level ``@dataclass`` config type.
        config_path: Directory containing config YAML files.
        config_name: Primary config filename (without ``.yaml``).
        overrides: Hydra-style override strings.
        version_base: Hydra version_base (default: ``None``).

    Returns:
        A fully typed instance of *cls*.

    Example::

        cfg = load_config(TrainConfig, config_name="base",
                          overrides=["model=large", "lr=0.001"])
        print(cfg.model.hidden_dim)  # 512, fully typed
    """
    if not dataclasses.is_dataclass(cls):
        raise ConfigError(f"{cls.__name__} is not a @dataclass")

    with hydra.initialize_config_dir(
        config_dir=os.path.abspath(config_path),
        version_base=version_base,
    ):
        dict_cfg = hydra.compose(config_name=config_name, overrides=overrides or [])
        plain = OmegaConf.to_container(dict_cfg, resolve=True)
        return typing.cast(T, _dict_to_typed(plain, cls))


# ---------------------------------------------------------------------------
# Color config printing
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def print_config(
    cfg: Any,
    overrides: Optional[List[str]] = None,
    use_color: bool = True,
) -> str:
    """Pretty-print a resolved dataclass config, highlighting overridden values.

    Args:
        cfg: A resolved dataclass config instance.
        overrides: Hydra-style override strings (``["model=large", "lr=0.001"]``).
                   Paths referenced are highlighted in yellow.
        use_color: If ``True`` (default), use ANSI terminal colors.

    Colors:
        - **Green**: default values (unchanged).
        - **Yellow**: values that were explicitly overridden.
        - **Cyan**: section headers.
        - **Dim**: type annotations.
    """
    override_paths: set = set()
    for item in (overrides or []):
        override_paths.update(_override_paths(item))

    lines: List[str] = []
    _print_inner(cfg, "", override_paths, lines, use_color)
    result = "\n".join(lines)

    if overrides:
        summary = [
            "",
            f"{_BOLD}{_YELLOW}Overrides applied:{_RESET}" if use_color else "Overrides applied:",
        ]
        for item in overrides:
            prefix = f"  {_YELLOW}{item}{_RESET}" if use_color else f"  {item}"
            summary.append(prefix)
        result += "\n" + "\n".join(summary)

    print(result)
    return result


def _override_paths(override: str) -> List[str]:
    """Extract dotted paths from a hydra override string."""
    for prefix in ("++", "+", "~"):
        if override.startswith(prefix):
            override = override[len(prefix):]
            break
    if "=" in override:
        key = override.split("=")[0]
    else:
        key = override
    return [key]


def _print_inner(
    obj: Any, path: str, override_paths: set, lines: List[str], use_color: bool,
    indent: int = 0,
) -> None:
    prefix_spacer = "  " * indent
    if dataclasses.is_dataclass(obj):
        name = type(obj).__name__
        hdr = f"{prefix_spacer}{_BOLD}{_CYAN}[{name}]{_RESET}" if use_color else f"{prefix_spacer}[{name}]"
        lines.append(hdr)
        for f in dataclasses.fields(obj):
            child_path = f"{path}.{f.name}" if path else f.name
            _print_inner(getattr(obj, f.name), child_path, override_paths, lines, use_color, indent + 1)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            child_path = f"{path}.{k}" if path else k
            _print_inner(v, child_path, override_paths, lines, use_color, indent)
        return
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _print_inner(item, f"{path}[{i}]", override_paths, lines, use_color, indent)
        return

    field_name = path.split(".")[-1] if "." in path else path
    is_overridden = path in override_paths or _prefix_match(path, override_paths)
    color = _YELLOW if (is_overridden and use_color) else (_GREEN if use_color else "")
    val_repr = _format_value(obj)

    if use_color:
        line = f"{prefix_spacer}{color}{field_name}{_RESET}{_DIM}: {type(obj).__name__} = {_RESET}{color}{val_repr}{_RESET}"
    else:
        line = f"{prefix_spacer}{field_name}: {type(obj).__name__} = {val_repr}"
    if is_overridden and use_color:
        line += f"  {_YELLOW}{_BOLD}# <-- overridden{_RESET}"
    lines.append(line)


def _prefix_match(path: str, paths: set) -> bool:
    parts = path.split(".")
    for i in range(len(parts)):
        if ".".join(parts[: i + 1]) in paths:
            return True
    return False


def _format_value(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        return f"{val:g}"
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
# to_plain
# ---------------------------------------------------------------------------


def to_plain(cfg: Any) -> dict:
    """Convert a dataclass config tree to a plain dict (for YAML dump)."""
    if dataclasses.is_dataclass(cfg):
        return {f.name: to_plain(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)}
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
