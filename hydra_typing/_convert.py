"""Type conversion + instantiation for hydra_typing.

Converts plain dicts (from OmegaConf) into typed dataclass instances,
validating all values against declared field types.

Supports the full Python typing vocabulary — ``Literal``, ``Enum``,
``Union``/``Optional``, ``Path``, ``datetime``, nested dataclasses, etc. —
going well beyond OmegaConf's built-in type support.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import difflib
import enum
import json
import types
import typing
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

_MISSING: Any = dataclasses.MISSING


def _is_missing(v: Any) -> bool:
    return v is _MISSING


_FieldInfo = collections.namedtuple(
    "_FieldInfo", ["type", "default", "default_factory", "metadata"]
)


@lru_cache(maxsize=None)
def _type_hints(cls: type) -> Dict[str, Any]:
    return typing.get_type_hints(cls)


@lru_cache(maxsize=None)
def field_map(cls: type) -> Dict[str, _FieldInfo]:
    """Build the authoritative field map for a dataclass."""
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


def convert(value: Any, expected_type: Any, path: str) -> Any:
    """Convert *value* to *expected_type*, raising ``ConfigError`` on mismatch."""
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
        from hydra_typing import ConfigError
        raise ConfigError(f"{path}: got None/null but field is not Optional")

    # Optional[T] — unwrap
    if k == "optional":
        origin = typing.get_origin(expected_type)
        args = typing.get_args(expected_type)
        if origin is None:
            origin = getattr(expected_type, "__origin__", None)
            args = getattr(expected_type, "__args__", ())
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        return convert(value, non_none[0] if non_none else str, path)

    # Union
    if k == "union":
        args = typing.get_args(expected_type) or getattr(expected_type, "__args__", ())
        from hydra_typing import ConfigError
        for arg in args:
            try:
                return convert(value, arg, path)
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
            from hydra_typing import ConfigError
            raise ConfigError(f"{path}: expected list, got {type(value).__name__}")
        return [convert(item, inner, f"{path}[{i}]") for i, item in enumerate(value)]

    # dict[K,V]
    if k == "dict":
        args = typing.get_args(expected_type)
        key_type = args[0] if len(args) > 0 else Any
        val_type = args[1] if len(args) > 1 else Any
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                from hydra_typing import ConfigError
                raise ConfigError(f"{path}: cannot parse dict from string")
        if not isinstance(value, dict):
            from hydra_typing import ConfigError
            raise ConfigError(f"{path}: expected dict, got {type(value).__name__}")
        return {convert(k, key_type, f"{path}.keys"): convert(v, val_type, f"{path}.{k}")
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
            from hydra_typing import ConfigError
            raise ConfigError(f"{path}: expected tuple, got {type(value).__name__}")
        if len(args) == 2 and args[1] is ...:
            return tuple(convert(item, args[0], f"{path}[{i}]") for i, item in enumerate(value))
        if len(value) != len(args):
            from hydra_typing import ConfigError
            raise ConfigError(f"{path}: tuple length mismatch: expected {len(args)}, got {len(value)}")
        return tuple(convert(item, args[i], f"{path}[{i}]") for i, item in enumerate(value))

    # Literal
    if k == "literal":
        args = typing.get_args(expected_type)
        for arg in args:
            try:
                converted = convert(value, type(arg), path)
            except Exception:
                continue
            if converted == arg:
                return converted
        from hydra_typing import ConfigError
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
        from hydra_typing import ConfigError
        valid = ", ".join(f"{m.name}={m.value!r}" for m in expected_type)
        raise ConfigError(f"{path}: {value!r} is not a valid {expected_type.__name__} (valid: {valid})")

    # Dataclass
    if k == "dataclass":
        if isinstance(value, expected_type):
            return value
        if not isinstance(value, dict):
            from hydra_typing import ConfigError
            raise ConfigError(f"{path}: expected mapping for {expected_type.__name__}, got {type(value).__name__}")
        return instantiate(expected_type, value, path)

    # Path
    if k == "path":
        return Path(str(value))

    # datetime
    if k == "datetime":
        if isinstance(value, str):
            return datetime.datetime.fromisoformat(value)
        if isinstance(value, datetime.datetime):
            return value
        from hydra_typing import ConfigError
        raise ConfigError(f"{path}: expected datetime string")

    # date
    if k == "date":
        if isinstance(value, str):
            return datetime.date.fromisoformat(value)
        if isinstance(value, datetime.date):
            return value
        from hydra_typing import ConfigError
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
        from hydra_typing import ConfigError
        raise ConfigError(f"{path}: expected bool, got {value!r}")

    # int
    if k == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            if value == int(value):
                return int(value)
            from hydra_typing import ConfigError
            raise ConfigError(f"{path}: lossy float->int: {value}")
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
                from hydra_typing import ConfigError
                raise ConfigError(f"{path}: lossy float->int: {v!r}")
            except ValueError:
                from hydra_typing import ConfigError
                raise ConfigError(f"{path}: expected int, got {v!r}")
        from hydra_typing import ConfigError
        raise ConfigError(f"{path}: expected int, got {type(value).__name__} {value!r}")

    # float
    if k == "float":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                from hydra_typing import ConfigError
                raise ConfigError(f"{path}: expected float, got {value!r}")
        from hydra_typing import ConfigError
        raise ConfigError(f"{path}: expected float, got {type(value).__name__} {value!r}")

    # str
    if k == "str":
        return str(value)

    # Fallback
    try:
        return expected_type(value)
    except Exception:
        return value


def instantiate(cls: type, data: dict, path: str) -> Any:
    """Construct a typed dataclass instance from *data*, validating all fields.

    Args:
        cls: The dataclass type.
        data: Resolved plain-dict values (from OmegaConf).
        path: Dotted path prefix for error messages.

    Raises:
        ConfigError: On unknown keys, missing required fields, type mismatches.
    """
    if not dataclasses.is_dataclass(cls):
        from hydra_typing import ConfigError
        raise ConfigError(f"{path}: {cls.__name__} is not a @dataclass")

    fields_info = field_map(cls)
    unknown = set(data) - set(fields_info)
    if unknown:
        from hydra_typing import ConfigError
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
                    from hydra_typing import ConfigError
                    raise ConfigError(f"required field {field_path!r} has no value")
            else:
                kwargs[name] = convert(raw, info.type, field_path)
        elif not _is_missing(info.default):
            kwargs[name] = info.default
        elif not _is_missing(info.default_factory):
            kwargs[name] = info.default_factory()
        else:
            from hydra_typing import ConfigError
            raise ConfigError(f"missing required field {field_path!r} (no default)")
    return cls(**kwargs)


def dict_to_typed(data: dict, cls: type) -> Any:
    """Convert a plain dict (from OmegaConf.to_container) to a typed dataclass."""
    return instantiate(cls, data, "")
