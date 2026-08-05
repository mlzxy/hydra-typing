"""
hydra_typing — Typed Configs for Hydra
=======================================

**One job: make Hydra configs typed.**  Your ``@hydra.main`` function
receives a ``DictConfig``.  After one import, it receives your ``@dataclass``
instance instead — with full IDE autocompletion, mypy/pyright checking, and
all Python types supported (``Literal``, ``Enum``, ``Union``, ``Path``,
``datetime``, nested dataclasses, etc.).

Install: ``pip install hydra-typing``  (one dependency: ``hydra-core``).

Transparent patch (recommended)
-------------------------------

.. code-block:: python

    import hydra_typing; hydra_typing.patch()

    @hydra.main(config_path="conf", config_name="base", version_base=None)
    def main(cfg: TrainConfig) -> None:
        # cfg is typed!  No DictConfig, no OmegaConf.
        print(f"Training {cfg.exp_name} with dim={cfg.model.hidden_dim}")

Explicit decorator
------------------

.. code-block:: python

    from hydra_typing import hydra_main

    @hydra_main(config_path="conf", config_name="base")
    def main(cfg: TrainConfig) -> None:
        ...

Programmatic (notebooks)
-------------------------

.. code-block:: python

    from hydra_typing import load_config

    cfg = load_config(TrainConfig, config_name="base",
                      overrides=["model=large", "lr=0.001"])

Overriding nested collections
-----------------------------

.. code-block:: bash

    # List[LayerConfig] — by index
    python train.py model.layers.0.dim=1024 model.layers.1.type=mlp

    # Dict[str, HeadConfig] — by key
    python train.py model.heads.attention.dim=512

    # Append to list
    python train.py +model.layers.2.type=conv +model.layers.2.dim=512

All Hydra features unchanged: ``defaults:`` groups, ``${}`` interpolation,
CLI overrides, ``--multirun`` sweeps, launchers, sweepers, output management.

API
---

.. autofunction:: patch
.. autofunction:: hydra_main
.. autofunction:: load_config
.. autofunction:: to_plain
.. autoexception:: ConfigError
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import difflib
import enum
import functools
import json
import os
import types
import typing
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar

import hydra
from omegaconf import DictConfig, OmegaConf

__version__ = "0.2.0"
__all__ = [
    "patch",
    "hydra_main",
    "load_config",
    "to_plain",
    "to_omegaconf",
    "HydraConfig",
    "RunDir",
    "SweepDir",
    "JobConf",
    "RuntimeConf",
    "ConfigError",
]

T = TypeVar("T")
_MISSING: Any = dataclasses.MISSING


def _is_missing(v: Any) -> bool:
    return v is _MISSING


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised on config schema / type errors."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unwrap_optional(t: Any) -> Any:
    origin = typing.get_origin(t)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(t)
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return non_none[0]
    return t


def _extract_schema(fn: Callable) -> Type[Any]:
    hints = typing.get_type_hints(fn)
    params = [k for k in fn.__code__.co_varnames[:fn.__code__.co_argcount]]
    cfg_param = params[0] if params else "cfg"
    schema = hints.get(cfg_param)
    if schema is None or not dataclasses.is_dataclass(_unwrap_optional(schema)):
        raise ConfigError(
            f"@hydra.main requires the first parameter of '{fn.__name__}' "
            f"to be a typed @dataclass.  Got: {schema}"
        )
    return _unwrap_optional(schema)


# ---------------------------------------------------------------------------
# Schema extraction + type conversion
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


def _convert(value: Any, expected_type: Any, path: str) -> Any:
    """Convert *value* to *expected_type*, raising ``ConfigError`` on mismatch."""
    k = _kind(expected_type)

    if k == "any":
        return value

    if isinstance(expected_type, type) and isinstance(value, expected_type):
        return value

    if value is None or (isinstance(value, str) and value.lower() in ("null", "none")):
        if k == "optional":
            return None
        raise ConfigError(f"{path}: got None/null but field is not Optional")

    if k == "optional":
        origin = typing.get_origin(expected_type)
        args = typing.get_args(expected_type)
        if origin is None:
            origin = getattr(expected_type, "__origin__", None)
            args = getattr(expected_type, "__args__", ())
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        return _convert(value, non_none[0] if non_none else str, path)

    if k == "union":
        args = typing.get_args(expected_type) or getattr(expected_type, "__args__", ())
        for arg in args:
            try:
                return _convert(value, arg, path)
            except Exception:
                pass
        raise ConfigError(f"{path}: {value!r} does not match Union")

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

    if k == "dataclass":
        if isinstance(value, expected_type):
            return value
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected mapping for {expected_type.__name__}, got {type(value).__name__}")
        # If the dict has _target_ but the dataclass doesn't have _target_ as
        # a field, call hydra.utils.instantiate() to create the object.
        # If the dataclass DOES have _target_, treat it as a normal field.
        if "_target_" in value and "_target_" not in _field_map(expected_type):
            return hydra.utils.instantiate(value)
        return _instantiate(expected_type, value, path)

    if k == "path":
        return Path(str(value))

    if k == "datetime":
        if isinstance(value, str):
            return datetime.datetime.fromisoformat(value)
        if isinstance(value, datetime.datetime):
            return value
        raise ConfigError(f"{path}: expected datetime string")

    if k == "date":
        if isinstance(value, str):
            return datetime.date.fromisoformat(value)
        if isinstance(value, datetime.date):
            return value
        raise ConfigError(f"{path}: expected date string")

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

    if k == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            if value == int(value):
                return int(value)
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
                raise ConfigError(f"{path}: lossy float->int: {v!r}")
            except ValueError:
                raise ConfigError(f"{path}: expected int, got {v!r}")
        raise ConfigError(f"{path}: expected int, got {type(value).__name__} {value!r}")

    if k == "float":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                raise ConfigError(f"{path}: expected float, got {value!r}")
        raise ConfigError(f"{path}: expected float, got {type(value).__name__} {value!r}")

    if k == "str":
        return str(value)

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
    """Convert a plain dict (from OmegaConf.to_container) to a typed dataclass."""
    # Auto-populate hydra config if the schema has a 'hydra' field
    _inject_hydra_config(data, cls)
    return _instantiate(cls, data, "")


# ---------------------------------------------------------------------------
# HydraConfig — typed mirror of Hydra's built-in runtime config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunDir:
    """Hydra run output directory config."""
    dir: str = "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"


@dataclasses.dataclass
class SweepDir:
    """Hydra sweep output directory config."""
    dir: str = "multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}"
    subdir: str = "${hydra.job.num}"


@dataclasses.dataclass
class JobConf:
    """Hydra job runtime info."""
    name: str = ""
    chdir: Optional[bool] = None
    override_dirname: str = ""
    id: str = ""
    num: int = 0
    config_name: Optional[str] = None


@dataclasses.dataclass
class RuntimeConf:
    """Hydra runtime info (populated after composition)."""
    version: str = ""
    version_base: str = ""
    cwd: str = ""
    output_dir: str = ""


@dataclasses.dataclass
class HydraConfig:
    """Typed mirror of Hydra's built-in config (``hydra.*`` keys).

    Add this to your top-level config to get typed access to Hydra's
    runtime information::

        @dataclass
        class TrainConfig:
            model: ModelConfig = field(default_factory=ModelConfig)
            lr: float = 3e-4
            hydra: HydraConfig = field(default_factory=HydraConfig)
            # ^^^ populated automatically during load

    Then access::

        cfg.hydra.run.dir          # output directory
        cfg.hydra.job.name         # job name
        cfg.hydra.job.num          # sweep job number
        cfg.hydra.runtime.cwd      # original working directory
        cfg.hydra.overrides.task   # list of CLI override strings
    """
    run: RunDir = dataclasses.field(default_factory=RunDir)
    sweep: SweepDir = dataclasses.field(default_factory=SweepDir)
    job: JobConf = dataclasses.field(default_factory=JobConf)
    runtime: RuntimeConf = dataclasses.field(default_factory=RuntimeConf)
    output_subdir: Optional[str] = ".hydra"
    overrides: Dict[str, List[str]] = dataclasses.field(
        default_factory=lambda: {"task": [], "hydra": []}
    )
    verbose: bool = False


def _safe_get(cfg: Any, attr: str, default: Any = "") -> Any:
    """Get *attr* from OmegaConf config, returning *default* on interpolation errors."""
    try:
        val = getattr(cfg, attr)
        # If it's still an interpolation string that can't resolve, return default
        if isinstance(val, str) and "${" in val:
            return default
        return val
    except Exception:
        return default


def _inject_hydra_config(data: dict, cls: type) -> None:
    """If *cls* has a ``hydra: HydraConfig`` field, populate it from HydraConfig.get()."""
    from hydra_typing import HydraConfig as HC

    hints = typing.get_type_hints(cls)
    if "hydra" not in hints:
        return

    hydra_field_type = hints["hydra"]
    if not dataclasses.is_dataclass(_unwrap_optional(hydra_field_type)):
        return

    try:
        from hydra.core.hydra_config import HydraConfig as H
        hc = H.get()
    except Exception:
        return  # not inside a hydra run

    # Only populate if user hasn't already overridden it via YAML/CLI
    if "hydra" not in data or isinstance(data.get("hydra"), dict):
        hydra_data: dict = data.get("hydra", {}) if isinstance(data.get("hydra"), dict) else {}

        hydra_data.setdefault("run", {})
        hydra_data.setdefault("sweep", {})
        hydra_data.setdefault("job", {})
        hydra_data.setdefault("runtime", {})

        # Run dir — safe access (may contain unresolved ${now:...})
        if hasattr(hc, "run"):
            hydra_data["run"].setdefault("dir", _safe_get(hc.run, "dir", ""))

        # Sweep
        if hasattr(hc, "sweep"):
            hydra_data["sweep"].setdefault("dir", _safe_get(hc.sweep, "dir", ""))
            hydra_data["sweep"].setdefault("subdir", _safe_get(hc.sweep, "subdir", ""))

        # Job
        if hasattr(hc, "job"):
            hydra_data["job"].setdefault("name", _safe_get(hc.job, "name", ""))
            hydra_data["job"].setdefault("num", _safe_get(hc.job, "num", 0))
            hydra_data["job"].setdefault("id", _safe_get(hc.job, "id", ""))
            hydra_data["job"].setdefault("override_dirname", _safe_get(hc.job, "override_dirname", ""))
            hydra_data["job"].setdefault("config_name", _safe_get(hc.job, "config_name", None))

        # Runtime
        if hasattr(hc, "runtime"):
            hydra_data["runtime"].setdefault("version", _safe_get(hc.runtime, "version", ""))
            hydra_data["runtime"].setdefault("cwd", _safe_get(hc.runtime, "cwd", ""))
            hydra_data["runtime"].setdefault("output_dir", _safe_get(hc.runtime, "output_dir", ""))

        # Output subdir
        hydra_data.setdefault("output_subdir", _safe_get(hc, "output_subdir", ".hydra"))

        # Overrides — these are lists, no interpolation issues
        if hasattr(hc, "overrides"):
            hydra_data.setdefault("overrides", {
                "task": list(getattr(hc.overrides, "task", [])),
                "hydra": list(getattr(hc.overrides, "hydra", [])),
            })

        # Verbose
        hydra_data.setdefault("verbose", _safe_get(hc, "verbose", False))

        data["hydra"] = hydra_data


# ---------------------------------------------------------------------------
# patch() — transparently make @hydra.main typed
# ---------------------------------------------------------------------------

_patched = False


def patch() -> None:
    """Patch ``hydra.main`` so ``@hydra.main`` delivers typed configs.

    Call once (typically at import time).  After that, any ``@hydra.main``
    function whose first parameter has a ``@dataclass`` type annotation
    receives a typed instance instead of a ``DictConfig``.

    Functions *without* typed annotations are left alone — no breakage.
    """
    global _patched
    if _patched:
        return
    _patched = True

    _original_main = hydra.main

    def _patched_main(
        config_path: Optional[str] = None,
        config_name: Optional[str] = None,
        version_base: Optional[str] = None,
    ) -> Callable:
        decorator = _original_main(
            config_path=config_path,
            config_name=config_name,
            version_base=version_base,
        )

        def _wrapper(fn: Callable) -> Callable:
            try:
                schema = _extract_schema(fn)
            except ConfigError:
                return decorator(fn)

            params = [k for k in fn.__code__.co_varnames[:fn.__code__.co_argcount]]
            cfg_param = params[0]

            @functools.wraps(fn)
            def _typed_fn(*args: Any, **kwargs: Any) -> Any:
                if args and isinstance(args[0], DictConfig):
                    plain = OmegaConf.to_container(args[0], resolve=True, enum_to_str=True)
                    args = (_dict_to_typed(plain, schema),) + args[1:]
                elif cfg_param in kwargs and isinstance(kwargs[cfg_param], DictConfig):
                    plain = OmegaConf.to_container(kwargs[cfg_param], resolve=True, enum_to_str=True)
                    kwargs[cfg_param] = _dict_to_typed(plain, schema)
                return fn(*args, **kwargs)

            return decorator(_typed_fn)

        return _wrapper

    hydra.main = _patched_main


# ---------------------------------------------------------------------------
# hydra_main — explicit decorator (wraps @hydra.main)
# ---------------------------------------------------------------------------


def hydra_main(
    config_path: Optional[str] = None,
    config_name: Optional[str] = None,
    version_base: Optional[str] = None,
    schema: Optional[Type[Any]] = None,
) -> Callable:
    """Explicit drop-in for ``@hydra.main`` that delivers typed configs."""
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        actual_schema = schema or _extract_schema(fn)
        params = [k for k in fn.__code__.co_varnames[:fn.__code__.co_argcount]]
        cfg_param = params[0]
        resolved_path = config_path or "conf"

        @functools.wraps(fn)
        @hydra.main(
            config_path=os.path.abspath(resolved_path),
            config_name=config_name,
            version_base=version_base,
        )
        def wrapper(dict_cfg: DictConfig, *args: Any, **kwargs: Any) -> T:
            plain = OmegaConf.to_container(dict_cfg, resolve=True, enum_to_str=True)
            kwargs[cfg_param] = _dict_to_typed(plain, actual_schema)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# to_omegaconf — typed dataclass → OmegaConf DictConfig (100% compat fallback)
# ---------------------------------------------------------------------------


def to_omegaconf(cfg: Any) -> Any:
    """Convert a typed dataclass back to an OmegaConf ``DictConfig``.

    This is the **compatibility safety net**: any Hydra feature or
    third-party code that expects an OmegaConf ``DictConfig`` can always
    get one back from your typed config.

    Uses ``to_plain`` → ``OmegaConf.create`` to avoid OmegaConf's
    limited type vocabulary (``Literal``, ``Union``, etc.).  Full
    round-trip guarantees — no data is lost.

    Example::

        cfg = load_config(TrainConfig, config_name="base")
        # ... use typed cfg ...
        oc = to_omegaconf(cfg)
        OmegaConf.save(oc, "exported.yaml")        # any OmegaConf API works
        hydra.utils.instantiate(oc.model.lora)      # nested instantiate, etc.
    """
    return OmegaConf.create(to_plain(cfg))


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------


def load_config(
    schema: Type[T],
    *,
    config_path: str = "conf",
    config_name: str = "config",
    overrides: Optional[List[str]] = None,
    version_base: Optional[str] = None,
) -> T:
    """Load a typed config programmatically (notebooks, scripts).

    Uses Hydra's ``initialize`` + ``compose``, then converts to typed.

    Example::

        cfg = load_config(TrainConfig, config_name="base",
                          overrides=["model=large", "lr=0.001"])
        print(cfg.model.hidden_dim)  # 512, fully typed
    """
    if not dataclasses.is_dataclass(schema):
        raise ConfigError(f"{schema.__name__} is not a @dataclass")

    with hydra.initialize_config_dir(
        config_dir=os.path.abspath(config_path),
        version_base=version_base,
    ):
        dict_cfg = hydra.compose(config_name=config_name, overrides=overrides or [])
        plain = OmegaConf.to_container(dict_cfg, resolve=True, enum_to_str=True)
        return typing.cast(T, _dict_to_typed(plain, schema))


# ---------------------------------------------------------------------------
# to_plain — dataclass → plain dict
# ---------------------------------------------------------------------------


def to_plain(cfg: Any) -> dict:
    """Convert a dataclass config tree to a plain dict (for YAML / JSON dump)."""
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
