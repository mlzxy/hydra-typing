"""
hydra_typing — Typed Configs for Hydra
=======================================

**One job: make Hydra configs typed.**

Your ``@hydra.main`` function receives a ``DictConfig``.  After one import,
it receives your ``@dataclass`` instance instead — with full IDE
autocompletion, mypy/pyright checking, and all Python types supported
(``Literal``, ``Enum``, ``Union``, ``Path``, ``datetime``, nested
dataclasses, etc.).

Install: ``pip install hydra-typing``  (one dependency: ``hydra-core``).

Quickstart — transparent patch (recommended)
---------------------------------------------

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

import dataclasses
import datetime
import enum
import functools
import os
import types
import typing
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

import hydra
from omegaconf import DictConfig, OmegaConf

# Re-export type conversion (used internally, public for advanced use)
from hydra_typing._convert import dict_to_typed, field_map

__version__ = "0.2.0"
__all__ = [
    "patch",
    "hydra_main",
    "load_config",
    "to_plain",
    "ConfigError",
]

T = TypeVar("T")


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
            f"to be a typed @dataclass. Got: {schema}"
        )
    return _unwrap_optional(schema)


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

    Usage::

        import hydra_typing; hydra_typing.patch()

        @hydra.main(config_path="conf", config_name="base", version_base=None)
        def main(cfg: TrainConfig) -> None:
            # cfg is a typed TrainConfig instance
            print(cfg.model.hidden_dim)  # IDE autocompletion works
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
                return decorator(fn)  # no typed annotation — pass through

            params = [k for k in fn.__code__.co_varnames[:fn.__code__.co_argcount]]
            cfg_param = params[0]

            @functools.wraps(fn)
            def _typed_fn(*args: Any, **kwargs: Any) -> Any:
                if args and isinstance(args[0], DictConfig):
                    plain = OmegaConf.to_container(args[0], resolve=True, enum_to_str=True)
                    args = (dict_to_typed(plain, schema),) + args[1:]
                elif cfg_param in kwargs and isinstance(kwargs[cfg_param], DictConfig):
                    plain = OmegaConf.to_container(kwargs[cfg_param], resolve=True, enum_to_str=True)
                    kwargs[cfg_param] = dict_to_typed(plain, schema)
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
    """Explicit drop-in for ``@hydra.main`` that delivers typed configs.

    Same as ``patch()`` but explicit per-function.  Use when you want to
    opt in selectively rather than patching globally.

    Usage::

        @hydra_main(config_path="conf", config_name="base")
        def main(cfg: TrainConfig) -> None:
            print(cfg.model.hidden_dim)  # typed!
    """
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
            kwargs[cfg_param] = dict_to_typed(plain, actual_schema)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


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
        return typing.cast(T, dict_to_typed(plain, schema))


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


# ---------------------------------------------------------------------------
# Hydra plugin registration
# ---------------------------------------------------------------------------


class HydraTypingPlugin:
    """Hydra plugin — ``hydra-typing`` discoverable via ``--info plugins``."""
    def __init__(self) -> None:
        pass
