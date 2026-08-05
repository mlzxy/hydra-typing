"""
hydra_typing — Typed Configs for Hydra
=======================================

**Hydra sidecar** that adds type-safe dataclass configs to Hydra.

Two ways to use it, both equally transparent:

1. **One-liner patch** (transparent — keep your ``@hydra.main``)::

    import hydra_typing; hydra_typing.patch()

    @hydra.main(config_path="conf", config_name="base", version_base=None)
    def main(cfg: TrainConfig) -> None:
        '''cfg is a typed TrainConfig — not a DictConfig.'''
        print_config(cfg)                       # auto-detects overrides
        print(f"Training {cfg.exp_name}")

2. **Decorator** (explicit drop-in for ``@hydra.main``)::

    from hydra_typing import hydra_main

    @hydra_main(config_path="conf", config_name="base")
    def main(cfg: TrainConfig) -> None:
        ...

All Hydra features work unchanged: YAML composition, ``defaults:`` groups,
``${}`` interpolation, CLI override grammar, ``--multirun`` sweeps, launchers,
sweepers, output management.

Install: ``pip install hydra-typing`` (one dependency: ``hydra-core``).

Run with standard Hydra CLI::

    python train.py model=large lr=0.001 exp_name=test
    python train.py model.hidden_dim=512 +optimizer.momentum=0.9
    python train.py --multirun lr=1e-4,3e-4,1e-3
    python train.py --help

For programmatic use (notebooks, scripts)::

    from hydra_typing import load_config
    cfg = load_config(TrainConfig, config_name="base",
                      overrides=["model=large", "lr=0.001"])

How it works
------------

1. **Schema extraction**: the config type is inferred from your function's
   type annotation (``cfg: TrainConfig``). No separate schema registration.

2. **Hydra does its thing**: YAML composition, ``defaults:`` groups,
   ``${}`` interpolation, CLI parsing, output management — all standard.

3. **Typed conversion**: after Hydra composes the ``DictConfig``, we convert
   it to your typed dataclass via ``OmegaConf.to_container`` + type validation
   supporting the full Python typing vocabulary.

4. **Your code**: receives a real, typed Python object. IDE autocompletion,
   mypy/pyright checking, and attribute-access all work.

``print_config`` auto-detects overrides from ``HydraConfig`` — call it
with no arguments::

    print_config(cfg)  # highlights overridden values automatically

CLI Override Grammar (standard Hydra)
--------------------------------------

=======  ===============================  ==============================
Syntax   Meaning                           Example
=======  ===============================  ==============================
``k=v``  Override config value             ``model.hidden_dim=512``
``+k=v`` Append new key                    ``+optimizer.momentum=0.9``
``++k=v`` Force-set (override if exists)   ``++seed=42``
``~k``   Delete a key                      ``~wandb_project``
``a,b``  Sweep (with ``--multirun``)       ``lr=1e-4,3e-4,1e-3``
=======  ===============================  ==============================

API Reference
-------------

.. autofunction:: patch
.. autofunction:: hydra_main
.. autofunction:: load_config
.. autofunction:: print_config
.. autofunction:: to_plain
.. autoexception:: ConfigError
"""

from __future__ import annotations

import dataclasses
import functools
import os
import types
import typing
from typing import Any, Callable, List, Optional, Type, TypeVar

import hydra
from omegaconf import DictConfig, OmegaConf

# Re-export
from hydra_typing._convert import dict_to_typed, field_map
from hydra_typing._print import to_plain, _print_config

__version__ = "0.2.0"
__all__ = [
    "patch",
    "hydra_main",
    "load_config",
    "print_config",
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


def _get_overrides() -> List[str]:
    """Auto-detect applied overrides from HydraConfig (if inside a hydra run)."""
    try:
        from hydra.core.hydra_config import HydraConfig
        return HydraConfig.get().overrides.task
    except Exception:
        return []


# ---------------------------------------------------------------------------
# print_config — auto-detects overrides from HydraConfig
# ---------------------------------------------------------------------------


def print_config(
    cfg: Any,
    overrides: Optional[List[str]] = None,
    use_color: bool = True,
) -> str:
    """Pretty-print a resolved dataclass config, highlighting overridden values.

    Overrides are **auto-detected** from ``HydraConfig`` when called inside a
    hydra run — just call ``print_config(cfg)`` with no overrides argument.

    Args:
        cfg: A resolved dataclass config instance.
        overrides: Optional override strings. If ``None``, auto-detected.
        use_color: If ``True`` (default), use ANSI terminal colors.

    Colors: **green** = defaults, **yellow** = overridden, **cyan** = headers.
    """
    if overrides is None:
        overrides = _get_overrides()
    return _print_config(cfg, overrides=overrides, use_color=use_color)


# ---------------------------------------------------------------------------
# patch() — transparently make @hydra.main typed
# ---------------------------------------------------------------------------


_patched = False


def patch() -> None:
    """Patch ``hydra.main`` so ``@hydra.main`` delivers typed configs.

    After calling this once (typically at import time), your existing
    ``@hydra.main`` code works unchanged — but the config object passed
    to your function is a **typed dataclass instance** instead of a
    ``DictConfig``.

    Usage::

        import hydra_typing; hydra_typing.patch()

        @hydra.main(config_path="conf", config_name="base", version_base=None)
        def main(cfg: TrainConfig) -> None:
            # cfg is typed! IDE autocompletion works.
            print_config(cfg)

    The schema is inferred from the function's type annotation
    (``cfg: TrainConfig``).  If the annotation is missing or not a
    ``@dataclass``, the function is left unpatched — no breakage.
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
            # Try to extract schema from annotation
            try:
                schema = _extract_schema(fn)
            except ConfigError:
                # No typed annotation — pass through unchanged
                return decorator(fn)

            params = [k for k in fn.__code__.co_varnames[:fn.__code__.co_argcount]]
            cfg_param = params[0]

            # Wrap fn: intercept DictConfig, convert to typed, pass to original
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
    """Drop-in replacement for ``@hydra.main`` that delivers typed configs.

    Usage::

        @hydra_main(config_path="conf", config_name="base")
        def main(cfg: TrainConfig) -> None:
            print(cfg.model.hidden_dim)  # typed!

    All standard Hydra features work unchanged.
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

    Uses Hydra's ``initialize`` + ``compose``, then converts to typed dataclass.

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
# Hydra plugin registration
# ---------------------------------------------------------------------------


class HydraTypingPlugin:
    """Hydra plugin — ``hydra-typing`` discoverable via ``--info plugins``."""
    def __init__(self) -> None:
        pass
