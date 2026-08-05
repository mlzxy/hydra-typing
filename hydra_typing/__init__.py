"""
hydra_typing — Typed Configs for Hydra
=======================================

**Hydra plugin** that adds type-safe dataclass configs to Hydra.

Drop-in sidecar for existing Hydra projects: use ``@hydra_main`` instead of
``@hydra.main``, and your config arrives as a **fully typed dataclass
instance** — with ``Literal``, ``Enum``, ``Union``, ``Path``, ``datetime``,
nested dataclasses, and every other Python type you declare.

All Hydra features work unchanged: YAML composition, ``defaults:`` groups,
``${}`` interpolation, CLI override grammar, ``--multirun`` sweeps, launchers,
sweepers, output management.

Install: ``pip install hydra-typing`` (depends on ``hydra-core``).

Quickstart
----------

.. code-block:: python

    from dataclasses import dataclass, field
    from hydra_typing import hydra_main, print_config

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

    @hydra_main(config_path="conf", config_name="base")
    def main(cfg: TrainConfig) -> None:
        '''Train a model — cfg is fully typed.'''
        print_config(cfg)
        print(f"Training {cfg.exp_name} with dim={cfg.model.hidden_dim}")

    if __name__ == "__main__":
        main()

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

Config Groups (standard Hydra)
-------------------------------

.. code-block:: yaml

    # conf/base.yaml
    defaults:
      - model: base
      - optimizer: adamw
      - data: imagenet
      - _self_

    exp_name: default
    seed: 42

Output Management (standard Hydra)
-----------------------------------

Each run produces::

    outputs/<date>/<time>/
      .hydra/
        config.yaml     # fully resolved config
        hydra.yaml      # hydra configuration
        overrides.yaml  # applied overrides

API Reference
-------------

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
import sys
import types
import typing
from typing import Any, Callable, List, Optional, Type, TypeVar

import hydra
from omegaconf import DictConfig, OmegaConf

# Re-export public API
from hydra_typing._convert import dict_to_typed, field_map
from hydra_typing._print import print_config, to_plain

__version__ = "0.2.0"
__all__ = [
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
    """Return ``T`` if *t* is ``Optional[T]``, else *t*."""
    origin = typing.get_origin(t)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(t)
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return non_none[0]
    return t


def _extract_schema(fn: Callable) -> Type[Any]:
    """Extract the config type from the first parameter's type annotation."""
    hints = typing.get_type_hints(fn)
    params = [k for k in fn.__code__.co_varnames[:fn.__code__.co_argcount]]
    cfg_param = params[0] if params else "cfg"
    schema = hints.get(cfg_param)
    if schema is None or not dataclasses.is_dataclass(_unwrap_optional(schema)):
        raise ConfigError(
            f"@hydra_main requires the first parameter of '{fn.__name__}' "
            f"to be a typed @dataclass.  Got: {schema}"
        )
    return _unwrap_optional(schema)


# ---------------------------------------------------------------------------
# hydra_main — drop-in replacement for @hydra.main
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
            # cfg is a typed TrainConfig instance — not a DictConfig
            print(cfg.model.hidden_dim)  # IDE autocompletion works!

    The config type is inferred from the function's first parameter annotation
    (``cfg: TrainConfig``).  Or pass ``schema=TrainConfig`` explicitly.

    All standard Hydra features work: ``--multirun``, ``--help``, override
    grammar, ``defaults:`` groups, output management, sweepers, launchers.

    Args:
        config_path: Directory containing config YAML files (default: ``"conf"``).
        config_name: Primary config filename without ``.yaml`` (default: ``"config"``).
        version_base: Hydra version_base (default: ``None``).
        schema: Optional explicit config type. If ``None``, inferred from annotation.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        actual_schema = schema or _extract_schema(fn)
        params = [k for k in fn.__code__.co_varnames[:fn.__code__.co_argcount]]
        cfg_param = params[0]

        # Default config_path relative to the caller's file
        resolved_config_path = config_path
        if resolved_config_path is None:
            resolved_config_path = "conf"

        @functools.wraps(fn)
        @hydra.main(
            config_path=os.path.abspath(resolved_config_path),
            config_name=config_name,
            version_base=version_base,
        )
        def wrapper(dict_cfg: DictConfig, *args: Any, **kwargs: Any) -> T:
            # Convert DictConfig → plain dict → typed dataclass
            plain = OmegaConf.to_container(dict_cfg, resolve=True, enum_to_str=True)
            typed_cfg = dict_to_typed(plain, actual_schema)
            kwargs[cfg_param] = typed_cfg
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Programmatic API (notebooks / scripts — no decorator)
# ---------------------------------------------------------------------------


def load_config(
    schema: Type[T],
    *,
    config_path: str = "conf",
    config_name: str = "config",
    overrides: Optional[List[str]] = None,
    version_base: Optional[str] = None,
) -> T:
    """Load a typed config programmatically — no decorator needed.

    Uses Hydra's ``initialize`` + ``compose`` under the hood, then converts
    the ``DictConfig`` to a typed dataclass instance.

    Args:
        schema: The top-level ``@dataclass`` config type.
        config_path: Directory containing config YAML files.
        config_name: Primary config filename (without ``.yaml``).
        overrides: Hydra-style override strings.
        version_base: Hydra version_base (default: ``None``).

    Returns:
        A fully typed instance of *schema*.

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
# Hydra plugin registration — makes hydra-typing discoverable via --info plugins
# ---------------------------------------------------------------------------


class HydraTypingPlugin:
    """Hydra plugin: enables typed dataclass configs alongside standard Hydra.

    Installing ``hydra-typing`` is sufficient — no code changes needed in
    existing Hydra projects.  Use ``@hydra_main`` instead of ``@hydra.main``
    to opt in to typed configs.
    """

    def __init__(self) -> None:
        pass
