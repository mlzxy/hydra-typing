#!/usr/bin/env python3
"""
train.py — Example CLI for pm (typed config management).

Demonstrates the full workflow:

1. Load a config from YAML files (with ``defaults:`` group composition)
2. Apply CLI overrides (``--model.hidden_dim=512``)
3. Resolve ``${...}`` interpolations (cross-refs, ``env``, ``now``, ``eval``)
4. Print the resolved config with **color highlighting** of overridden values
5. Export full reproducible config + metadata to a timestamped run directory

Usage
-----

.. code-block:: bash

    # Use the base config (model=base, optimizer=adamw, data=imagenet)
    python train.py

    # Override individual fields (dotted paths for nested fields)
    python train.py -o model.hidden_dim=512 model.num_layers=12 exp_name=big_model

    # Combine with a different output directory
    python train.py -o optimizer.lr=0.001 data.batch_size=64 -d runs/hparam_search

    # Quick dry-run: print config only, don't save
    python train.py -o model.hidden_dim=1024 --dry-run

    # Show the config schema
    python train.py --help-config

Config File Layout
------------------

::

    examples/conf/
      base.yaml               # top-level defaults: [{model: base}, {optimizer: adamw}, ...]
      model/
        small.yaml             # hidden_dim=128, num_layers=4
        base.yaml              # hidden_dim=256, num_layers=6
        large.yaml             # hidden_dim=512, num_layers=12
      optimizer/
        adamw.yaml             # AdamW with cosine schedule + warmup
        adam.yaml              # Adam
        sgd.yaml               # SGD with momentum
      data/
        imagenet.yaml          # ImageNet (224x224, batch=32)
        cifar10.yaml           # CIFAR-10 (32x32, batch=128)

The ``base.yaml`` file uses Hydra-style ``defaults:`` lists to compose
config fragments — to swap a component (e.g. model size), edit the YAML
or create a new config file that references a different group::

    # conf/experiments/small_model.yaml
    defaults:
      - model: small       # use conf/model/small.yaml
      - optimizer: adamw   # use conf/optimizer/adamw.yaml
      - data: cifar10      # use conf/data/cifar10.yaml
    exp_name: small_model_experiment

Then run: ``python train.py -c experiments/small_model``

For quick one-off field changes, use dotted-path overrides::

    python train.py -o model.hidden_dim=1024 optimizer.lr=0.001

Overrides
---------

Pass space-separated ``key=value`` pairs after ``-o``::

    -o model.hidden_dim=1024 exp_name=final_run seed=999

Nested fields use dotted paths::

    -o optimizer.lr=0.001 data.batch_size=64

Output
------

Each run produces::

    runs/<exp_name>/<timestamp>/
      config.yaml       # fully resolved config (round-trip safe)
      meta.json         # timestamp, git sha, command, python version
      overrides.txt     # which values were overridden
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Literal

# Add repo root so pm.py is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pm import load_config, print_config, to_plain, render_help, ConfigError  # noqa: E402


# ---------------------------------------------------------------------------
# Config dataclass — must match the YAML schema
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Model architecture."""
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    activation: Literal["relu", "gelu", "silu"] = "gelu"
    vocab_size: int = 32000
    max_seq_len: int = 2048


@dataclass
class OptimizerConfig:
    """Optimizer + LR schedule."""
    name: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    momentum: float = 0.0
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_steps: int = 1000


@dataclass
class DataConfig:
    """Data loading."""
    path: str = "/data/imagenet"
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    augment: bool = True
    image_size: int = 224
    pin_memory: bool = True


@dataclass
class TrainConfig:
    """Top-level training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    exp_name: str = "default"
    seed: int = 42
    max_steps: int = 10000
    eval_every: int = 500
    log_dir: str = "runs/${exp_name}/${now:%Y-%m-%d_%H-%M-%S}"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Typed config management — load YAML, override, resolve, save.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py
  python train.py -o model.hidden_dim=512 model.num_layers=12 exp_name=big_model
  python train.py -o optimizer.lr=0.001 data.batch_size=64 -d runs/hparam_search
  python train.py -o model.hidden_dim=1024 --dry-run
  python train.py --help-config
        """,
    )

    parser.add_argument(
        "-c", "--config",
        default="base",
        help="Config file name (looked up in conf/<name>.yaml). Default: base",
    )
    parser.add_argument(
        "-o", "--overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. -o model.hidden_dim=512 optimizer.lr=0.001",
    )
    parser.add_argument(
        "-d", "--output-dir",
        default=None,
        help="Output directory for saving config + metadata. "
             "Overrides the log_dir field in config. Supports ${} templates.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config to terminal only, do not save to disk.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )
    parser.add_argument(
        "--help-config",
        action="store_true",
        help="Show the config schema and exit.",
    )

    args = parser.parse_args()

    # Show schema and exit
    if args.help_config:
        print(render_help(TrainConfig))
        return

    # Resolve config file path
    conf_dir = Path(__file__).resolve().parent / "conf"
    config_path = conf_dir / f"{args.config}.yaml"
    if not config_path.exists():
        print(f"Error: config '{args.config}' not found at {config_path}")
        print(f"Available configs: {', '.join(sorted(f.stem for f in conf_dir.glob('*.yaml')))}")
        sys.exit(1)

    # Resolve output directory
    output_dir = args.output_dir

    # Load config
    print(f"Loading config: {config_path.name}")
    if args.overrides:
        print(f"Overrides: {' '.join(args.overrides)}")
    print()

    try:
        cfg = load_config(
            TrainConfig,
            config_files=[str(config_path)],
            overrides=args.overrides,
            output_dir=output_dir,
            save=not args.dry_run,
        )
    except ConfigError as e:
        print(f"Config error: {e}")
        sys.exit(1)

    # Print with color highlighting
    override_map = _parse_override_map(args.overrides)
    print_config(cfg, overrides=override_map, use_color=not args.no_color)

    # Show output path
    from pm import last_output_dir  # noqa: E402

    if last_output_dir:
        print()
        print(f"Config saved to: {last_output_dir}")
    elif args.dry_run:
        print()
        print("(dry run — nothing saved to disk)")


def _parse_override_map(overrides: List[str]) -> dict:
    """Parse a list of 'key=value' overrides into a dict."""
    result = {}
    for item in overrides:
        if "=" in item:
            k, _, v = item.partition("=")
            result[k] = v
    return result


if __name__ == "__main__":
    main()
