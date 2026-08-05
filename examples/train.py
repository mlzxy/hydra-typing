#!/usr/bin/env python3
"""
train.py — Example CLI using pm (hydra + typed dataclass bridge).

Run with Hydra-compatible syntax — no argparse, no ``-o`` flags::

    python train.py
    python train.py model=large optimizer.lr=0.001 exp_name=big_run
    python train.py model=small data=cifar10 optimizer=sgd
    python train.py model.hidden_dim=1024 +optimizer.momentum=0.9
    python train.py --multirun lr=1e-4,3e-4,1e-3
    python train.py --help

Output goes to ``outputs/<date>/<time>/`` with:
  - ``.hydra/config.yaml`` — fully resolved config
  - ``.hydra/hydra.yaml`` — hydra configuration
  - ``.hydra/overrides.yaml`` — applied overrides
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal

# Add repo root so pm.py is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pm import cli, print_config, ConfigError  # noqa: E402


# ---------------------------------------------------------------------------
# Config dataclasses — single source of truth for schema + typing + CLI
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
    warmup_ratio: float = 0.1  # fraction of max_steps for warmup


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
    """Top-level training configuration.

    Run with::

        python train.py model=large lr=0.001 exp_name=my_run
        python train.py --multirun lr=1e-4,3e-4,1e-3
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    exp_name: str = "default"
    seed: int = 42
    max_steps: int = 10000
    eval_every: int = 500
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


# ---------------------------------------------------------------------------
# Main — decorated with @pm.cli for typed config + hydra CLI
# ---------------------------------------------------------------------------

@cli(config_path="conf", config_name="base")
def main(cfg: TrainConfig) -> None:
    """Train a model — cfg is a fully typed TrainConfig instance."""

    # Get the overrides Hydra applied (from the command line)
    overrides = sys.argv[1:]

    # Color-printed config — highlights what was overridden
    print_config(cfg, overrides=overrides if overrides else None, use_color=True)

    # --- Your training code goes here ---
    print()
    print(f"Experiment: {cfg.exp_name}")
    print(f"Model:      {cfg.model.num_layers} layers, hidden_dim={cfg.model.hidden_dim}")
    print(f"Optimizer:  {cfg.optimizer.name}, lr={cfg.optimizer.lr}, "
          f"warmup_ratio={cfg.optimizer.warmup_ratio}")
    print(f"Data:       {cfg.data.path}, batch={cfg.data.batch_size}, "
          f"image_size={cfg.data.image_size}")
    print(f"Seed:       {cfg.seed}")
    print(f"WandB:      {cfg.wandb_project or '(disabled)'}")

    # Simulate training
    print(f"\nTraining for {cfg.max_steps} steps...")
    print("(replace this with your actual training loop)")


if __name__ == "__main__":
    main()
