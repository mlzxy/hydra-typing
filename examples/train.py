#!/usr/bin/env python3
"""
train.py — Example CLI using hydra_typing (transparent patch mode).

One import makes @hydra.main deliver typed configs.  All Hydra features
work unchanged — YAML composition, ${} interpolation, --multirun, etc.

==== Basic usage (standard Hydra CLI) ==========================================

    python train.py
    python train.py model=large optimizer.lr=0.001 exp_name=big_run
    python train.py model=small data=cifar10 optimizer=sgd
    python train.py model.hidden_dim=1024 +optimizer.momentum=0.9
    python train.py --multirun lr=1e-4,3e-4,1e-3
    python train.py --help

==== Overriding nested fields (dotted paths) ===================================

    python train.py model.hidden_dim=512 optimizer.weight_decay=0.1

==== Overriding list elements (0-indexed) ======================================

    # layers is List[LayerConfig].  Index into it:
    python train.py layers.0.dim=1024 layers.1.type=mlp

    # Append a new element with +:
    python train.py +layers.2.type=conv +layers.2.kernel=3

==== Overriding dict values ====================================================

    # heads is Dict[str, HeadConfig].  Access by key:
    python train.py heads.attention.dim=512 heads.mlp.ratio=8

==== Sweeps over nested fields =================================================

    python train.py --multirun layers.0.dim=256,512,1024
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict

# Make the package importable from the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# One import — transparently makes @hydra.main typed
import hydra  # noqa: E402
import hydra.utils  # noqa: E402
import hydra_typing; hydra_typing.patch()  # noqa: E402, E702
from hydra_typing import HydraConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Config dataclasses — single source of truth
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    """Instantiable config — _target_ is a regular typed field.

    Standard Hydra pattern: include ``_target_`` in the dataclass, then
    ``hydra.utils.instantiate(config)`` calls the target class/function.

    Usage::

        # Access _target_ as a typed field
        print(cfg.model.lora._target_)   # "__main__.LoRAConfig"

        # Deferred instantiate via OmegaConf round-trip
        oc = to_omegaconf(cfg)
        lora_module = hydra.utils.instantiate(oc.model.lora)
    """
    _target_: str = "__main__.LoRAConfig"
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0

    def __post_init__(self) -> None:
        self.scaling = self.alpha / self.rank


@dataclass
class LayerConfig:
    """A single transformer layer."""
    type: Literal["attention", "mlp", "conv"] = "attention"
    dim: int = 256
    heads: int = 8
    dropout: float = 0.0


@dataclass
class HeadConfig:
    """Configuration for a specific head/module."""
    dim: int = 128
    ratio: int = 4


@dataclass
class ModelConfig:
    """Model architecture with nested collections + instantiate.

    Demonstrates List[dataclass], Dict[str, dataclass], and _target_:
      - ``layers``: ordered list of layer configs
      - ``heads``: named dict of head configs
      - ``lora``: instantiated via _target_ (see conf/model/lora.yaml)

    CLI override examples::

        # List elements by index
        python train.py model.layers.0.dim=1024

        # Dict elements by key
        python train.py model.heads.attention.dim=512

        # Instantiate via _target_ — add the lora group
        python train.py +model.lora=model/lora
    """
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    activation: Literal["relu", "gelu", "silu"] = "gelu"
    vocab_size: int = 32000
    max_seq_len: int = 2048
    layers: List[LayerConfig] = field(default_factory=lambda: [
        LayerConfig(type="attention", dim=256, heads=8),
        LayerConfig(type="mlp", dim=512, heads=1),
    ])
    heads: Dict[str, HeadConfig] = field(default_factory=lambda: {
        "attention": HeadConfig(dim=128, ratio=4),
        "mlp": HeadConfig(dim=256, ratio=8),
    })
    lora: Optional[LoRAConfig] = None


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
    warmup_ratio: float = 0.1


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

    Built-in Hydra config is typed too — just add ``hydra: HydraConfig``::

        cfg.hydra.run.dir          # "outputs/2026-08-05/14-30-00"
        cfg.hydra.job.name         # "train"
        cfg.hydra.job.num          # 0 (or sweep index)
        cfg.hydra.runtime.cwd      # original working directory
        cfg.hydra.overrides.task   # ["model=large", "lr=0.001"]
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hydra: HydraConfig = field(default_factory=HydraConfig)  # typed!
    exp_name: str = "default"
    seed: int = 42
    max_steps: int = 10000
    eval_every: int = 500
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


# ---------------------------------------------------------------------------
# Main — vanilla @hydra.main, but cfg is typed thanks to hydra_typing.patch()
# ---------------------------------------------------------------------------


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: TrainConfig) -> None:
    """Train a model — cfg is a fully typed TrainConfig instance."""

    print(f"Experiment: {cfg.exp_name}")
    print(f"Model:      {cfg.model.num_layers} layers, "
          f"hidden_dim={cfg.model.hidden_dim}, "
          f"heads={cfg.model.num_heads}")

    # List[LayerConfig] — fully typed, IDE knows each element is LayerConfig
    for i, layer in enumerate(cfg.model.layers):
        print(f"  layer[{i}]: {layer.type}, dim={layer.dim}, heads={layer.heads}")

    # Dict[str, HeadConfig] — fully typed key-value access
    for name, head in cfg.model.heads.items():
        print(f"  head[{name}]: dim={head.dim}, ratio={head.ratio}")

    print(f"Optimizer:  {cfg.optimizer.name}, lr={cfg.optimizer.lr}, "
          f"warmup_ratio={cfg.optimizer.warmup_ratio}")
    print(f"Data:       {cfg.data.path}, batch={cfg.data.batch_size}")
    print(f"Seed:       {cfg.seed}")

    # Typed access to Hydra's built-in config
    print(f"\nHydra run dir:  {cfg.hydra.run.dir}")
    print(f"Hydra job:      {cfg.hydra.job.name} (#{cfg.hydra.job.num})")
    print(f"Overrides:      {cfg.hydra.overrides.get('task', [])}")

    # _target_ is a regular typed field — accessible like any other
    if cfg.model.lora is not None:
        print(f"LoRA config:    _target_={cfg.model.lora._target_}, "
              f"rank={cfg.model.lora.rank}, "
              f"alpha={cfg.model.lora.alpha}")

        # Deferred instantiate: to_omegaconf → hydra.utils.instantiate, 100% compat
        lora_module = hydra.utils.instantiate(hydra_typing.to_omegaconf(cfg.model.lora))
        print(f"LoRA instance:  type={type(lora_module).__name__}, "
              f"scaling={lora_module.scaling:.2f}")

    print(f"\nTraining for {cfg.max_steps} steps...")


if __name__ == "__main__":
    main()
