# hydra-typing

**Typed dataclass configs for Hydra — so both you and your AI read the config easier.**

[![PyPI](https://img.shields.io/pypi/v/hydra-typing)](https://pypi.org/project/hydra-typing/)

Your `@hydra.main` function receives an untyped `DictConfig`. With `hydra-typing`, it receives your `@dataclass` instance instead — full IDE autocompletion, mypy/pyright checking, and all Python types supported (`Literal`, `Enum`, `Union`, `Path`, `datetime`, nested dataclasses, `List[Dataclass]`, `Dict[str, Dataclass]`, etc.).

All Hydra features work unchanged — `defaults:` groups, `${}` interpolation, CLI overrides, `--multirun`, sweepers, launchers, output management.

## Install

```bash
pip install hydra-typing
```

One dependency: `hydra-core`.

## Usage

**Transparent patch (recommended)** — keep your `@hydra.main`:

```python
import hydra
import hydra_typing; hydra_typing.patch()

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: TrainConfig) -> None:
    # cfg is typed!  No DictConfig, no OmegaConf.
    print(cfg.model.hidden_dim)  # IDE autocompletion works
```

**Explicit decorator:**

```python
from hydra_typing import hydra_main

@hydra_main(config_path="conf", config_name="config")
def main(cfg: TrainConfig) -> None:
    ...
```

**Programmatic (notebooks, scripts):**

```python
from hydra_typing import load_config

cfg = load_config(TrainConfig, config_name="base",
                  overrides=["model=large", "lr=0.001"])
```

## Features

- **Typed configs** — real `@dataclass` instances, not `DictConfig`
- **Full Python type support** — `Literal`, `Enum`, `Union`, `Path`, `datetime`, nested dataclasses, `List[Dataclass]`, `Dict[str, Dataclass]`
- **`HydraConfig`** — typed access to Hydra's built-in runtime config (`run.dir`, `job.name`, `overrides.task`, etc.)
- **`_target_` / `instantiate`** — standard Hydra `_target_` pattern works as a typed field
- **`to_omegaconf()`** — 100% compatibility fallback: convert typed config back to OmegaConf `DictConfig`
- **Non-invasive** — functions without type annotations pass through unchanged
- **Single file** — `hydra_typing.py`, ~700 lines, one dependency

## Incremental adoption

You don't need dataclasses to start.  Add types gradually, at your own pace:

```python
# Step 1 — add one import, nothing changes
import hydra_typing; hydra_typing.patch()

@hydra.main(...)
def main(cfg):                    # ← still untyped, works as before
    print(cfg.model.hidden_dim)   # DictConfig access unchanged
```

```python
# Step 2 — add a dataclass with just the fields you care about
@dataclass
class TrainConfig:
    model: Any = None             # Any = accept whatever YAML gives you
    lr: float = 3e-4

@hydra.main(...)
def main(cfg: TrainConfig):       # ← now typed for lr, model still flexible
    cfg.lr                         # float, IDE-completes
    cfg.model.hidden_dim           # Any, still works
```

```python
# Step 3 — narrow types as you go
@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_layers: int = 6

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)  # ← fully typed
    lr: float = 3e-4
```

No big rewrite.  One field at a time.  The patch passes through anything you haven't typed yet.

## Complex nested configs

```python
@dataclass
class LayerConfig:
    type: Literal["attention", "mlp"] = "attention"
    dim: int = 256

@dataclass
class TrainConfig:
    layers: List[LayerConfig] = field(default_factory=lambda: [
        LayerConfig(type="attention", dim=256),
        LayerConfig(type="mlp", dim=512),
    ])
```

CLI overrides for nested collections:

```bash
# List elements by index
python train.py model.layers.0.dim=1024

# Dict elements by key
python train.py model.heads.attention.dim=512

# Append to list
python train.py +model.layers.2.type=conv +model.layers.2.dim=512
```

## `_target_` / instantiate

```python
@dataclass
class LoRAConfig:
    _target_: str = "__main__.LoRAConfig"
    rank: int = 8
    alpha: int = 16

    def __post_init__(self):
        self.scaling = self.alpha / self.rank

# Deferred instantiate via OmegaConf round-trip (100% compat)
import hydra.utils
oc = hydra_typing.to_omegaconf(cfg.model.lora)
lora = hydra.utils.instantiate(oc)
```

## `HydraConfig` — typed built-in config

```python
@dataclass
class TrainConfig:
    hydra: HydraConfig = field(default_factory=HydraConfig)

# Auto-populated:
cfg.hydra.run.dir          # "outputs/2026-08-05/15-24-20"
cfg.hydra.job.name         # "train"
cfg.hydra.overrides.task   # ["model=large", "lr=0.001"]
```

## vs Hydra

| | hydra | hydra-typing |
|---|---|---|
| Config object | `DictConfig` | typed `@dataclass` |
| IDE autocomplete | limited | full |
| `Literal`, `Union` | unsupported | supported |
| `Path`, `datetime` | unsupported | supported |
| YAML composition | yes | yes (unchanged) |
| CLI overrides | yes | yes (unchanged) |
| `--multirun` | yes | yes (unchanged) |
| `_target_` / instantiate | yes | yes |
| Output management | yes | yes (unchanged) |

## API

```python
hydra_typing.patch()           # make @hydra.main typed (call once)
hydra_typing.hydra_main(...)   # explicit decorator
hydra_typing.load_config(...)  # programmatic (notebooks)
hydra_typing.to_plain(cfg)     # dataclass → dict
hydra_typing.to_omegaconf(cfg) # dataclass → OmegaConf DictConfig (100% compat)
```

## About

This project was built to scratch a personal itch: I wanted Hydra's YAML composition and CLI, but with real typed configs I can trust my IDE with. I'm not yet writing the actual training code — but I want the config management to be clean from day one.

**This code was written entirely by [Claude Code](https://claude.ai/code) (Anthropic) using the DeepSeek API.** I acted as the product manager — specifying what the library should do, reviewing the output, and iterating. The implementation, tests, examples, and documentation were all generated by Claude.

## License

MIT
