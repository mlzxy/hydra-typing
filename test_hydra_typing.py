"""Tests for hydra_typing — typed configs for Hydra."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hydra_typing import ConfigError, load_config, to_plain  # noqa: E402


def _write_yaml(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


class Scheduler(Enum):
    COSINE = "cosine"
    LINEAR = "linear"


@dataclass
class InnerCfg:
    dim: int = 64
    dropout: float = 0.1


@dataclass
class SimpleCfg:
    a: int = 1
    b: str = "hello"
    c: float = 3.14
    flag: bool = True


@dataclass
class NestedCfg:
    inner: InnerCfg = field(default_factory=InnerCfg)
    lr: float = 3e-4


@dataclass
class RichCfg:
    scheduler: Scheduler = Scheduler.COSINE
    mode: Literal["train", "eval"] = "train"
    path: Path = Path("/tmp")
    items: List[int] = field(default_factory=lambda: [1, 2, 3])
    meta: Dict[str, Any] = field(default_factory=dict)
    maybe: Optional[str] = None


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_pure_defaults(self, tmp_path):
        conf = tmp_path / "conf"
        conf.mkdir()
        _write_yaml(str(conf / "config.yaml"), "a: 42\nb: yaml_val\n")
        cfg = load_config(SimpleCfg, config_path=str(conf), config_name="config")
        assert cfg.a == 42
        assert cfg.b == "yaml_val"

    def test_overrides(self, tmp_path):
        conf = tmp_path / "conf"
        conf.mkdir()
        _write_yaml(str(conf / "config.yaml"), "a: 1\nb: base\n")
        cfg = load_config(SimpleCfg, config_path=str(conf), config_name="config",
                          overrides=["a=99", "b=overridden"])
        assert cfg.a == 99
        assert cfg.b == "overridden"

    def test_nested_config(self, tmp_path):
        conf = tmp_path / "conf"
        conf.mkdir()
        _write_yaml(str(conf / "config.yaml"), "inner:\n  dim: 128\n  dropout: 0.5\nlr: 1e-4\n")
        cfg = load_config(NestedCfg, config_path=str(conf), config_name="config")
        assert cfg.inner.dim == 128
        assert cfg.inner.dropout == 0.5
        assert cfg.lr == 1e-4

    def test_nested_override(self, tmp_path):
        conf = tmp_path / "conf"
        conf.mkdir()
        _write_yaml(str(conf / "config.yaml"), "inner:\n  dim: 64\nlr: 3e-4\n")
        cfg = load_config(NestedCfg, config_path=str(conf), config_name="config",
                          overrides=["inner.dim=256"])
        assert cfg.inner.dim == 256

    def test_non_dataclass_raises(self):
        class NotADataclass:
            pass
        with pytest.raises(ConfigError):
            load_config(NotADataclass, config_path="conf")


# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------


class TestTypeConversion:
    def test_enum(self, tmp_path):
        conf = tmp_path / "conf"
        conf.mkdir()
        _write_yaml(str(conf / "config.yaml"), "scheduler: linear\n")
        cfg = load_config(RichCfg, config_path=str(conf), config_name="config")
        assert cfg.scheduler == Scheduler.LINEAR

    def test_literal(self, tmp_path):
        conf = tmp_path / "conf"
        conf.mkdir()
        _write_yaml(str(conf / "config.yaml"), "mode: eval\n")
        cfg = load_config(RichCfg, config_path=str(conf), config_name="config")
        assert cfg.mode == "eval"

    def test_path(self, tmp_path):
        conf = tmp_path / "conf"
        conf.mkdir()
        _write_yaml(str(conf / "config.yaml"), "path: /home/user/data\n")
        cfg = load_config(RichCfg, config_path=str(conf), config_name="config")
        assert cfg.path == Path("/home/user/data")

    def test_optional_none(self, tmp_path):
        conf = tmp_path / "conf"
        conf.mkdir()
        _write_yaml(str(conf / "config.yaml"), "maybe: null\n")
        cfg = load_config(RichCfg, config_path=str(conf), config_name="config")
        assert cfg.maybe is None

    def test_optional_value(self, tmp_path):
        conf = tmp_path / "conf"
        conf.mkdir()
        _write_yaml(str(conf / "config.yaml"), "maybe: hello\n")
        cfg = load_config(RichCfg, config_path=str(conf), config_name="config")
        assert cfg.maybe == "hello"

    def test_missing_required(self, tmp_path):
        @dataclass
        class ReqCfg:
            name: str

        conf = tmp_path / "conf"
        conf.mkdir()
        _write_yaml(str(conf / "config.yaml"), "{}\n")
        with pytest.raises(ConfigError, match="required"):
            load_config(ReqCfg, config_path=str(conf), config_name="config")

    def test_required_filled(self, tmp_path):
        @dataclass
        class ReqCfg:
            name: str

        conf = tmp_path / "conf"
        conf.mkdir()
        _write_yaml(str(conf / "config.yaml"), "name: provided\n")
        cfg = load_config(ReqCfg, config_path=str(conf), config_name="config")
        assert cfg.name == "provided"

    def test_unknown_key(self, tmp_path):
        conf = tmp_path / "conf"
        conf.mkdir()
        _write_yaml(str(conf / "config.yaml"), "a: 1\nbogus: xxx\n")
        with pytest.raises(ConfigError, match="bogus"):
            load_config(SimpleCfg, config_path=str(conf), config_name="config")


# ---------------------------------------------------------------------------
# to_plain
# ---------------------------------------------------------------------------


class TestToPlain:
    def test_simple(self):
        plain = to_plain(SimpleCfg(a=99, b="test", c=1.5, flag=False))
        assert plain == {"a": 99, "b": "test", "c": 1.5, "flag": False}

    def test_nested(self):
        plain = to_plain(NestedCfg(inner=InnerCfg(dim=128, dropout=0.5), lr=1e-4))
        assert plain == {"inner": {"dim": 128, "dropout": 0.5}, "lr": 1e-4}

    def test_enum_path(self):
        plain = to_plain(RichCfg(scheduler=Scheduler.LINEAR, path=Path("/data")))
        assert plain["scheduler"] == "linear"
        assert plain["path"] == "/data"


# ---------------------------------------------------------------------------
# patch()
# ---------------------------------------------------------------------------


class TestPatch:
    def test_patch_idempotent(self):
        from hydra_typing import patch
        import hydra

        patch()
        fn_before = hydra.main
        patch()
        assert hydra.main is fn_before  # second call is no-op

    def test_untyped_function_passthrough(self):
        """patch() should not break functions without typed annotations."""
        from hydra_typing import patch
        import hydra

        patch()

        @hydra.main(config_path="examples/conf", config_name="base", version_base=None)
        def untyped(cfg):
            pass

        assert callable(untyped)
