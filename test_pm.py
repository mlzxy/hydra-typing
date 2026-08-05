"""Tests for pm — typed config management (hydra + tyro integration)."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pm import ConfigError, load_config, print_config, to_plain  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Test configs
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
# Tests: load_config programmatic API
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

        cfg = load_config(
            SimpleCfg,
            config_path=str(conf),
            config_name="config",
            overrides=["a=99", "b=overridden"],
        )
        assert cfg.a == 99
        assert cfg.b == "overridden"

    def test_nested_config(self, tmp_path):
        conf = tmp_path / "conf"
        conf.mkdir()
        _write_yaml(
            str(conf / "config.yaml"),
            "inner:\n  dim: 128\n  dropout: 0.5\nlr: 1e-4\n",
        )
        cfg = load_config(NestedCfg, config_path=str(conf), config_name="config")
        assert cfg.inner.dim == 128
        assert cfg.inner.dropout == 0.5
        assert cfg.lr == 1e-4

    def test_nested_override(self, tmp_path):
        conf = tmp_path / "conf"
        conf.mkdir()
        _write_yaml(str(conf / "config.yaml"), "inner:\n  dim: 64\nlr: 3e-4\n")
        cfg = load_config(
            NestedCfg,
            config_path=str(conf),
            config_name="config",
            overrides=["inner.dim=256"],
        )
        assert cfg.inner.dim == 256

    def test_non_dataclass_raises(self):
        class NotADataclass:
            pass

        with pytest.raises(ConfigError):
            load_config(NotADataclass, config_path="conf")


# ---------------------------------------------------------------------------
# Tests: type conversion
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
        _write_yaml(str(conf / "config.yaml"), "{}\n")  # empty — name not provided
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
# Tests: print_config
# ---------------------------------------------------------------------------


class TestPrintConfig:
    def test_no_color_output(self):
        cfg = SimpleCfg(a=42, b="test")
        result = print_config(cfg, use_color=False)
        assert "[SimpleCfg]" in result
        assert "a: int = 42" in result
        assert "b: str = 'test'" in result

    def test_highlights_overrides(self):
        cfg = SimpleCfg(a=99, b="overridden")
        result = print_config(cfg, overrides=["a=99"], use_color=False)
        assert "Overrides applied:" in result
        assert "a=99" in result

    def test_nested_print(self):
        cfg = NestedCfg(inner=InnerCfg(dim=128, dropout=0.5), lr=1e-4)
        result = print_config(cfg, use_color=False)
        assert "[NestedCfg]" in result
        assert "[InnerCfg]" in result
        assert "dim: int = 128" in result

    def test_returns_string(self):
        result = print_config(SimpleCfg(), use_color=False)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tests: to_plain
# ---------------------------------------------------------------------------


class TestToPlain:
    def test_simple(self):
        cfg = SimpleCfg(a=99, b="test", c=1.5, flag=False)
        plain = to_plain(cfg)
        assert plain == {"a": 99, "b": "test", "c": 1.5, "flag": False}

    def test_nested(self):
        cfg = NestedCfg(inner=InnerCfg(dim=128, dropout=0.5), lr=1e-4)
        plain = to_plain(cfg)
        assert plain == {"inner": {"dim": 128, "dropout": 0.5}, "lr": 1e-4}

    def test_enum_path(self):
        cfg = RichCfg(scheduler=Scheduler.LINEAR, path=Path("/data"))
        plain = to_plain(cfg)
        assert plain["scheduler"] == "linear"
        assert plain["path"] == "/data"
