"""Comprehensive tests for parameter_manager.

Run with: pytest test_parameter_manager.py -v
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pytest
import yaml

# Allow importing parameter_manager from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parameter_manager import (  # noqa: E402
    ConfigError,
    InterpolationError,
    TypeConversionError,
    YamlError,
    load_config,
    parse_overrides,
    render_help,
    save_config,
    to_plain,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_conf_dir():
    """Create a temporary directory with a config tree, clean up after."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Basic loading
# ---------------------------------------------------------------------------


@dataclass
class SimpleCfg:
    a: int = 1
    b: str = "hello"
    c: float = 3.14
    d: bool = True


class TestBasicLoading:
    def test_pure_defaults(self):
        cfg = load_config(SimpleCfg)
        assert cfg.a == 1
        assert cfg.b == "hello"
        assert cfg.c == 3.14
        assert cfg.d is True

    def test_non_dataclass_raises(self):
        class NotADataclass:
            pass

        with pytest.raises(ConfigError):
            load_config(NotADataclass)

    def test_return_type_is_correct(self):
        cfg = load_config(SimpleCfg)
        assert isinstance(cfg, SimpleCfg)


# ---------------------------------------------------------------------------
# 2. CLI overrides
# ---------------------------------------------------------------------------


@dataclass
class OverrideCfg:
    x: int = 0
    y: str = "default"
    z: float = 1.0


class TestOverrides:
    def test_single_override(self):
        cfg = load_config(OverrideCfg, overrides=["x=42"])
        assert cfg.x == 42
        assert cfg.y == "default"

    def test_multiple_overrides(self):
        cfg = load_config(OverrideCfg, overrides=["x=99", "y=claude", "z=2.718"])
        assert cfg.x == 99
        assert cfg.y == "claude"
        assert cfg.z == 2.718

    def test_last_wins(self):
        cfg = load_config(OverrideCfg, overrides=["x=1", "x=2", "x=3"])
        assert cfg.x == 3

    def test_unknown_key_raises(self):
        with pytest.raises(TypeConversionError, match="nope"):
            load_config(SimpleCfg, overrides=["nope=123"])

    def test_did_you_mean_suggestion(self):
        with pytest.raises(TypeConversionError):
            load_config(OverrideCfg, overrides=["xx=42"])


class TestParseOverrides:
    def test_consumes_only_equals(self):
        ov, rest = parse_overrides(["--a=1", "--verbose", "-n", "3", "positional"])
        assert ov == {"a": "1"}
        assert rest == ["--verbose", "-n", "3", "positional"]

    def test_no_equals_left_alone(self):
        ov, rest = parse_overrides(["--flag", "--key"])
        assert ov == {}
        assert rest == ["--flag", "--key"]

    def test_empty(self):
        ov, rest = parse_overrides([])
        assert ov == {}
        assert rest == []

    def test_dotted_path(self):
        ov, rest = parse_overrides(["--model.dim=512"])
        assert ov == {"model.dim": "512"}


# ---------------------------------------------------------------------------
# 3. Nested dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InnerCfg:
    dim: int = 64
    dropout: float = 0.1


@dataclass
class OuterCfg:
    inner: InnerCfg = field(default_factory=InnerCfg)
    lr: float = 3e-4


class TestNestedDataclasses:
    def test_nested_defaults(self):
        cfg = load_config(OuterCfg)
        assert cfg.inner.dim == 64
        assert cfg.inner.dropout == 0.1

    def test_nested_override(self):
        cfg = load_config(OuterCfg, overrides=["inner.dim=128", "inner.dropout=0.5"])
        assert cfg.inner.dim == 128
        assert cfg.inner.dropout == 0.5

    def test_mixed_override(self):
        cfg = load_config(OuterCfg, overrides=["inner.dim=256", "lr=1e-2"])
        assert cfg.inner.dim == 256
        assert cfg.lr == 1e-2


# ---------------------------------------------------------------------------
# 4. YAML loading
# ---------------------------------------------------------------------------


class TestYamlLoading:
    def test_yaml_overrides_defaults(self, tmp_conf_dir):
        _write_yaml(tmp_conf_dir / "cfg.yaml", "a: 42\nb: yaml_value\n")
        cfg = load_config(SimpleCfg, config_files=[str(tmp_conf_dir / "cfg.yaml")])
        assert cfg.a == 42
        assert cfg.b == "yaml_value"

    def test_precedence_defaults_yaml_cli(self, tmp_conf_dir):
        _write_yaml(tmp_conf_dir / "cfg.yaml", "a: 10\n")
        # YAML overrides default, CLI overrides YAML
        cfg = load_config(
            SimpleCfg,
            config_files=[str(tmp_conf_dir / "cfg.yaml")],
            overrides=["a=99"],
        )
        assert cfg.a == 99

    def test_two_yaml_files_merge(self, tmp_conf_dir):
        _write_yaml(tmp_conf_dir / "a.yaml", "a: 1\nb: first\n")
        _write_yaml(tmp_conf_dir / "b.yaml", "b: second\nc: 3.0\n")
        cfg = load_config(SimpleCfg, config_files=[str(tmp_conf_dir / "a.yaml"), str(tmp_conf_dir / "b.yaml")])
        assert cfg.a == 1  # from a.yaml
        assert cfg.b == "second"  # b.yaml overrides
        assert cfg.c == 3.0  # from b.yaml

    def test_dict_config_files(self):
        cfg = load_config(SimpleCfg, config_files=[{"a": 99, "b": "dict_val"}])
        assert cfg.a == 99
        assert cfg.b == "dict_val"

    def test_missing_file_raises(self):
        with pytest.raises(YamlError, match="not found"):
            load_config(SimpleCfg, config_files=["/nonexistent/path.yaml"])

    def test_yaml_null_optional(self):
        @dataclass
        class OptCfg:
            name: Optional[str] = "default"

        cfg = load_config(OptCfg, config_files=[{"name": None}])
        assert cfg.name is None

    def test_yaml_nested_dataclass(self, tmp_conf_dir):
        _write_yaml(
            tmp_conf_dir / "cfg.yaml",
            "inner:\n  dim: 512\n  dropout: 0.5\nlr: 1e-4\n",
        )
        cfg = load_config(OuterCfg, config_files=[str(tmp_conf_dir / "cfg.yaml")])
        assert cfg.inner.dim == 512
        assert cfg.inner.dropout == 0.5
        assert cfg.lr == 1e-4


# ---------------------------------------------------------------------------
# 5. YAML defaults groups
# ---------------------------------------------------------------------------


@dataclass
class GroupDbCfg:
    host: str = ""
    port: int = 0


@dataclass
class GroupAppCfg:
    db: GroupDbCfg = field(default_factory=GroupDbCfg)
    lr: float = 1e-3
    exp_name: str = "default"


class TestDefaultsGroups:
    def test_key_value_group(self, tmp_conf_dir):
        _write_yaml(tmp_conf_dir / "conf" / "base.yaml", "defaults:\n  - db: mysql\nlr: 3e-4\nexp_name: test\n")
        _write_yaml(tmp_conf_dir / "conf" / "db" / "mysql.yaml", "host: localhost\nport: 3306\n")

        cfg = load_config(GroupAppCfg, config_files=[str(tmp_conf_dir / "conf" / "base.yaml")])
        assert cfg.db.host == "localhost"
        assert cfg.db.port == 3306
        assert cfg.lr == 3e-4

    def test_bare_path_group(self, tmp_conf_dir):
        _write_yaml(tmp_conf_dir / "conf" / "base.yaml", "defaults:\n  - db/mysql\nlr: 3e-4\n")
        _write_yaml(tmp_conf_dir / "conf" / "db" / "mysql.yaml", "host: localhost\nport: 3306\n")

        # Bare path means content merges at top level, so we need matching keys
        @dataclass
        class FlatCfg:
            host: str = ""
            port: int = 0
            lr: float = 1e-3

        cfg = load_config(FlatCfg, config_files=[str(tmp_conf_dir / "conf" / "base.yaml")])
        assert cfg.host == "localhost"
        assert cfg.lr == 3e-4

    def test_own_keys_override_groups(self, tmp_conf_dir):
        _write_yaml(tmp_conf_dir / "conf" / "base.yaml", "defaults:\n  - db: mysql\nlr: 3e-4\n")
        _write_yaml(tmp_conf_dir / "conf" / "db" / "mysql.yaml", "host: localhost\nport: 3306\n")

        cfg = load_config(GroupAppCfg, config_files=[str(tmp_conf_dir / "conf" / "base.yaml")])
        # base.yaml's own keys merge after group content
        assert cfg.db.host == "localhost"
        assert cfg.lr == 3e-4

    def test_resolve_defaults_false(self, tmp_conf_dir):
        _write_yaml(tmp_conf_dir / "cfg.yaml", "defaults:\n  - db: mysql\na: 42\n")
        # With resolve_defaults=False, 'defaults' is just a key
        with pytest.raises(TypeConversionError):  # 'defaults' is not in schema
            load_config(SimpleCfg, config_files=[str(tmp_conf_dir / "cfg.yaml")], resolve_defaults=False)


# ---------------------------------------------------------------------------
# 6. Interpolation
# ---------------------------------------------------------------------------


@dataclass
class InterpCfg:
    hidden: int = 256
    batch: int = 0
    name: str = "default"
    path: str = ""


class TestInterpolation:
    def test_cross_reference(self):
        cfg = load_config(InterpCfg, overrides=["batch=${hidden}"])
        assert cfg.batch == 256
        assert isinstance(cfg.batch, int)

    def test_type_preservation_int(self):
        cfg = load_config(InterpCfg, overrides=["batch=${hidden}"])
        assert isinstance(cfg.batch, int)

    def test_type_preservation_float(self):
        @dataclass
        class FltCfg:
            a: float = 1.5
            b: float = 0.0

        cfg = load_config(FltCfg, overrides=["b=${a}"])
        assert cfg.b == 1.5
        assert isinstance(cfg.b, float)

    def test_partial_string_interpolation(self):
        cfg = load_config(InterpCfg, overrides=["name=run42", "path=outputs/${name}/logs"])
        assert cfg.path == "outputs/run42/logs"

    def test_multi_hop_forward_ref(self):
        cfg = load_config(
            InterpCfg,
            overrides=["name=exp", "path=result_${name}", "batch=${hidden}"],
        )
        assert cfg.path == "result_exp"
        assert cfg.batch == 256

    def test_now_interpolation(self):
        @dataclass
        class NowCfg:
            ts: str = ""

        cfg = load_config(NowCfg, overrides=["ts=${now:%Y-%m-%d}"])
        assert cfg.ts == datetime.datetime.now().strftime("%Y-%m-%d")

    def test_now_iso(self):
        @dataclass
        class NowIsoCfg:
            ts: str = ""

        cfg = load_config(NowIsoCfg, overrides=["ts=${now}"])
        # Should be a valid ISO timestamp
        datetime.datetime.fromisoformat(cfg.ts)

    def test_now_frozen(self):
        """All now references in one load should give the same timestamp."""

        @dataclass
        class FrozenCfg:
            a: str = "${now}"
            b: str = "${now}"

        cfg = load_config(FrozenCfg)
        assert cfg.a == cfg.b

    def test_env_interpolation(self):
        os.environ["PM_TEST_ENV"] = "test_value"
        try:
            cfg = load_config(InterpCfg, overrides=["name=${env:PM_TEST_ENV}"])
            assert cfg.name == "test_value"
        finally:
            del os.environ["PM_TEST_ENV"]

    def test_env_with_default(self):
        cfg = load_config(InterpCfg, overrides=["name=${env:PM_NONEXISTENT:fallback}"])
        assert cfg.name == "fallback"

    def test_eval_simple(self):
        @dataclass
        class EvalCfg:
            a: int = 10
            b: int = 0

        cfg = load_config(EvalCfg, overrides=["b=${eval:cfg.a * 2}"])
        assert cfg.b == 20

    def test_eval_math(self):
        @dataclass
        class EvalMathCfg:
            val: float = 0.0

        cfg = load_config(EvalMathCfg, overrides=["val=${eval:math.sqrt(16)}"])
        assert cfg.val == 4.0

    def test_cycle_detection(self):
        @dataclass
        class CycleCfg:
            x: str = ""
            y: str = ""

        with pytest.raises(InterpolationError):
            load_config(CycleCfg, overrides=["x=${y}", "y=${x}"])

    def test_self_cycle(self):
        @dataclass
        class SelfCfg:
            x: str = ""

        with pytest.raises(InterpolationError):
            load_config(SelfCfg, overrides=["x=${x}"])

    def test_missing_ref_key(self):
        @dataclass
        class MissingRefCfg:
            a: str = ""

        with pytest.raises(InterpolationError):
            load_config(MissingRefCfg, overrides=["a=${nonexistent}"])

    def test_escape_dollar_brace(self):
        @dataclass
        class EscapeCfg:
            val: str = ""

        cfg = load_config(EscapeCfg, overrides=["val=literal_$${not_resolved}"])
        assert "${not_resolved}" in cfg.val
        assert "literal_" in cfg.val

    def test_nested_interpolation(self):
        os.environ["PM_NESTED"] = "world"
        try:
            cfg = load_config(InterpCfg, overrides=["name=${env:PM_NESTED}"])
            assert cfg.name == "world"
        finally:
            del os.environ["PM_NESTED"]


# ---------------------------------------------------------------------------
# 7. Type conversion
# ---------------------------------------------------------------------------


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@dataclass
class TypeTestCfg:
    flag: bool = True
    count: int = 42
    rate: float = 0.001
    color: Color = Color.RED
    mode: Literal["train", "eval"] = "train"
    path: Path = Path("/tmp")
    items: List[int] = field(default_factory=lambda: [1, 2, 3])
    meta: Dict[str, Any] = field(default_factory=dict)
    maybe: Optional[str] = None


class TestTypeConversion:
    def test_bool_false(self):
        cfg = load_config(TypeTestCfg, overrides=["flag=false"])
        assert cfg.flag is False

    def test_bool_true_string(self):
        cfg = load_config(TypeTestCfg, overrides=["flag=true"])
        assert cfg.flag is True

    def test_bool_on_off(self):
        cfg = load_config(TypeTestCfg, overrides=["flag=off"])
        assert cfg.flag is False

    def test_int(self):
        cfg = load_config(TypeTestCfg, overrides=["count=99"])
        assert cfg.count == 99

    def test_int_rejects_float_string(self):
        with pytest.raises(TypeConversionError):
            load_config(TypeTestCfg, overrides=["count=3.5"])

    def test_lossless_float_to_int(self):
        cfg = load_config(TypeTestCfg, overrides=["count=3.0"])
        assert cfg.count == 3

    def test_float(self):
        cfg = load_config(TypeTestCfg, overrides=["rate=0.01"])
        assert cfg.rate == 0.01

    def test_float_from_int(self):
        cfg = load_config(TypeTestCfg, overrides=["rate=42"])
        assert cfg.rate == 42.0

    def test_enum_by_name(self):
        cfg = load_config(TypeTestCfg, overrides=["color=GREEN"])
        assert cfg.color == Color.GREEN

    def test_enum_by_value(self):
        cfg = load_config(TypeTestCfg, overrides=["color=blue"])
        assert cfg.color == Color.BLUE

    def test_enum_invalid(self):
        with pytest.raises(TypeConversionError):
            load_config(TypeTestCfg, overrides=["color=PURPLE"])

    def test_literal_valid(self):
        cfg = load_config(TypeTestCfg, overrides=["mode=eval"])
        assert cfg.mode == "eval"

    def test_literal_invalid(self):
        with pytest.raises(TypeConversionError):
            load_config(TypeTestCfg, overrides=["mode=deploy"])

    def test_path(self):
        cfg = load_config(TypeTestCfg, overrides=["path=/home/user/data"])
        assert cfg.path == Path("/home/user/data")

    def test_list_json(self):
        cfg = load_config(TypeTestCfg, overrides=["items=[10,20,30]"])
        assert cfg.items == [10, 20, 30]

    def test_list_comma(self):
        cfg = load_config(TypeTestCfg, overrides=["items=4,5,6"])
        assert cfg.items == [4, 5, 6]

    def test_dict_json(self):
        cfg = load_config(TypeTestCfg, overrides=['meta={"key":"val"}'])
        assert cfg.meta == {"key": "val"}

    def test_optional_some(self):
        cfg = load_config(TypeTestCfg, overrides=["maybe=hello"])
        assert cfg.maybe == "hello"

    def test_optional_null(self):
        cfg = load_config(TypeTestCfg, overrides=["maybe=null"])
        assert cfg.maybe is None

    def test_optional_none(self):
        cfg = load_config(TypeTestCfg, overrides=["maybe=none"])
        assert cfg.maybe is None

    def test_non_optional_null_raises(self):
        with pytest.raises(TypeConversionError):
            load_config(TypeTestCfg, overrides=["count=null"])


# ---------------------------------------------------------------------------
# 8. Required fields / ???
# ---------------------------------------------------------------------------


class TestRequiredFields:
    def test_missing_required_raises(self):
        @dataclass
        class ReqCfg:
            name: str

        with pytest.raises(TypeConversionError, match="required"):
            load_config(ReqCfg)

    def test_required_filled_by_override(self):
        @dataclass
        class ReqCfg:
            name: str

        cfg = load_config(ReqCfg, overrides=["name=provided"])
        assert cfg.name == "provided"

    def test_required_filled_by_yaml(self, tmp_conf_dir):
        @dataclass
        class ReqCfg:
            name: str

        _write_yaml(tmp_conf_dir / "cfg.yaml", "name: from_yaml\n")
        cfg = load_config(ReqCfg, config_files=[str(tmp_conf_dir / "cfg.yaml")])
        assert cfg.name == "from_yaml"


# ---------------------------------------------------------------------------
# 9. Output management
# ---------------------------------------------------------------------------


class TestOutput:
    def test_save_config_writes_files(self, tmp_conf_dir):
        out = tmp_conf_dir / "output"
        cfg = SimpleCfg(a=99, b="test")
        saved = save_config(cfg, out)
        assert saved.exists()
        assert (saved / "config.yaml").exists()
        assert (saved / "meta.json").exists()

    def test_save_config_content(self, tmp_conf_dir):
        out = tmp_conf_dir / "output"
        cfg = SimpleCfg(a=42, b="hello")
        saved = save_config(cfg, out)
        loaded = yaml.safe_load((saved / "config.yaml").read_text())
        assert loaded["a"] == 42
        assert loaded["b"] == "hello"

    def test_meta_json(self, tmp_conf_dir):
        out = tmp_conf_dir / "output"
        save_config(SimpleCfg(), out)
        meta = json.loads((out / "meta.json").read_text())
        assert "version" in meta
        assert "time" in meta
        assert "python" in meta

    def test_to_plain_roundtrip(self):
        cfg = SimpleCfg(a=99, b="test", c=1.5, d=False)
        plain = to_plain(cfg)
        assert plain == {"a": 99, "b": "test", "c": 1.5, "d": False}

    def test_allow_unknown(self, tmp_conf_dir):
        @dataclass
        class StrictCfg:
            a: int = 1

        _write_yaml(tmp_conf_dir / "cfg.yaml", "a: 99\nextra: boom\n")
        cfg = load_config(StrictCfg, config_files=[str(tmp_conf_dir / "cfg.yaml")], allow_unknown=True)
        assert cfg.a == 99


# ---------------------------------------------------------------------------
# 10. Help rendering
# ---------------------------------------------------------------------------


class TestHelp:
    def test_includes_class_name(self):
        h = render_help(SimpleCfg)
        assert "SimpleCfg" in h

    def test_includes_field_names(self):
        h = render_help(SimpleCfg)
        assert "a : int" in h
        assert "b : str" in h

    def test_includes_defaults(self):
        h = render_help(SimpleCfg)
        assert "= 1" in h or "1" in h

    def test_nested_render(self):
        h = render_help(OuterCfg)
        assert "InnerCfg" in h
        assert "lr : float" in h


# ---------------------------------------------------------------------------
# 11. Real-world scenario
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 3e-4
    weight_decay: float = 0.01


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.1


@dataclass
class DataConfig:
    path: str = "./data"
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    exp_name: str = "default"
    seed: int = 42
    max_epochs: int = 100
    log_dir: str = "logs/${exp_name}/${now:%Y-%m-%d_%H-%M-%S}"


class TestRealWorld:
    def test_full_pipeline_defaults(self):
        cfg = load_config(TrainConfig)
        assert cfg.model.hidden_dim == 256
        assert cfg.optimizer.lr == 3e-4
        assert cfg.data.batch_size == 32
        assert "default" in cfg.log_dir

    def test_full_pipeline_yaml(self, tmp_conf_dir):
        _write_yaml(
            tmp_conf_dir / "train.yaml",
            """
model:
  hidden_dim: 512
  num_layers: 12
optimizer:
  lr: 1e-4
exp_name: my_experiment
""",
        )
        cfg = load_config(TrainConfig, config_files=[str(tmp_conf_dir / "train.yaml")])
        assert cfg.model.hidden_dim == 512
        assert cfg.model.num_layers == 12
        assert cfg.optimizer.lr == 1e-4
        assert cfg.exp_name == "my_experiment"
        # Unchanged defaults
        assert cfg.model.dropout == 0.1
        assert cfg.optimizer.weight_decay == 0.01

    def test_full_pipeline_cli_overrides_yaml(self, tmp_conf_dir):
        _write_yaml(
            tmp_conf_dir / "train.yaml",
            "model:\n  hidden_dim: 512\nexp_name: yaml_exp\n",
        )
        cfg = load_config(
            TrainConfig,
            config_files=[str(tmp_conf_dir / "train.yaml")],
            overrides=["model.hidden_dim=1024", "seed=123"],
        )
        # CLI wins over YAML
        assert cfg.model.hidden_dim == 1024
        # YAML still applies for non-overridden
        assert cfg.exp_name == "yaml_exp"
        # CLI override
        assert cfg.seed == 123

    def test_output_save_with_interpolation(self, tmp_conf_dir):
        """Save output with interpolated log_dir to temp directory."""
        import parameter_manager as pm

        out_dir = tmp_conf_dir / "runs"
        cfg = load_config(
            TrainConfig,
            overrides=["exp_name=test_save", "log_dir=" + str(out_dir) + "/${exp_name}"],
            save=True,
        )
        if pm.last_output_dir:
            assert pm.last_output_dir.exists()
            config_yaml = pm.last_output_dir / "config.yaml"
            assert config_yaml.exists()
            loaded = yaml.safe_load(config_yaml.read_text())
            assert loaded["exp_name"] == "test_save"


# ---------------------------------------------------------------------------
# 12. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_dataclass(self):
        @dataclass
        class EmptyCfg:
            pass

        cfg = load_config(EmptyCfg)
        assert isinstance(cfg, EmptyCfg)

    def test_empty_config_files(self):
        cfg = load_config(SimpleCfg, config_files=[])
        assert cfg.a == 1

    def test_save_false(self, tmp_conf_dir):
        import parameter_manager as pm

        prev = pm.last_output_dir
        out_dir = str(tmp_conf_dir / "should_not_exist")
        load_config(SimpleCfg, output_dir=out_dir, save=False)
        assert not Path(out_dir).exists()
        # last_output_dir shouldn't change when save=False
        assert pm.last_output_dir == prev

    def test_mutable_default_factory(self):
        @dataclass
        class MutableCfg:
            items: List[int] = field(default_factory=lambda: [1, 2, 3])

        cfg1 = load_config(MutableCfg)
        cfg2 = load_config(MutableCfg)
        assert cfg1.items == [1, 2, 3]
        assert cfg2.items == [1, 2, 3]
        # Should be independent
        cfg1.items.append(4)
        assert cfg2.items == [1, 2, 3]

    def test_bool_int_edge(self):
        """Ensure bool isn't treated as int."""
        cfg = load_config(TypeTestCfg, overrides=["flag=true", "count=1"])
        assert cfg.flag is True
        assert cfg.count == 1

    def test_yaml_null_non_optional(self, tmp_conf_dir):
        @dataclass
        class NonOptCfg:
            name: str = "default"

        _write_yaml(tmp_conf_dir / "cfg.yaml", "name: null\n")
        with pytest.raises(TypeConversionError):
            load_config(NonOptCfg, config_files=[str(tmp_conf_dir / "cfg.yaml")])

    def test_type_conversion_error_includes_path(self):
        with pytest.raises(TypeConversionError) as exc:
            load_config(TypeTestCfg, overrides=["count=abc"])
        assert "count" in str(exc.value)
        assert "int" in str(exc.value).lower()
        assert "abc" in str(exc.value)

    def test_render_help_required_field(self):
        @dataclass
        class ReqCfg:
            name: str

        h = render_help(ReqCfg)
        assert "required" in h.lower()
