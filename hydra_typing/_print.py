"""Color config printing for hydra_typing.

Pretty-prints a resolved dataclass config with ANSI terminal colors,
highlighting overridden values.
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# ANSI escape codes
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def print_config(
    cfg: Any,
    overrides: Optional[List[str]] = None,
    use_color: bool = True,
) -> str:
    """Pretty-print a resolved dataclass config, highlighting overridden values.

    Args:
        cfg: A resolved dataclass config instance.
        overrides: Hydra-style override strings (``["model=large", "lr=0.001"]``).
                   Paths referenced are highlighted in yellow.
        use_color: If ``True`` (default), use ANSI terminal colors.

    Colors:
        - **Green**: default values (unchanged).
        - **Yellow**: values that were explicitly overridden.
        - **Cyan**: section headers.
        - **Dim**: type annotations and separators.

    Returns:
        The formatted string (also printed to stdout).
    """
    override_paths: set = set()
    for item in (overrides or []):
        override_paths.update(_override_paths(item))

    lines: List[str] = []
    _print_inner(cfg, "", override_paths, lines, use_color)
    result = "\n".join(lines)

    if overrides:
        summary = [
            "",
            f"{_BOLD}{_YELLOW}Overrides applied:{_RESET}" if use_color else "Overrides applied:",
        ]
        for item in overrides:
            prefix = f"  {_YELLOW}{item}{_RESET}" if use_color else f"  {item}"
            summary.append(prefix)
        result += "\n" + "\n".join(summary)

    print(result)
    return result


def _override_paths(override: str) -> List[str]:
    """Extract dotted paths from a hydra override string."""
    for prefix in ("++", "+", "~"):
        if override.startswith(prefix):
            override = override[len(prefix):]
            break
    if "=" in override:
        key = override.split("=")[0]
    else:
        key = override
    return [key]


def _print_inner(
    obj: Any, path: str, override_paths: set, lines: List[str], use_color: bool,
    indent: int = 0,
) -> None:
    """Recursive helper for ``print_config``."""
    prefix_spacer = "  " * indent

    if dataclasses.is_dataclass(obj):
        name = type(obj).__name__
        hdr = f"{prefix_spacer}{_BOLD}{_CYAN}[{name}]{_RESET}" if use_color else f"{prefix_spacer}[{name}]"
        lines.append(hdr)
        for f in dataclasses.fields(obj):
            child_path = f"{path}.{f.name}" if path else f.name
            _print_inner(getattr(obj, f.name), child_path, override_paths, lines, use_color, indent + 1)
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            child_path = f"{path}.{k}" if path else k
            _print_inner(v, child_path, override_paths, lines, use_color, indent)
        return

    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _print_inner(item, f"{path}[{i}]", override_paths, lines, use_color, indent)
        return

    # Leaf value
    field_name = path.split(".")[-1] if "." in path else path
    is_overridden = path in override_paths or _prefix_match(path, override_paths)
    color = _YELLOW if (is_overridden and use_color) else (_GREEN if use_color else "")
    val_repr = _format_value(obj)

    if use_color:
        line = f"{prefix_spacer}{color}{field_name}{_RESET}{_DIM}: {type(obj).__name__} = {_RESET}{color}{val_repr}{_RESET}"
    else:
        line = f"{prefix_spacer}{field_name}: {type(obj).__name__} = {val_repr}"
    if is_overridden and use_color:
        line += f"  {_YELLOW}{_BOLD}# <-- overridden{_RESET}"
    lines.append(line)


def _prefix_match(path: str, paths: set) -> bool:
    parts = path.split(".")
    for i in range(len(parts)):
        if ".".join(parts[: i + 1]) in paths:
            return True
    return False


def _format_value(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        return f"{val:g}"
    if isinstance(val, str) and len(val) > 60:
        return f'"{val[:57]}..."'
    if isinstance(val, enum.Enum):
        return val.name
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, (list, tuple)):
        items = [_format_value(v) for v in val]
        if len(items) <= 5:
            return "[" + ", ".join(items) + "]"
        return "[" + ", ".join(items[:3]) + f", ... ({len(items)} items)]"
    return repr(val)


def to_plain(cfg: Any) -> dict:
    """Convert a dataclass config tree to a plain dict (for YAML dump)."""
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
