"""Common helpers such as seed control, config loading, and file IO."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import yaml
except Exception:
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike[str] = "config.yaml") -> dict[str, Any]:
    """Load the unified YAML configuration file."""

    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        text = f.read()
    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        data = _simple_yaml_load(text)
    return data


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [_parse_yaml_scalar(part.strip()) for part in body.split(",")]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by config.yaml when PyYAML is absent."""

    tokens: list[tuple[int, str]] = []
    for raw in text.splitlines():
        line = _strip_yaml_comment(raw).rstrip()
        if not line.strip():
            continue
        tokens.append((len(line) - len(line.lstrip(" ")), line.strip()))

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for idx, (indent, content) in enumerate(tokens):
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("Invalid YAML list indentation in config.yaml.")
            item = content[2:].strip()
            if ":" in item and not item.startswith(("'", '"')):
                key, value = item.split(":", 1)
                node: dict[str, Any] = {}
                parent.append(node)
                if value.strip():
                    node[key.strip()] = _parse_yaml_scalar(value)
                    stack.append((indent, node))
                else:
                    has_child = idx + 1 < len(tokens) and tokens[idx + 1][0] > indent
                    if has_child:
                        next_is_list = tokens[idx + 1][1].startswith("- ")
                        child: Any = [] if next_is_list else {}
                        node[key.strip()] = child
                        stack.append((indent, node))
                        stack.append((indent, child))
                    else:
                        node[key.strip()] = None
                        stack.append((indent, node))
            else:
                parent.append(_parse_yaml_scalar(item))
            continue

        if ":" not in content:
            raise ValueError(f"Invalid YAML line: {content}")
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not isinstance(parent, dict):
            raise ValueError("Invalid YAML mapping indentation in config.yaml.")
        if value:
            parent[key] = _parse_yaml_scalar(value)
        else:
            has_child = idx + 1 < len(tokens) and tokens[idx + 1][0] > indent
            if has_child:
                next_is_list = tokens[idx + 1][1].startswith("- ")
                child = [] if next_is_list else {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = None

    return root


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Return a nested config value using dot notation."""

    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch when available."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch is optional for non-neural utilities.
        pass


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create a directory if it does not exist and return it as a Path."""

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def project_path(*parts: str | os.PathLike[str]) -> Path:
    """Build an absolute path inside the project root."""

    return PROJECT_ROOT.joinpath(*map(str, parts))


def resolve_output_dir(config: Mapping[str, Any], *parts: str) -> Path:
    """Resolve and create an output subdirectory from config."""

    base = Path(get_nested(config, "project.output_dir", "outputs"))
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    return ensure_dir(base.joinpath(*parts))


def resolve_device(device: str = "auto") -> str:
    """Resolve `auto` to cuda, mps, or cpu."""

    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def save_json(data: Any, path: str | os.PathLike[str]) -> Path:
    """Save JSON with stable formatting."""

    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out


def load_json(path: str | os.PathLike[str]) -> Any:
    """Load a JSON file."""

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(df: Any, path: str | os.PathLike[str]) -> Path:
    """Save a pandas DataFrame to CSV, creating the parent directory."""

    out = Path(path)
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    return out


def flatten_dict(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary for tabular experiment logs."""

    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_dict(value, name))
        else:
            flat[name] = value
    return flat
