from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import yaml

from vn_receipt_ocr.config.models import TrainConfig


def load_yaml(path: Path | str) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into a copy of base. Lists are replaced, not concatenated."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _apply_dotted(target: dict, key: str, value: Any) -> None:
    """Mutate target so that target['a']['b']=value for key='a.b'."""
    parts = key.split(".")
    node = target
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def load_train_config(
    experiment_path: Path | str,
    configs_root: Path | str,
    overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    """
    Load and compose a TrainConfig from an experiment YAML.

    The experiment YAML may contain an 'include' list of relative paths under
    configs_root; each included YAML is loaded and merged in order, then the
    experiment YAML's own keys are applied as the highest-precedence file-level
    layer. CLI/API `overrides` (dotted keys) are applied last.
    """
    experiment_path = Path(experiment_path)
    configs_root = Path(configs_root)

    raw = load_yaml(experiment_path)
    includes = raw.pop("include", []) or []

    merged: dict = {}
    for rel in includes:
        included = load_yaml(configs_root / rel)
        merged = deep_merge(merged, included)
    merged = deep_merge(merged, raw)

    if overrides:
        for k, v in overrides.items():
            _apply_dotted(merged, k, v)

    return TrainConfig.model_validate(merged)
