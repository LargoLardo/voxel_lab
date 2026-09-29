"""YAML configuration helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def resolve_hidden_layers(width: int, layers: list[int] | tuple[int, ...] | None = None) -> tuple[int, ...]:
    """Use the legacy single width unless an explicit hidden-layer list is given."""
    layers = [width] if layers is None else layers
    if not isinstance(layers, (list, tuple)) or not layers or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in layers
    ):
        raise ValueError("hidden_layers must be a nonempty list of positive integers, e.g. [32, 32]")
    return tuple(layers)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject ambiguous roots."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
