"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    return config

