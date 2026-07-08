from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return cfg
