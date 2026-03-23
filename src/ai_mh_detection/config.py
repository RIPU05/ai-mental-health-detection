from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class AppConfig(BaseModel):
    data: dict[str, Any] = {}

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        p = Path(path)
        payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls(data=payload)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return repo_root() / "configs" / "default.yaml"
