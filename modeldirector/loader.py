"""YAML / dict configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from modeldirector.config import Config


def load_config(source: str | Path | dict) -> Config:
    """Load a Config from a YAML file path or a plain dict.

    Environment variables in the form ``${VAR_NAME}`` are expanded before
    validation.
    """
    if isinstance(source, dict):
        raw: dict[str, Any] = source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = yaml.safe_load(path.read_text()) or {}

    raw = _expand_env(raw)
    try:
        return Config.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}") from e


def _expand_env(obj: Any) -> Any:
    """Recursively expand ${VAR} placeholders in strings."""
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            var = obj[2:-1]
            val = os.environ.get(var)
            if val is None:
                raise ValueError(f"Environment variable ${var} is not set")
            return val
        return obj
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj
