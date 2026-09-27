"""Config loading (IMPLEMENTATION_SPEC §12)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from . import paths


@lru_cache(maxsize=8)
def load_config(name: str = "default") -> dict[str, Any]:
    with open(paths.CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_models() -> dict[str, dict[str, Any]]:
    with open(paths.CONFIGS / "models.yaml") as f:
        return yaml.safe_load(f)["models"]


@lru_cache(maxsize=1)
def load_entity_ranges() -> dict[str, dict[str, Any]]:
    with open(paths.CONFIGS / "entity_ranges.yaml") as f:
        return yaml.safe_load(f)["ranges"]


@lru_cache(maxsize=1)
def load_distractors() -> dict[str, list[dict[str, Any]]]:
    with open(paths.CONFIGS / "distractor_bank.yaml") as f:
        return yaml.safe_load(f)


def model_spec(model_id: str) -> dict[str, Any]:
    models = load_models()
    if model_id not in models:
        raise KeyError(f"unknown model id {model_id!r}; known: {list(models)}")
    return models[model_id]
