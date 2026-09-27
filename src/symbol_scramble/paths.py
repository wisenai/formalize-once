"""Absolute path resolution for the project. No hard-coded paths elsewhere."""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = two levels up from this file (src/symbol_scramble/paths.py).
ROOT = Path(__file__).resolve().parents[2]


def _env_or(default: Path, var: str) -> Path:
    v = os.environ.get(var)
    return Path(v).resolve() if v else default


DATA = _env_or(ROOT / "data", "SSCRAMBLE_DATA")
RAW = DATA / "raw"
CALC_IMPL = RAW / "calculator_implementations"
MEDCALC_CSV = RAW / "test_data.csv"
MEDQA_DIR = RAW / "medqa"
VARIANTS = DATA / "variants"
RESULTS = DATA / "results"
FIGURES = DATA / "figures"
CACHE = RESULTS / "cache"
CONFIGS = ROOT / "configs"


def ensure_dirs() -> None:
    for p in (DATA, RAW, VARIANTS, RESULTS, FIGURES, CACHE):
        p.mkdir(parents=True, exist_ok=True)
