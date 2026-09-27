"""Calculator runner — the ground-truth oracle (IMPLEMENTATION_SPEC §6).

Executes the *vendored* MedCalc-Bench calculator implementations on a structured
entity dict to recompute the answer for any (possibly perturbed) variant.

The shipped calculators are trusted, deterministic, pure functions that do
file-relative imports; we put their directory on ``sys.path`` and import them
in-process for speed. Untrusted, model-generated code (the neuro-symbolic branch)
is NEVER run here — it goes through ``sandbox.py`` in a subprocess.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from functools import lru_cache
from typing import Any

from . import paths

# Metadata keys inside name_to_python.json that are NOT entity->arg mappings.
_META_KEYS = {"file path", "explanation function", "calculator name", "type", "question"}


class CalcError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _name_to_python() -> dict[str, dict[str, Any]]:
    p = paths.CALC_IMPL / "name_to_python.json"
    if not p.exists():
        raise CalcError(f"missing {p}; run `sscramble download` first")
    with open(p) as f:
        return json.load(f)


def _ensure_on_path() -> None:
    ci = str(paths.CALC_IMPL)
    if ci not in sys.path:
        sys.path.insert(0, ci)


@lru_cache(maxsize=256)
def _load_fn(calc_id: str):
    n2p = _name_to_python()
    if calc_id not in n2p:
        raise CalcError(f"unknown calculator id {calc_id!r}")
    meta = n2p[calc_id]
    _ensure_on_path()
    module_name = os.path.basename(meta["file path"]).replace(".py", "")
    mod = importlib.import_module(module_name)
    fn = getattr(mod, meta["explanation function"])
    return fn, meta


def arg_mapping(calc_id: str) -> dict[str, str]:
    """entity-key -> function-arg-name mapping for a calculator."""
    _, meta = _load_fn(calc_id)
    return {k: v for k, v in meta.items() if k not in _META_KEYS}


def entity_arg_names(calc_id: str) -> set[str]:
    """The set of function arg-names this calculator actually consumes."""
    return set(arg_mapping(calc_id).values())


def build_params(calc_id: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Translate a Relevant-Entities dict into the calculator's ``params`` dict."""
    mapping = arg_mapping(calc_id)
    params: dict[str, Any] = {}
    for k, v in entities.items():
        params[mapping.get(k, k)] = v
    return params


def run_calculator(calc_id: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Return {"answer", "explanation"} by executing the shipped calculator."""
    fn, _ = _load_fn(str(calc_id))
    params = build_params(str(calc_id), entities)
    try:
        out = fn(params)
    except Exception as e:  # noqa: BLE001 - surface calculator failure to caller
        raise CalcError(f"calc {calc_id} failed on {entities!r}: {e}") from e
    if not isinstance(out, dict) or "Answer" not in out:
        raise CalcError(f"calc {calc_id} returned malformed output: {out!r}")
    return {"answer": out["Answer"], "explanation": out.get("Explanation", "")}


def answer(calc_id: str, entities: dict[str, Any]) -> Any:
    return run_calculator(calc_id, entities)["answer"]
