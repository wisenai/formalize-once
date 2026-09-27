"""RuleArena deterministic oracle (airline domain first).

Wraps the vendored RuleArena ``compute_answer.py`` so any (perturbed) structured
``info`` dict auto-labels to a ground-truth answer — the same pattern as the
MedCalc calculator oracle. The vendored module loads fee-table CSVs by relative
path, so we import it with its own directory as CWD.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
from functools import lru_cache
from typing import Any

from .. import paths

AIRLINE_DIR = paths.RAW / "rulearena" / "airline"
TAX_DIR = paths.RAW / "rulearena" / "tax"


@contextlib.contextmanager
def _chdir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@lru_cache(maxsize=1)
def _airline_module():
    spec = importlib.util.spec_from_file_location(
        "ra_airline_compute", AIRLINE_DIR / "compute_answer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    with _chdir(AIRLINE_DIR):  # fee_tables/* are loaded at import time by relative path
        spec.loader.exec_module(mod)
    return mod


def airline_answer(info: dict[str, Any]) -> int:
    """Total baggage+ticket cost for an airline ``info`` dict (ground truth)."""
    mod = _airline_module()
    with _chdir(AIRLINE_DIR):
        total, _ = mod.compute_answer(
            base_price=info["base_price"],
            direction=info["direction"],
            routine=info["routine"],
            customer_class=info["customer_class"],
            bag_list=info["bag_list"],
            check_base_tables=mod.check_base_tables,
        )
    return int(total)


@lru_cache(maxsize=1)
def _tax_modules():
    """Import the vendored tax oracle. It does `from prompt import ...` and
    `from structured_forms import ...`, so the tax dir must be importable."""
    import sys

    d = str(TAX_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)
    with _chdir(TAX_DIR):
        import micro_evaluation as me  # noqa: PLC0415
        import structured_forms as sf  # noqa: PLC0415
    return me, sf


# Input fields present in RuleArena's tax records that the reference computation
# never reads (verified by differential test: zeroing them changes no comp_0 answer).
# Most are gross-income duplicates of a ``taxable_*`` field the oracle does use;
# ``other_credits_or_payments`` is a payment the reference silently omits. We strip
# these from the model's view so the model and the oracle operate on an identical
# spec, then re-supply defaults here so the shipped TaxPayer model still constructs.
TAX_IGNORED_DEFAULTS = {
    "all_pensions": 0.0, "ira_distributions": 0.0, "nontaxable_combat_pay": 0.0,
    "other_credits_or_payments": 0, "social_security_benefits": 0.0,
    "spouse_blind": False, "tax_exempt_interest": 0.0,
}
TAX_IGNORED_FIELDS = frozenset(TAX_IGNORED_DEFAULTS)


def tax_answer(payer: dict[str, Any]) -> float:
    """Form 1040 amount owed (positive) or overpaid/refund (negative) for a tax
    record's ``pydantic`` payer dict (ground truth). Accepts a payer with the
    oracle-ignored fields stripped (they are re-filled with defaults, which does not
    change the answer). RuleArena grades with a tolerance, so this returns a float."""
    me, sf = _tax_modules()
    full = {**TAX_IGNORED_DEFAULTS, **payer}
    with _chdir(TAX_DIR):
        amount, _ = me.compute_answer(sf.TaxPayer(**full))
    return float(amount)


def answer(domain: str, info: dict[str, Any]):
    if domain == "airline":
        return airline_answer(info)
    if domain == "tax":
        return tax_answer(info)
    raise NotImplementedError(f"RuleArena domain {domain!r} oracle not wired yet")


@lru_cache(maxsize=4)
def reference_rules(domain: str = "airline") -> str:
    if domain == "tax":
        # tax rules aren't a .txt file — they live in prompt.py. comp_0 problems
        # are pure basic Form 1040 (all complexity flags off), so basic_forms is
        # the exact ruleset the model must formalize.
        import sys

        d = str(TAX_DIR)
        if d not in sys.path:
            sys.path.insert(0, d)
        with _chdir(TAX_DIR):
            import prompt as tax_prompt  # noqa: PLC0415
        return tax_prompt.basic_forms
    p = paths.RAW / "rulearena" / domain / "reference_rules.txt"
    return p.read_text()
