"""Deterministic note-text surgery + unit conversion for the perturbation engine.

Editing the note by exact find/replace (rather than always regenerating with an
LLM) keeps L2/Lu cheap and makes the round-trip QC trivial: if the old value
string was found and replaced, the note provably reflects the new value.
"""
from __future__ import annotations

import re
from typing import Optional

# ---- number formatting -------------------------------------------------------


def num_variants(v: float) -> list[str]:
    """Textual forms a numeric value might take in a clinical note."""
    out: list[str] = []
    if float(v).is_integer():
        iv = int(round(v))
        out += [str(iv), f"{iv}.0", f"{iv:,}"]
    else:
        out.append(f"{v:g}")
        out.append(f"{v:.1f}")
        out.append(f"{v:.2f}")
    # de-dup, keep order
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def fmt_num(v: float, round_to: int = 2) -> str:
    if float(v).is_integer():
        return str(int(round(v)))
    return f"{round(v, round_to):g}"


def find_number_in_note(note: str, value: float) -> Optional[str]:
    """Return the exact substring the value appears as in the note, or None."""
    for cand in num_variants(value):
        pat = r"(?<![\d.])" + re.escape(cand) + r"(?![\d])"
        if re.search(pat, note):
            return cand
    return None


def replace_number(note: str, old_value: float, new_text: str) -> Optional[str]:
    """Replace the first textual occurrence of ``old_value`` with ``new_text``."""
    cand = find_number_in_note(note, old_value)
    if cand is None:
        return None
    pat = r"(?<![\d.])" + re.escape(cand) + r"(?![\d])"
    return re.sub(pat, new_text, note, count=1)


def note_contains_number(note: str, value: float, tol: float = 0.0) -> bool:
    if find_number_in_note(note, value) is not None:
        return True
    if tol:  # allow small formatting tolerance for round-tripped values
        for m in re.findall(r"\d+\.?\d*", note):
            try:
                if abs(float(m) - value) <= tol:
                    return True
            except ValueError:
                pass
    return False


# ---- unit conversion (Lu) ----------------------------------------------------
# Each entry: (from_unit, to_unit, factor) meaning to_value = from_value * factor.
# Only conversions the shipped calculators' helpers understand are kept (the Lu
# operator additionally verifies the oracle answer is unchanged before emitting).

_CONVERSIONS = {
    "weight": [("kg", "lbs", 2.2046226218), ("lbs", "kg", 0.45359237)],
    "height": [("cm", "in", 0.3937007874), ("in", "cm", 2.54),
               ("cm", "m", 0.01), ("m", "cm", 100.0)],
    "creatinine": [("mg/dL", "µmol/L", 88.42), ("µmol/L", "mg/dL", 1 / 88.42)],
    "Pre-operative creatinine": [("mg/dL", "µmol/L", 88.42), ("µmol/L", "mg/dL", 1 / 88.42)],
    "Calcium": [("mg/dL", "mmol/L", 0.2495), ("mmol/L", "mg/dL", 1 / 0.2495)],
    "Glucose": [("mg/dL", "mmol/L", 0.0555), ("mmol/L", "mg/dL", 1 / 0.0555)],
    "Blood Urea Nitrogen (BUN)": [("mg/dL", "mmol/L", 0.357), ("mmol/L", "mg/dL", 1 / 0.357)],
    "Temperature": [("degrees celsius", "degrees fahrenheit", None),
                    ("degrees fahrenheit", "degrees celsius", None)],
}


def convert_unit(entity_key: str, value: float, unit: str):
    """Return (new_value, new_unit) for an alternate representation, or None."""
    rules = _CONVERSIONS.get(entity_key)
    if not rules:
        return None
    unit_l = unit.lower().strip()
    for frm, to, factor in rules:
        if frm.lower() == unit_l:
            if factor is None:  # temperature
                if "celsius" in frm:
                    nv = value * 9 / 5 + 32
                else:
                    nv = (value - 32) * 5 / 9
            else:
                nv = value * factor
            return round(nv, 3), to
    return None
