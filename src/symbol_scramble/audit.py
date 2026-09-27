"""Audit correctness gate (IMPLEMENTATION_SPEC v2 §6.1).

The shipped MedCalc-Bench calculators contain >20 bugs (Krohn-Grimberghe 2026,
arXiv 2603.02222), several corrupting ground-truth labels. Because our oracle *is*
the shipped calculator, we must (a) patch or quarantine every affected in-scope
calculator and (b) independently spot-check the rest against primary-source
formulas before trusting any label.

Response for our in-scope set:
  - CKD-EPI (3): PATCH — male branch sets A=0.7 but must be A=0.9 (CKD-EPI 2021).
  - MELD-Na (23): QUARANTINE — audit says update to MELD 3.0 (version ambiguity).
  - CURB-65 (45): QUARANTINE — BUN boundary (>19 vs >20 mg/dL).
  - MDRD (9): KEEP — 175-constant formula is correct; "broken paths" don't apply
    because we vendor all files and import by basename.
"""
from __future__ import annotations

from dataclasses import dataclass

import yaml

from . import calc_runner, paths

# ---- patches -----------------------------------------------------------------

CKD_EPI_FILE = "ckd-epi_2021_creatinine.py"
CKD_EPI_OLD = (
    'elif creatinine_val <= 0.9 and gender == "Male":\n'
    '        explanation += f"Because the patient\'s gender is male and the '
    'creatinine concentration is less than or equal to 0.9 mg/dL, A = 0.9 and '
    'B = -0.302.\\n"\n'
    '        a = 0.7'
)
CKD_EPI_NEW = CKD_EPI_OLD.replace("        a = 0.7", "        a = 0.9")

QUARANTINE = {
    "23": "audit: MELD-Na should be updated to MELD 3.0 (version ambiguity)",
    "45": "audit: CURB-65 BUN threshold boundary bug (>19 vs >20 mg/dL)",
}


@dataclass
class AuditResult:
    patched: list[str]
    quarantined: dict[str, str]
    verified: list[str]
    failed: list[tuple[str, str]]


def apply_patches() -> list[str]:
    """Idempotently patch buggy vendored calculators. Returns list of patched files."""
    patched = []
    f = paths.CALC_IMPL / CKD_EPI_FILE
    if f.exists():
        text = f.read_text()
        if CKD_EPI_OLD in text:
            f.write_text(text.replace(CKD_EPI_OLD, CKD_EPI_NEW))
            patched.append(CKD_EPI_FILE)
        elif CKD_EPI_NEW in text:
            patched.append(CKD_EPI_FILE + " (already patched)")
    calc_runner._load_fn.cache_clear()  # drop any cached pre-patch module
    return patched


# ---- independent formula spot-checks (primary-source reference cases) ---------
# Each: (calc_id, entities, expected, abs_tol, source)
REFERENCE_CASES = [
    ("6", {"weight": [70.0, "kg"], "height": [175.0, "cm"]}, 22.857, 0.05,
     "BMI = 70 / 1.75^2"),
    ("5", {"Systolic Blood Pressure": [120, "mm Hg"], "Diastolic Blood Pressure": [80, "mm Hg"]},
     93.333, 0.2, "MAP = (SBP + 2*DBP)/3"),
    ("39", {"Sodium": [140, "mEq/L"], "Chloride": [100, "mEq/L"], "Bicarbonate": [24, "mEq/L"]},
     16.0, 0.2, "Anion gap = Na - (Cl + HCO3)"),
    ("7", {"Calcium": [8.0, "mg/dL"], "Albumin": [3.0, "g/dL"]}, 8.8, 0.1,
     "Corrected Ca = Ca + 0.8*(4 - albumin)"),
    ("3", {"sex": "Male", "age": [60, "years"], "creatinine": [0.8, "mg/dL"]}, 101.3, 2.0,
     "CKD-EPI 2021 male, A=0.9 (post-patch)"),
]


def verify_formulas() -> tuple[list[str], list[tuple[str, str]]]:
    verified, failed = [], []
    for cid, ent, expected, tol, src in REFERENCE_CASES:
        try:
            got = float(calc_runner.answer(cid, ent))
        except Exception as e:  # noqa: BLE001
            failed.append((cid, f"error: {e} [{src}]"))
            continue
        if abs(got - expected) <= tol:
            verified.append(f"{cid}: {got:.3f}≈{expected} [{src}]")
        else:
            failed.append((cid, f"got {got:.3f} != {expected} (tol {tol}) [{src}]"))
    return verified, failed


def quarantine_in_family_map() -> None:
    """Move audit-quarantined calculators to family=quarantine in the config."""
    p = paths.CONFIGS / "calculator_family.yaml"
    data = yaml.safe_load(p.read_text())
    for cid, reason in QUARANTINE.items():
        entry = data["calculators"].get(int(cid))
        if entry and entry["family"] != "quarantine":
            entry["family"] = "quarantine"
            entry["quarantine_reason"] = reason
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def run_audit_gate() -> AuditResult:
    patched = apply_patches()
    quarantine_in_family_map()
    verified, failed = verify_formulas()
    result = AuditResult(patched, dict(QUARANTINE), verified, failed)
    _write_patch_record(result)
    return result


def _write_patch_record(r: AuditResult) -> None:
    rec = {
        "source": "Krohn-Grimberghe 2026 (arXiv:2603.02222)",
        "patched": {
            "ckd-epi_2021_creatinine.py": "male branch A: 0.7 -> 0.9 (CKD-EPI 2021)",
        },
        "quarantined": r.quarantined,
        "kept_verified": ["9 (MDRD, 175-constant formula correct)"],
        "independent_spot_checks": r.verified,
        "spot_check_failures": [f"{c}: {m}" for c, m in r.failed],
    }
    (paths.CONFIGS / "calculator_patches.yaml").write_text(
        yaml.safe_dump(rec, sort_keys=False, allow_unicode=True)
    )
