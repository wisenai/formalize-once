"""P0 audit correctness gate: patches applied, quarantines set, formulas verified."""
from symbol_scramble import audit, calc_runner


def test_ckd_epi_patched_male_coefficient():
    audit.apply_patches()
    # male, creatinine <= 0.9 must use A=0.9 (CKD-EPI 2021), giving ~101 not ~94
    got = float(calc_runner.answer("3", {"sex": "Male", "age": [60, "years"],
                                         "creatinine": [0.8, "mg/dL"]}))
    assert abs(got - 101.3) < 2.0, got


def test_independent_formula_spot_checks_pass():
    audit.apply_patches()
    verified, failed = audit.verify_formulas()
    assert not failed, failed
    assert len(verified) >= 5


def test_quarantined_calculators_excluded():
    from symbol_scramble.data_loader import load_seed_items

    audit.quarantine_in_family_map()
    calc_ids = {s.calc_id for s in load_seed_items()}
    assert "23" not in calc_ids  # MELD-Na
    assert "45" not in calc_ids  # CURB-65
