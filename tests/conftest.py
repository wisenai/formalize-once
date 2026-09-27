"""Skip the MedCalc-Bench-dependent test modules when its data is not present.

MedCalc-Bench carries no redistribution license, so the public release does not
vendor it; `sscramble download` fetches it, after which these tests run.
"""
from symbol_scramble import paths

_MEDCALC_PRESENT = paths.CALC_IMPL.exists() and paths.MEDCALC_CSV.exists()

collect_ignore = [] if _MEDCALC_PRESENT else [
    "test_audit.py",
    "test_calc_runner.py",
    "test_perturb.py",
]
