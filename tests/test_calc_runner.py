"""M1 gate: the oracle reproduces ground truth on all in-scope seed items."""
import pytest

from symbol_scramble import calc_runner
from symbol_scramble.data_loader import load_seed_items
from symbol_scramble.grading import is_correct
from symbol_scramble.healthcheck import _as_variant, run_healthcheck


def test_seed_items_loaded():
    items = load_seed_items()
    assert len(items) > 900
    assert all(i.family in ("equation", "score") for i in items)


def test_oracle_reproduces_all_ground_truth():
    report = run_healthcheck()
    assert report.total >= 900
    assert report.rate == 1.0, f"oracle failed on {report.per_calc_fail}"


def test_build_params_uses_arg_mapping():
    # CHA2DS2-VASc maps "Diabetes history" -> "diabetes"
    params = calc_runner.build_params("4", {"Diabetes history": True, "sex": "Female"})
    assert params["diabetes"] is True
    assert params["sex"] == "Female"


def test_recompute_changes_with_entities():
    items = [i for i in load_seed_items(families=("equation",)) if i.calc_id == "10"]
    item = items[0]  # Ideal Body Weight = f(sex, height)
    base = calc_runner.answer("10", item.entities)
    taller = dict(item.entities)
    taller["height"] = [item.entities["height"][0] + 20, item.entities["height"][1]]
    assert calc_runner.answer("10", taller) != base


def test_unknown_calculator_raises():
    with pytest.raises(calc_runner.CalcError):
        calc_runner.run_calculator("99999", {})
