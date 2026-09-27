"""M2: perturbation operators respect their invariance contracts."""
import random

import pytest

from symbol_scramble import calc_runner
from symbol_scramble.data_loader import load_seed_items
from symbol_scramble.perturb import note_edit
from symbol_scramble.perturb.note_rewrite import _entities_round_trip
from symbol_scramble.perturb.operators import ALL_OPERATORS


@pytest.fixture(scope="module")
def seeds():
    return load_seed_items()


def _seed_for_calc(seeds, cid):
    for s in seeds:
        if s.calc_id == cid:
            return s
    return None


def _recompute_matches(v):
    ans = calc_runner.answer(v.calc_id, v.entities)
    if v.output_type == "integer":
        return int(round(float(ans))) == int(round(float(v.gt_answer)))
    return abs(float(ans) - float(v.gt_answer)) <= 1e-6 + 0.01 * abs(float(v.gt_answer))


def test_gt_always_recomputes(seeds):
    """Every generated variant's stored GT must equal the oracle on its entities."""
    rng = random.Random(0)
    checked = 0
    for s in seeds[:120]:
        for level, op in ALL_OPERATORS.items():
            if s.family not in op.applies_to or op.needs_llm:
                continue
            for v in op.generate(s, random.Random(s.row_id), 3):
                assert _recompute_matches(v), (level, v.variant_id)
                checked += 1
    assert checked > 50


def test_invariant_ops_preserve_answer(seeds):
    """L0/Lu/L3/L4 must leave the recomputed answer equal to the seed answer."""
    for level in ("L0", "Lu", "L3", "L4"):
        op = ALL_OPERATORS[level]
        produced = 0
        for s in seeds:
            if s.family not in op.applies_to:
                continue
            for v in op.generate(s, random.Random(s.row_id), 2):
                assert v.answer_invariant
                if s.output_type == "integer":
                    assert int(round(float(v.gt_answer))) == int(round(float(s.gt_answer)))
                else:
                    assert abs(float(v.gt_answer) - float(s.gt_answer)) <= 0.01 * abs(
                        float(s.gt_answer)
                    ) + 1e-6
                produced += 1
            if produced > 20:
                break
        assert produced > 0, f"{level} produced no variants"


def test_value_shift_changes_answer(seeds):
    op = ALL_OPERATORS["L2"]
    changed = 0
    for s in seeds:
        for v in op.generate(s, random.Random(s.row_id), 3):
            assert v.op_params["entity"] in s.entities
            # note reflects the new value, not the old
            assert note_edit.note_contains_number(v.note, float(v.op_params["new"]), tol=0.5)
            changed += 1
        if changed > 15:
            break
    assert changed > 0


def test_flip_changes_score(seeds):
    op = ALL_OPERATORS["Flip"]
    flipped = 0
    for s in seeds:
        if s.family != "score":
            continue
        for v in op.generate(s, random.Random(s.row_id), 2):
            assert v.entities[v.op_params["entity"]] == v.op_params["to"]
            assert int(round(float(v.gt_answer))) != int(round(float(s.gt_answer)))
            flipped += 1
        if flipped > 8:
            break
    assert flipped > 0


def test_unit_conversion_roundtrips():
    # 70 kg -> lbs and back
    nv, nu = note_edit.convert_unit("weight", 70.0, "kg")
    assert nu == "lbs" and abs(nv - 154.32) < 0.1
    nv2, nu2 = note_edit.convert_unit("Temperature", 37.0, "degrees celsius")
    assert nu2 == "degrees fahrenheit" and abs(nv2 - 98.6) < 0.1


def test_entities_round_trip_gate():
    ents = {"sex": "Male", "age": [80, "years"], "weight": [70.0, "kg"]}
    good = "An 80-year-old man weighing 70 kg was admitted."
    bad = "A 55-year-old man weighing 70 kg was admitted."
    assert _entities_round_trip(good, ents)
    assert not _entities_round_trip(bad, ents)
