"""M3: answer parsing, tolerance grading, fail-mode attribution."""
from symbol_scramble.branches.base import consensus, extract_code
from symbol_scramble.grading import fail_mode, is_correct, parse_answer
from symbol_scramble.schemas import GradedRecord, Variant


def _variant(gt, ot, lo=None, hi=None):
    return Variant(
        variant_id="1:L0:0", seed_row_id=1, calc_id="2", calc_name="x",
        family="equation" if ot == "decimal" else "score", output_type=ot,
        level="L0", note="", question="", entities={}, gt_answer=gt,
        lower=lo, upper=hi, answer_invariant=True,
    )


def test_parse_boxed_and_final():
    assert parse_answer("blah FINAL ANSWER: 25.2", "decimal") == 25.2
    assert parse_answer(r"so \boxed{42}", "integer") == 42
    assert parse_answer("the answer is 3,500", "integer") == 3500
    assert parse_answer("The CrCl = 97.2 mL/min", "decimal") == 97.2
    assert parse_answer("no number here", "decimal") is None


def test_integer_exact_match():
    v = _variant(2, "integer")
    assert is_correct(2, v)
    assert not is_correct(3, v)


def test_decimal_tolerance_band():
    v = _variant(25.24, "decimal", 23.98, 26.50)
    assert is_correct(24.0, v)
    assert is_correct(26.4, v)
    assert not is_correct(27.0, v)


def test_extract_code_block():
    text = "Here:\n```python\ndef solve():\n    return 5\n```\ndone"
    assert extract_code(text) == "def solve():\n    return 5"


def test_consensus_mode_median():
    assert consensus([2, 2, 3, 2], "integer") == 2
    assert consensus([10.0, 12.0, 11.0], "decimal") == 11.0
    assert consensus([None, None], "decimal") is None


def test_fail_mode_attribution():
    v = _variant(10.0, "decimal", 9.5, 10.5)
    # correct -> none
    ok = GradedRecord(variant_id="1:L0:0", branch="neurosymbolic", model_id="m",
                      raw_response="", parsed_answer=10.0, exec_ok=True,
                      correct=True, ref_answer=10.0)
    assert fail_mode(ok, v) == "none"
    # code errored -> execution
    ex = GradedRecord(variant_id="1:L0:0", branch="neurosymbolic", model_id="m",
                      raw_response="", parsed_answer=None, exec_ok=False,
                      correct=False, ref_answer=10.0)
    assert fail_mode(ex, v) == "execution"
    # ran, matches oracle, but graded wrong -> arithmetic
    ar = GradedRecord(variant_id="1:L0:0", branch="neurosymbolic", model_id="m",
                      raw_response="", parsed_answer=10.0, exec_ok=True,
                      correct=False, ref_answer=10.0)
    assert fail_mode(ar, v) == "arithmetic"
    # ran, disagrees with oracle -> formalization
    fo = GradedRecord(variant_id="1:L0:0", branch="neurosymbolic", model_id="m",
                      raw_response="", parsed_answer=42.0, exec_ok=True,
                      correct=False, ref_answer=10.0)
    assert fail_mode(fo, v) == "formalization"
