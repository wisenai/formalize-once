"""Unfilled slots on RuleArena's harder tax forms reach both branches unchanged.

The records are blank-form encodings: a line the taxpayer left empty arrives as
"$TBD" or "[__]". We briefly normalized these to 0 because a generated program
raises TypeError on them. That was wrong -- it preserved every ground-truth label
while changing what the input means to a model, and per-case accuracy on
child-credit returns collapsed from ~0.73 to 0.00. These pin the revert.
"""
import pytest

from symbol_scramble.rulearena import oracle
from symbol_scramble.rulearena.loader import TAX_BLANKS, load_tax

BLANKS = ("$TBD", "[__]")


def test_placeholders_survive_loading():
    """The model must see the blank, not a zero: 0 reads as 'the credit is zero'."""
    found = [v for v in load_tax(1, limit=95)
             if any(isinstance(x, str) and x.strip() in BLANKS
                    for x in v.entities.values())]
    assert found, "comp_1 should carry unfilled form slots"


def test_level_zero_has_none():
    """comp_0 is unaffected either way, which is why the published cells stand."""
    for v in load_tax(0, limit=95):
        assert not [x for x in v.entities.values()
                    if isinstance(x, str) and x.strip() in BLANKS]


@pytest.mark.parametrize("level", [0, 1, 2])
def test_labels_agree_with_the_oracle(level):
    for v in load_tax(level, limit=95):
        assert abs(oracle.tax_answer(v.entities) - v.gt_answer) <= 1e-6


def test_the_oracle_is_indifferent_but_that_was_never_the_question():
    """Documents the check that misled us: it is true, and it proved the wrong thing.

    Zeroing the placeholders changes no label. It changes the model's reading, which
    is what actually mattered, so label-invariance is kept here as a fact about the
    oracle rather than as a licence to transform the input."""
    from symbol_scramble.rulearena.loader import _fill_blanks
    for v in load_tax(1, limit=95):
        if any(isinstance(x, str) and x.strip() in BLANKS for x in v.entities.values()):
            assert abs(oracle.tax_answer(_fill_blanks(v.entities))
                       - v.gt_answer) <= 1e-6
