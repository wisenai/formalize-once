"""formalize_once's candidate sweep must not change what the protocol selects.

record_candidates exists to show how much of a cell's accuracy was the draw, which
is only honest if turning it on leaves the reported program and the reported cost
alone. These pin that down without touching the API.
"""
import pytest

import symbol_scramble.rulearena.core_experiment as C


class _Dom:
    schema = "S"

    def rules(self):
        return "R"

    def fmt_answer(self, a):
        return a


@pytest.fixture
def stub(monkeypatch):
    """Five canned candidates; the test sets each one's worked-example score."""
    codes = ["A", "B", "C", "D", "E"]
    scores = {}
    monkeypatch.setattr(C, "_one_program",
                        lambda model, mid, user, mt, ci: (codes[ci], 100, 10))
    monkeypatch.setattr(C, "_worked_score",
                        lambda code, worked, dom: scores[code])
    worked = [type("P", (), {"entities": {}, "gt_answer": 1})() for _ in range(5)]

    def draw(sc, **kw):
        scores.clear()
        scores.update(sc)
        return C.formalize_once(None, "m", _Dom(), worked, n_candidates=5, **kw)

    return draw


def test_early_stop_takes_first_perfect_candidate(stub):
    code, pt, ct, cands = stub({"A": 3, "B": 5, "C": 5, "D": 2, "E": 5})
    assert code == "B"
    assert (pt, ct) == (200, 20)
    assert len(cands) == 2


def test_sweep_draws_all_but_selects_and_charges_the_same(stub):
    """The whole point: extra candidates are information, not a different protocol."""
    plain = stub({"A": 3, "B": 5, "C": 5, "D": 2, "E": 5})
    swept = stub({"A": 3, "B": 5, "C": 5, "D": 2, "E": 5}, full_sweep=True)
    assert swept[:3] == plain[:3] == ("B", 200, 20)
    assert [c[0] for c in swept[3]] == [3, 5, 5, 2, 5]


def test_no_perfect_candidate_keeps_first_maximum_and_full_cost(stub):
    code, pt, ct, _ = stub({"A": 1, "B": 4, "C": 2, "D": 4, "E": 0})
    assert code == "B"           # first at the maximum, not the last
    assert (pt, ct) == (500, 50)


def test_perfect_on_last_draw_makes_sweep_identical(stub):
    sc = {"A": 1, "B": 1, "C": 1, "D": 1, "E": 5}
    assert stub(sc)[:3] == stub(sc, full_sweep=True)[:3] == ("E", 500, 50)
