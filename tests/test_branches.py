"""M3: branch logic with a mock model (no API calls)."""
from symbol_scramble.branches.solvers import NeuroSymbolic, PlainCoT, Scaled
from symbol_scramble.llm.client import LLMResponse
from symbol_scramble.schemas import Variant


class MockModel:
    model_id = "mock"
    reasoning_supported = False

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def complete(self, system, user, max_tokens=2048, temperature=0.0, reasoning=None):
        t = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return LLMResponse(text=t, prompt_tokens=10, completion_tokens=20, model_id="mock")


def _v(ot="decimal", gt=10.0, lo=9.5, hi=10.5):
    return Variant(
        variant_id="1:L0:0", seed_row_id=1, calc_id="2", calc_name="x",
        family="equation", output_type=ot, level="L0", note="n", question="q",
        entities={}, gt_answer=gt, lower=lo, upper=hi, answer_invariant=True,
    )


def test_plain_cot_parses():
    out = PlainCoT().solve(_v(), MockModel(["reasoning...\nFINAL ANSWER: 10.0"]))
    assert out.branch == "plain_cot"
    assert out.parsed_answer == 10.0


def test_scaled_best_of_n_consensus():
    texts = ["FINAL ANSWER: 10.0", "FINAL ANSWER: 12.0", "FINAL ANSWER: 10.5"]
    m = MockModel(texts)
    out = Scaled(best_of_n=3).solve(_v(), m)
    assert m.calls == 3  # three samples
    assert out.parsed_answer == 10.5  # median
    assert out.completion_tokens == 60


def test_neurosymbolic_executes_code():
    code = "```python\ndef solve():\n    return 10.0\n```"
    out = NeuroSymbolic(refine_k=2).solve(_v(), MockModel([code]))
    assert out.exec_ok is True
    assert out.parsed_answer == 10.0
    assert out.n_refine == 0


def test_neurosymbolic_refines_on_error():
    bad = "```python\ndef solve():\n    return undefined_var\n```"
    good = "```python\ndef solve():\n    return 10.0\n```"
    m = MockModel([bad, good])
    out = NeuroSymbolic(refine_k=2).solve(_v(), m)
    assert out.exec_ok is True
    assert out.parsed_answer == 10.0
    assert out.n_refine == 1  # one repair


def test_neurosymbolic_gives_up_gracefully():
    bad = "```python\ndef solve():\n    return 1/0\n```"
    out = NeuroSymbolic(refine_k=1).solve(_v(), MockModel([bad]))
    assert out.exec_ok is False
    assert out.parsed_answer is None
