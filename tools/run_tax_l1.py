"""Finish tax Level 1 only.

Level 2 is deliberately not run. At Level 1 the five worked examples have already
stopped separating candidate programs -- Gemini 3.1 Pro's selection took a program
scoring 0.00 with a 0.50 candidate in the same draw -- so the cells measure the draw
as much as the model. Another tier of that is not worth buying; the candidate spreads
recorded here make the point directly.
"""
import sys

sys.path.insert(0, "src")
from symbol_scramble.rulearena.core_experiment import run   # noqa: E402

run(["claude_sonnet5", "claude_opus48", "gemini31_pro", "gemini35_flash", "gpt55"],
    domain="tax", n_test=90, n_worked=5, level=1, n_candidates=5,
    plain_fewshot=True, plain_budget="3000", plain_sc=1, max_workers=3,
    out_tag="_L1", plain_ceiling=60000, record_candidates=True)
