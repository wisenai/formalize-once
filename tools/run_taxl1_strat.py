"""Re-run tax comp_1 with worked examples that cover the ruleset's branches.

tools/taxl1_stratified.py showed the reported comp_1 fidelity was largely an artifact
of the worked-example draw: the first five records contain no education-credit return
while 32 of the 90 test cases do, and the programs scored 0.22 there. Adding one
education example lifts fidelity 0.64 -> 0.87 (GPT-5.5) and 0.53 -> 0.89 (Sonnet 5).

That test changed only the formalize branch, so its comparison against the reported
plain numbers is not few-shot matched. This re-runs BOTH branches on the stratified
examples, keeping the same 90 test cases, so the paired test is honest again.

The test set is untouched (records 5-94). The worked examples are five records drawn
from 0-4 and 95-99 -- outside the test set either way -- chosen to cover all four
comp_1 categories.

Run:  .venv/bin/python tools/run_taxl1_strat.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from taxl1_stratified import pick_stratified                        # noqa: E402
from symbol_scramble.rulearena.core_experiment import DOMAINS, run  # noqa: E402

dom = DOMAINS["tax"]
probs = dom.loader(1, limit=500)
test = probs[5:95]
worked = pick_stratified(probs[0:5] + probs[95:])
# run() takes worked = problems[:5] and test = problems[5:95]
problems = worked + test
assert len(problems) == 95 and not ({id(w) for w in worked} & {id(t) for t in test})

run(["gemini31_pro", "gemini35_flash", "gpt55", "claude_sonnet5"],
    domain="tax", n_test=90, n_worked=5, level=1, n_candidates=5,
    plain_fewshot=True, plain_budget="3000", plain_sc=1, max_workers=3,
    problems=problems, out_tag="_L1strat", plain_ceiling=60000,
    record_candidates=True)
