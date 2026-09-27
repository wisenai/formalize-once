"""Contamination-robustness run: repeat the headline formalize-once vs plain comparison
on freshly GENERATED cases (novel inputs, oracle-relabeled) instead of RuleArena's public
fixed cases. If formalize-once's accuracy edge holds on inputs no model could have
memorized, the result is not a contamination artifact.

Writes only to *_fresh_core.jsonl (out_tag='_fresh'); never touches the main results.
Every LLM call is disk-cached, so it is safe to interrupt and resume for free.
Single-pass plain (the headline config); same models, n, and candidate budget.

The first attempt at this run was withdrawn after review found three defects. All
three are fixed and each is checked before the run, not assumed:
  1. plain calls that errored were scored as wrong answers -- the runner now records
     them in per_case_plain_err and the analysis excludes them (tools/ra_cells.py);
  2. plain ran against a 16,000-token completion ceiling while formalize had 60,000,
     so a long reasoner was truncated into wrong answers -- plain_ceiling is matched
     here, and tools/audit_truncation.py checks every cell afterwards;
  3. the generator resampled bag dimensions independently and pushed 56% of bags
     outside the public priced envelope (>115 linear inches) -- gen_fresh.py now
     reject-samples on the sum, verified at 0% outside before this run.
"""
import sys
sys.path.insert(0, "tools")
from gen_fresh import fresh_cases
from symbol_scramble.rulearena.core_experiment import run

MODELS = ["gemini35_flash", "gpt55", "gemini31_pro", "claude_sonnet5", "claude_opus48"]
# (domain, n_candidates matching the headline)
# Airline only. The tax generator has not been checked against the public
# marginals the way the airline one now is (tools/check_fresh_match.py), and an
# unmatched distribution makes a contamination control uninterpretable rather
# than merely noisy -- that is what withdrew the first airline run.
CELLS = [("airline", 3)]
N_TEST, N_WORKED = 90, 5

for domain, nc in CELLS:
    fresh = fresh_cases(domain, N_TEST + N_WORKED)
    print(f"\n=== FRESH {domain}: {len(fresh)} generated cases (novel, oracle-labeled) ===",
          flush=True)
    run(MODELS, domain=domain, n_test=N_TEST, n_worked=N_WORKED, n_candidates=nc,
        plain_fewshot=True, plain_budget="3000", plain_sc=1, max_workers=8,
        problems=fresh, out_tag="_fresh", plain_ceiling=60000)

print("\nALL FRESH CELLS DONE", flush=True)
