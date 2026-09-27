"""Is the headline Sonnet gap a completion-ceiling artifact?

The published L0 cells ran plain with a 16,000-token completion ceiling. Sonnet
averaged 10,073 completion tokens per case there and scored 0.411 -- the lowest
plain accuracy of any Sonnet configuration on record, and the number the whole
accuracy result rests on. Pilot 1 showed that at L1 the same ceiling truncates
Sonnet outright, and that with the ceiling raised it answers in ~5k tokens.

So: re-run the L0 plain branch for Sonnet and Opus with the ceiling raised, on the
same 90 cases, changing nothing else. If accuracy holds, the headline survives and
the budget asymmetry has been tested rather than spot-checked. If it rises, the gap
was partly a harness artifact and the paper's headline has to change.

Isolated output (_ceil60k); never touches the published results.

Run:  python tools/run_ceiling_retest.py
"""
import sys
sys.path.insert(0, "src")
from symbol_scramble.rulearena.core_experiment import run   # noqa: E402

for mid in ["claude_sonnet5", "claude_opus48"]:      # load-bearing cell first
    print(f"\n=== CEILING RETEST {mid} airline L0 ceiling=60000 ===", flush=True)
    run([mid], domain="airline", n_test=90, n_worked=5, level=0, n_candidates=3,
        plain_fewshot=True, plain_budget="3000", plain_sc=1, max_workers=12,
        out_tag="_ceil60k", plain_ceiling=60000)

print("\nCEILING RETEST DONE", flush=True)
