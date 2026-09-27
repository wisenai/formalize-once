"""Complete the airline difficulty sweep: the cells the published sweep never ran.

The reviewer's sharpest objection is that the accuracy result rests on one cell --
Sonnet 5 on airline -- and reads as a quirk. Table 3 answers that with a trend, but
it was run on the three cheaper models only and explicitly not on Sonnet. This adds
Sonnet and Opus at L1/L2, plus the Flash L1 cell that failed with no parseable
program, so the trend either includes the suspicious cell or it does not.

Sonnet and Opus need the plain ceiling raised: at the default 16,000 they spend the
whole ceiling reasoning and emit no answer, which grading would score wrong (a
harness artifact -- see tools/audit_truncation.py). Flash keeps the default, which
it provably never approaches, so its cached calls stay valid.

Most-suspicious cell first, so a budget halt still leaves the informative result.

Run:  python tools/run_sweep2.py
"""
import sys
sys.path.insert(0, "src")
from symbol_scramble.rulearena.core_experiment import run   # noqa: E402

PLAN = [
    (["claude_sonnet5"], 1, 60000),   # the cell under suspicion
    (["claude_sonnet5"], 2, 60000),
    (["claude_opus48"], 1, 60000),
    (["claude_opus48"], 2, 60000),
    (["gemini35_flash"], 1, None),    # previously failed: no parseable program
]

for models, level, ceiling in PLAN:
    print(f"\n=== SWEEP2 {models[0]} airline L{level} ceiling={ceiling} ===", flush=True)
    run(models, domain="airline", n_test=90, n_worked=5, level=level, n_candidates=3,
        plain_fewshot=True, plain_budget="3000", plain_sc=1, max_workers=12,
        out_tag=f"_L{level}", plain_ceiling=ceiling)

print("\nSWEEP2 DONE", flush=True)
