"""Sampling-matched robustness run: give the plain branch self-consistency@N (matching
formalize's best-of-N) on the three cells where formalize-once significantly wins, to
show its accuracy edge is not an artifact of best-of-N sampling. Budget stays at plain's
normal 3000 (the budget asymmetry is covered separately by the Sonnet-airline spot-check).

Every LLM call is disk-cached (core_experiment.CALLCACHE), so this is safe to interrupt
(lid close, network drop): re-running replays completed calls for free and only makes the
missing ones. Flagship cell first.
"""
from symbol_scramble.rulearena.core_experiment import run

# (models, domain, n_candidates == plain_sc) for each significant cell.
CELLS = [
    (["claude_sonnet5"], "airline", 3),   # flagship: +0.51, p<0.001
    (["gemini31_pro"], "airline", 3),      # +0.11, p=0.002 (survives Bonferroni)
    (["claude_sonnet5"], "tax", 5),        # +0.08, p=0.016 (marginal)
]

for models, domain, nc in CELLS:
    print(f"\n=== matched cell: {models[0]} / {domain}  (SC@{nc}, budget=3000) ===",
          flush=True)
    run(models, domain=domain, n_test=90, n_worked=5, level=0, n_candidates=nc,
        plain_fewshot=True, plain_budget="3000", plain_sc=nc, max_workers=12)

print("\nALL MATCHED CELLS DONE", flush=True)
