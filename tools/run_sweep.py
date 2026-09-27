"""Difficulty sweep: does the formalize-once vs plain accuracy gap GROW with rule-
application difficulty? Runs the cheap models on the harder airline tiers (L1, L2) that
RuleArena already ships, using the exact headline config. If a model that ties at L0
opens a gap at L1/L2, the Sonnet-airline result is the extreme of a trend, not an outlier.

Isolated output (out_tag='_L{level}'); never touches the headline results. Cache-safe and
halt-loudly (stops + stores on budget/empty). Hardest tier first, so a budget halt still
leaves the most informative result.
"""
import sys
sys.path.insert(0, "tools")
from symbol_scramble.rulearena.core_experiment import run

# Most-informative-first (Gemini-Pro already has an L0 gap; GPT ties at L0), so a budget
# halt still leaves the key cells. Flash last (cheap context; it ties even at L2).
MODELS = ["gemini31_pro", "gpt55", "gemini35_flash"]
LEVELS = [2, 1]  # L2 resumes (only Flash left, cached rows skip); then L1 for the full trend

for level in LEVELS:
    print(f"\n=== SWEEP airline L{level} ===", flush=True)
    run(MODELS, domain="airline", n_test=90, n_worked=5, level=level, n_candidates=3,
        plain_fewshot=True, plain_budget="3000", plain_sc=1, max_workers=12,
        out_tag=f"_L{level}")

print("\nSWEEP DONE", flush=True)
