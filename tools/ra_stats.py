"""Bootstrap 95% CIs and paired McNemar tests from per-case correctness stored in
the RuleArena result rows (per_case_form / per_case_plain). Prints a table and emits
LaTeX-ready CI/p macros. No API."""
import json
import math
import random
import sys
from pathlib import Path

random.seed(0)
RES = Path(__file__).resolve().parent.parent / "data" / "results" / "rulearena"
ORDER = ["claude_opus48", "claude_sonnet5", "gpt55", "gemini35_flash", "gemini31_pro"]
LABEL = {"claude_opus48": "Opus", "claude_sonnet5": "Sonnet", "gpt55": "GPT-5.5",
         "gemini35_flash": "Gemini3.5F", "gemini31_pro": "Gemini3.1P"}


def boot_ci(hits, B=5000):
    n = len(hits)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(B):
        s = sum(hits[random.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    return (means[int(0.025 * B)], means[int(0.975 * B)])


def mcnemar(form, plain):
    # discordant pairs: b = form right & plain wrong, c = form wrong & plain right
    b = sum(1 for f, p in zip(form, plain) if f and not p)
    c = sum(1 for f, p in zip(form, plain) if not f and p)
    n = b + c
    if n == 0:
        return b, c, 1.0
    # exact binomial two-sided p under H0 p=0.5
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) * 0.5 ** n for i in range(0, k + 1))
    return b, c, min(1.0, p)


def load(domain, level=0):
    p = RES / f"{domain}_core.jsonl"
    rows = {}
    if p.exists():
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if ("per_case_form" in r and r.get("level", 0) == level
                        and r.get("code_extracted") is not False
                        and r.get("plain_sc") in (None, 1)):
                    rows[r["model"]] = r
    return rows


for domain in ["airline", "tax"]:
    rows = load(domain)
    if not rows:
        print(f"[{domain}] no per-case-logged rows yet")
        continue
    print(f"\n=== {domain} ===")
    print(f"{'model':11s} {'form (95% CI)':>20s} {'plain (95% CI)':>20s} {'gap':>6s} {'McNemar b/c':>12s} {'p':>8s}")
    for m in ORDER:
        if m not in rows:
            continue
        r = rows[m]
        f, pl = r["per_case_form"], r["per_case_plain"]
        fa, pa = sum(f) / len(f), sum(pl) / len(pl)
        fl, fh = boot_ci(f)
        pl_l, pl_h = boot_ci(pl)
        b, c, pv = mcnemar(f, pl)
        star = "*" if pv < 0.05 else " "
        print(f"{LABEL[m]:11s} {fa:.2f} [{fl:.2f},{fh:.2f}]   {pa:.2f} [{pl_l:.2f},{pl_h:.2f}]   "
              f"{fa-pa:+.2f}  {b}/{c:<8d} {pv:.3f}{star}")
