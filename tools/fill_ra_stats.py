"""Emit paper/ra_stats_macros.tex: per-model McNemar p, significance markers, and
key 95% bootstrap CIs from the per-case correctness in the matched result rows."""
import json
import math
import random
from pathlib import Path

random.seed(0)
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "results" / "rulearena"
OUT = ROOT / "paper" / "ra_stats_macros.tex"
SHORT = {"claude_opus48": "Opus", "claude_sonnet5": "Sonnet", "gpt55": "GPT",
         "gemini35_flash": "Gemini", "gemini31_pro": "GeminiPro"}


def boot_ci(hits, B=5000):
    n = len(hits)
    m = sorted(sum(hits[random.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return m[int(0.025 * B)], m[int(0.975 * B)]


def mcnemar_p(b, c):
    """Exact two-sided McNemar p from discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) * 0.5 ** n for i in range(k + 1)))


def sig(p):
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def load(domain, level=0):
    """Delegates to tools/ra_cells.py so every generator selects the same rows."""
    from ra_cells import rows
    return rows(domain, level)


lines = ["% Auto-generated significance macros (McNemar p, 95% bootstrap CIs)."]
counts = {"airline": {"sig": 0, "rev": 0, "tot": 0}, "tax": {"sig": 0, "rev": 0, "tot": 0}}
for domain, pref in [("airline", "ra"), ("tax", "raTax")]:
    for mk, short in SHORT.items():
        rows = load(domain)
        if mk not in rows:
            continue
        r = rows[mk]
        # discordant counts over cases that actually produced an answer: a case
        # that never ran or never answered is not evidence against per-case
        # reasoning (see tools/ra_cells.py)
        from ra_cells import paired
        b, c, _n = paired(r, domain, 0)
        p = mcnemar_p(b, c)
        counts[domain]["tot"] += 1
        if p < 0.05 and b > c:
            counts[domain]["sig"] += 1
        if p < 0.05 and c > b:
            counts[domain]["rev"] += 1
        lines.append(f"\\newcommand{{\\{pref}{short}Sig}}{{{sig(p)}}}")
        lines.append(f"\\newcommand{{\\{pref}{short}P}}{{{p:.3f}}}")
    # CI for the standout cells (plain has the wide interval)
for dom, mk, name in [("airline", "claude_sonnet5", "raSonnetPlainCI")]:
    r = load(dom)[mk]
    lo, hi = boot_ci(r["per_case_plain"])
    lines.append(f"\\newcommand{{\\{name}}}{{[{lo:.2f},\\,{hi:.2f}]}}")
# Which cells survive a Bonferroni correction across all tests, and which has the
# smallest p. Both were hand-typed in the results opener and both had gone stale
# after the completion-ceiling correction moved the cells around: it named Sonnet as
# a survivor (p=0.041) and as the smallest p (it is Gemini 3.1 Pro, p=0.002).
NICE = {"claude_opus48": "Opus~4.8", "claude_sonnet5": "Sonnet~5",
        "gpt55": "GPT-5.5", "gemini35_flash": "Gemini~3.5~Flash",
        "gemini31_pro": "Gemini~3.1~Pro"}
allcells = []
for dom in ("airline", "tax"):
    for mk, r in load(dom).items():
        from ra_cells import paired as _paired
        b, c, n = _paired(r, dom, 0)
        allcells.append((NICE[mk], dom, (b - c) / n, mcnemar_p(b, c)))
ntests = len(allcells)
surv = [x for x in allcells if x[3] < 0.05 / ntests]
best = min(allcells, key=lambda x: x[3])
lines.append(f"\\newcommand{{\\raNbonf}}{{{len(surv)}}}")
lines.append("\\newcommand{\\raBonfNames}{"
             + " and ".join(f"{m} on {d}" for m, d, _, _ in surv) + "}")
lines.append(f"\\newcommand{{\\raBestCell}}{{{best[0]} on {best[1]}}}")
lines.append(f"\\newcommand{{\\raBestP}}{{{best[3]:.3f}}}")
print("Bonferroni survivors:", [m for m, d, _, _ in surv], "| smallest p:", best[0], round(best[3], 4))

tot = sum(counts[d]["tot"] for d in counts)
sign = sum(counts[d]["sig"] for d in counts)
rev = sum(counts[d]["rev"] for d in counts)
lines.append(f"\\newcommand{{\\raNsig}}{{{sign}}}")
lines.append(f"\\newcommand{{\\raNcells}}{{{tot}}}")
lines.append(f"\\newcommand{{\\raNrev}}{{{rev}}}")
OUT.write_text("\n".join(lines) + "\n")
print("wrote", OUT)
print(f"significant formalize>plain: {sign}/{tot} cells; plain>formalize significant: {rev}/{tot}")
