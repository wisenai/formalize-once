"""Appendix figure: the cost advantage as a curve in the number of cases.

The paper states the cost result as a ratio at n=90 and a break-even in prose. Neither
shows the part that is actually structural: with structured inputs the advantage has no
ceiling, because formalize-once stops paying after the program is written, while on free
text it flattens, because extraction is billed per case like reasoning is. Those are
different shapes, not different constants, and a reader cannot get that from two numbers.

Both families are drawn from the same measured per-case costs as Tables 1 and 8, so the
n=90 markers sit exactly on the ratios those tables report.

Run:  python tools/make_cost_curve_figure.py
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from price_extract import rates                                     # noqa: E402

RES = ROOT / "data" / "results" / "rulearena"
OUT = ROOT / "paper" / "figures"

BLUE, ORANGE, INK, MUTED, RULE = "#2C7FB8", "#E67832", "#1F2933", "#5A6472", "#C9D1DA"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "URW Palladio L", "DejaVu Serif"],
    "font.size": 8, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.direction": "out",
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "savefig.facecolor": "white",
})

STEM = {"claude_opus48": "Opus", "claude_sonnet5": "Sonnet", "gpt55": "GPT",
        "gemini35_flash": "Gemini", "gemini31_pro": "GeminiPro"}
NAME = {"claude_opus48": "Opus 4.8", "claude_sonnet5": "Sonnet 5", "gpt55": "GPT-5.5",
        "gemini35_flash": "Gemini 3.5 Flash", "gemini31_pro": "Gemini 3.1 Pro"}
ORDER = ["claude_opus48", "claude_sonnet5", "gpt55", "gemini35_flash", "gemini31_pro"]

MAC = {m.group(1): m.group(2) for m in
       (re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", l.strip())
        for l in (ROOT / "paper" / "ra_macros.tex").read_text().splitlines()) if m}

N = 90
# structured: formalize pays C_prog once, then nothing. plain pays p per case.
struct = {}
for mid, stem in STEM.items():
    c_prog = float(MAC[f"ra{stem}CostForm"])
    p = float(MAC[f"ra{stem}CostPlain"]) / N
    struct[mid] = (c_prog, p)

# free text: formalize also pays an extraction call per case, so the ratio is bounded.
free = {}
for line in (RES / "airline_extract.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("n") != N:
        continue
    a, b = rates(r["model"])
    ept, ect, ppt, pct = r["tokens"]
    e = (ept * a + ect * b) / 1000 / N                  # extraction, per case
    fpt, fct = r["form_tokens"][:2]
    free[r["model"]] = ((fpt * a + fct * b) / 1000, e, (ppt * a + pct * b) / 1000 / N)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.45), sharey=True,
                             gridspec_kw={"wspace": 0.10})
n = np.logspace(0, 5, 600)

for i, mid in enumerate(ORDER):
    c_prog, p = struct[mid]
    ax.plot(n, n * p / c_prog, color=BLUE, lw=1.0, alpha=0.85)
    ax.plot([N], [N * p / c_prog], "o", ms=3.0, color=BLUE, mec="white", mew=0.5, zorder=5)
    if mid in free:
        cf, e, pl = free[mid]
        bx.plot(n, n * pl / (cf + n * e), color=ORANGE, lw=1.0, alpha=0.85)
        bx.plot([N], [N * pl / (cf + N * e)], "o", ms=3.0, color=ORANGE,
                mec="white", mew=0.5, zorder=5)


for a_, ttl, sub in ((ax, "Structured records", "no ceiling: one program, then free execution"),
                     (bx, "Free text", "bounded: extraction is billed per case")):
    a_.set_xscale("log"); a_.set_yscale("log")
    a_.axhline(1.0, color=INK, lw=0.7)
    a_.axvline(N, color=MUTED, lw=0.5, ls="--", alpha=0.6)
    a_.set_xlim(1, 1e5); a_.set_ylim(0.05, 4e4)
    a_.grid(True, which="major", color=RULE, lw=0.4, alpha=0.6)
    a_.set_axisbelow(True)
    a_.set_xlabel("number of cases")
    a_.set_title(ttl, fontsize=8, pad=9)
    a_.text(0.5, 1.005, sub, transform=a_.transAxes, ha="center", va="bottom",
            fontsize=6.4, color=MUTED)
    for s in ("top", "right"):
        a_.spines[s].set_visible(False)

ax.set_ylabel("cost of per-case reasoning\n$\\div$ cost of formalize-once")
ax.text(3.5e4, 1.9, "formalize-once cheaper", fontsize=6.4, color=MUTED,
        ha="right", va="bottom")
ax.text(3.5e4, 0.55, "per-case reasoning cheaper", fontsize=6.4, color=MUTED,
        ha="right", va="top")
ax.annotate("break-even", xy=(2.9, 1.0), xytext=(1.15, 0.13), fontsize=6.4, color=INK,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
asyms = sorted(pl / e for cf, e, pl in free.values())
bx.axhspan(asyms[0], asyms[-1], color=ORANGE, alpha=0.13, lw=0, zorder=0)
bx.text(1.4, asyms[-1] * 1.9,
        f"every model flattens into\nthis band: {asyms[0]:.1f}\u2013{asyms[-1]:.1f}$\\times$",
        fontsize=6.4, color=INK, va="bottom", ha="left")
# label the fastest and slowest model in each panel; the band between them is the rest
best = max(ORDER, key=lambda m: struct[m][1] / struct[m][0])
worst = min(ORDER, key=lambda m: struct[m][1] / struct[m][0])
for mid, va in ((best, "bottom"), (worst, "top")):
    c_prog, p = struct[mid]
    off = 1.0      # the backing box masks the curve, so sit the label on it
    ax.text(7e4, 7e4 * p / c_prog * off, NAME[mid], fontsize=6.0, color=BLUE,
            va="center", ha="right", bbox=dict(fc="white", ec="none", pad=0.6))
fb = max(free, key=lambda m: free[m][2] / free[m][1])
fw = min(free, key=lambda m: free[m][2] / free[m][1])
for mid, va in ((fb, "bottom"), (fw, "top")):
    cf, e, pl = free[mid]
    off = 1.0
    bx.text(7e4, 7e4 * pl / (cf + 7e4 * e) * off, NAME[mid], fontsize=6.0,
            color=ORANGE, va="center", ha="right", bbox=dict(fc="white", ec="none", pad=0.6))
for a_ in (ax, bx):
    a_.text(N, 0.062, " n=90\n as run", fontsize=6.0, color=MUTED, ha="left", va="bottom")
    a_.text(0.015, 0.94, "5 models", transform=a_.transAxes, fontsize=6.0, color=MUTED)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "ra_cost_curve.pdf", bbox_inches="tight", pad_inches=0.02)
print(f"wrote {OUT / 'ra_cost_curve.pdf'}")
print(f"  structured: ratio at n=90 = "
      f"{', '.join(f'{N * p / c:.0f}x' for c, p in (struct[m] for m in ORDER))}")
print(f"  free text : asymptotes {asyms[0]:.1f}x to {asyms[-1]:.1f}x")
