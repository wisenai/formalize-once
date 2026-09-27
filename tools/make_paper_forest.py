"""Figure 2 of the paper: the paired difference per cell, with 95% intervals.

Replaces the grouped-bar figure, which plotted exactly the accuracy columns of
Tables 1-2 and added nothing. This shows the quantity the paper actually tests --
formalize-once minus per-case accuracy on identical cases -- so the headline claim
(no cell's interval reaches below zero by more than sampling noise) is visible.

Interval: Tango (1998) score interval for the difference of paired proportions.
Not a percentile bootstrap, which degenerates on paired binary data.

Run:  python tools/make_paper_forest.py
"""
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
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

KEYS = ["Opus", "Sonnet", "GPT", "Gemini", "GeminiPro"]
MODEL_IDS = ["claude_opus48", "claude_sonnet5", "gpt55", "gemini35_flash", "gemini31_pro"]
MODELS = ["Opus 4.8", "Sonnet 5", "GPT-5.5", "Gemini 3.5 Flash", "Gemini 3.1 Pro"]

MAC = {}
for _f in ("ra_macros.tex", "ra_stats_macros.tex"):
    for _line in (ROOT / "paper" / _f).read_text().splitlines():
        _m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", _line.strip())
        if _m:
            MAC[_m.group(1)] = _m.group(2)


def stars(name):
    return "*" * MAC.get(name, "").count("*")


def cells(fname):
    """Delegates to tools/ra_cells.py: same rows, same exclusions, as the tables.

    Reading the JSONL directly here is how this figure kept plotting a cell the
    tables had already superseded."""
    import sys as _s
    _s.path.insert(0, str(ROOT / "tools"))
    from ra_cells import rows
    domain = "airline" if fname.startswith("airline") else "tax"
    level = 1 if "_L1" in fname else 2 if "_L2" in fname else 0
    return rows(domain, level)


def tango(b, c, n, z=1.96):
    def score(d):
        A = 2 * n
        B = -b - c + (2 * n - b + c) * d
        C = -c * d * (1 - d)
        q = (-B + math.sqrt(max(B * B - 4 * A * C, 0.0))) / (2 * A)
        var = n * (2 * q + d * (1 - d))
        if var <= 0:
            return math.inf if (b - c - n * d) > 0 else -math.inf
        return (b - c - n * d) / math.sqrt(var)

    def solve(lo, hi, t):
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if score(mid) - t > 0 else (lo, mid)
        return (lo + hi) / 2

    pt = (b - c) / n
    return pt, solve(-0.999, pt, z), solve(pt, 0.999, -z)


def rows_for(fname, sig_prefix):
    rows, data = [], cells(fname)
    for key, mid, name in zip(KEYS, MODEL_IDS, MODELS):
        r = data[mid]
        import sys as _s
        _s.path.insert(0, str(ROOT / "tools"))
        from ra_cells import paired
        dom = "tax" if sig_prefix == "raTax" else "airline"
        b, c, n = paired(r, dom, 0)
        rows.append((name,) + tango(b, c, n) + (stars(f"{sig_prefix}{key}Sig"),))
    return rows


def forest(ax, rows, title, xlim, xticks):
    y = list(range(len(rows)))[::-1]
    ax.axvline(0, color=INK, lw=0.7, zorder=1)
    for yi, (_, pt, lo, hi, st) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=MUTED, lw=0.9, zorder=2)
        for xe in (lo, hi):
            ax.plot([xe, xe], [yi - 0.16, yi + 0.16], color=MUTED, lw=0.9, zorder=2)
        ax.plot([pt], [yi], "o", ms=4.2, color=BLUE, mec="white", mew=0.5, zorder=3)
        ax.text(1.02, yi, f"{pt:+.2f}{st}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", clip_on=False, fontsize=7.0, color=INK,
                fontweight="bold" if st else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title(title, fontsize=8.5, pad=4, loc="left")
    ax.tick_params(axis="x", labelsize=7.5, length=2.5, width=0.6)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def bars(ax, fname, title):
    data = cells(fname)
    form = [data[m]["formalize_once_acc"] for m in MODEL_IDS]
    plain = [data[m]["plain_acc"] for m in MODEL_IDS]
    x = range(len(MODEL_IDS))
    w = 0.38
    ax.bar([i - w / 2 for i in x], form, w, color=BLUE, label="formalize-once",
           zorder=3)
    ax.bar([i + w / 2 for i in x], plain, w, color=ORANGE, label="per-case reasoning",
           zorder=3)
    for i, v in enumerate(plain):
        if v < 0.6:                      # annotate the cell that carries the result
            ax.text(i + w / 2, v + 0.04, f"{v:.2f}", ha="center", fontsize=7,
                    color=ORANGE, fontweight="bold", zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Opus", "Sonnet", "GPT", "Flash", "Pro"],
                       fontsize=7.0)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Accuracy", fontsize=8)
    ax.set_title(title, fontsize=8.5, pad=4, loc="left")
    ax.tick_params(labelsize=7.5, length=2.5, width=0.6)
    ax.yaxis.grid(True, color=RULE, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)



# Explicit axes rectangles, not tight_layout. The forest's point-estimate labels
# sit outside the axes, so tight_layout pads every column to clear them and the
# bars end up squeezed into half their column. Placing the four axes by hand
# gives the bars the full column and reserves space only where it is used.
W, H = 5.5, 3.60   # \linewidth wide; height trimmed to fit the page budget
DX = 0.49                                # offset of the right column
BAR_Y, BAR_H = 0.665, 0.255
FOR_Y, FOR_H = 0.335, 0.250
BAR_X, BAR_W = 0.175, 0.320              # flush right with the forest labels
FOR_X, FOR_W = 0.175, 0.245              # 0.42 -> 0.495 holds the estimate labels

fig = plt.figure(figsize=(W, H))
ax_bar_a = fig.add_axes([BAR_X, BAR_Y, BAR_W, BAR_H])
ax_bar_t = fig.add_axes([BAR_X + DX, BAR_Y, BAR_W, BAR_H])
ax_for_a = fig.add_axes([FOR_X, FOR_Y, FOR_W, FOR_H])
ax_for_t = fig.add_axes([FOR_X + DX, FOR_Y, FOR_W, FOR_H])

bars(ax_bar_a, "airline_core.jsonl", "Airline baggage fees")
bars(ax_bar_t, "tax_core.jsonl", "U.S. Form 1040 tax")
h_, l_ = ax_bar_a.get_legend_handles_labels()
# the only clear band is between the forest x-labels and the footnote
fig.legend(h_, l_, fontsize=7, frameon=False, ncol=2, loc="center",
           bbox_to_anchor=(0.58, 0.058), handlelength=1.1, handletextpad=0.45,
           columnspacing=2.2)

forest(ax_for_a, rows_for("airline_core.jsonl", "ra"), "",
       (-0.06, 0.24), [0.0, 0.1, 0.2])
forest(ax_for_t, rows_for("tax_core.jsonl", "raTax"), "",
       (-0.06, 0.18), [0.0, 0.1])
# A difference plot is only legible if the reader can see, without decoding a minus
# sign, which side means what. Label the direction on each side of zero instead.
for _ax in (ax_for_a, ax_for_t):
    _ax.set_xlabel(r"$\Delta$ accuracy  $=$  formalize-once $-$ per-case,"
                   "\non identical cases", fontsize=7.2, labelpad=14)
    _ax.annotate("per-case better", xy=(0, -0.30), xycoords=("data", "axes fraction"),
                 ha="right", va="center", fontsize=6.4, color=ORANGE,
                 xytext=(-4, 0), textcoords="offset points")
    _ax.annotate("formalize-once better", xy=(0, -0.30),
                 xycoords=("data", "axes fraction"), ha="left", va="center",
                 fontsize=6.4, color=BLUE, xytext=(4, 0), textcoords="offset points")
fig.text(0.58, 0.008, "Each row is one model, on n = 90 paired cases.   "
         "Paired McNemar:  * p<.05   ** p<.01   *** p<.001",
         ha="center", fontsize=6.8, color=MUTED)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "ra_forest.pdf")
print("wrote", OUT / "ra_forest.pdf")
