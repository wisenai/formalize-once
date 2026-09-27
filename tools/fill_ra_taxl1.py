"""Macros and table for the worked-example coverage result (harder tax tier).

paper/ra_taxl1_macros.tex, paper/tables/taxl1.tex.

This is a method result, not a failure report. Formalize-once accuracy IS formalization
fidelity, so a low number means the program is wrong -- and a program can be wrong
because the model cannot write it, or because it was never shown that a branch of the
ruleset exists. Those are different problems with different fixes, and on RuleArena's
comp_1 tier the second one dominates.

The reported comp_1 cells take the first five records as worked examples. That draw
holds four self-employed returns, one basic, one with children, and no education-credit
return, while 32 of the 90 test cases have one. Redrawing five examples to cover all
four branches, with the test set untouched and examples still taken from outside it,
lifts fidelity by 0.22 and 0.36 and moves both gaps from significant losses to a tie
and a significant win. The plain branch, which re-reads the ruleset on every case,
moves by 0.01.

Emits both draws so the comparison is visible rather than asserted.
"""
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "results" / "rulearena"
MACROS = ROOT / "paper" / "ra_taxl1_macros.tex"
TABLE = ROOT / "paper" / "tables" / "taxl1.tex"
ORDER = ["gpt55", "claude_sonnet5", "gemini35_flash", "gemini31_pro",
         "claude_opus48"]
NAME = {"gpt55": "GPT-5.5", "claude_sonnet5": "Sonnet~5",
        "gemini35_flash": "Gemini~3.5~Flash", "gemini31_pro": "Gemini~3.1~Pro",
        "claude_opus48": "Opus~4.8"}
STEM = {"gpt55": "GPT", "claude_sonnet5": "Sonnet",
        "gemini35_flash": "Gemini", "gemini31_pro": "GeminiPro",
        "claude_opus48": "Opus"}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n * 2)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def load_formbudget():
    """Formalize results at a reasoning budget verified not to bind.

    Our nominal reasoning="8000" is honoured very differently per vendor: on the
    identical formalize call Sonnet spends 48-60k completion tokens and Flash 39-59k,
    while Pro spends 6-16k and GPT 11-15k. For the two already far above the nominal
    budget it cannot be binding. For the other two we checked by raising it to 24000:
    GPT's spend did not move (13-14k) and its fidelity went 0.87 -> 0.89, while Pro's
    spend trebled and its fidelity went 0.47 -> 0.79. So the budget bound one model
    and the correction is a ceiling fix rather than a knob turned until a result
    improved -- the control model was unaffected.
    """
    out = {}
    fp = RES / "tax_L1_formbudget.jsonl"
    if fp.exists():
        for line in fp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["model"]] = r
    return out


def load(fname):
    out = {}
    for line in (RES / fname).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("n") == 90 and not sum(r.get("per_case_plain_err", [])):
                out[r["model"]] = r
    return out


def draw_interval(cands, reps=8000, seed=0):
    """How much of a cell's fidelity is the formalization draw?

    Every interval elsewhere in this paper is over the 90 test cases with the program
    held fixed, which says nothing about the step that produced the program. Best-of-N
    is already a variance control, so the question is not how much candidates vary but
    how much the SELECTED program varies -- i.e. how much the whole procedure moves if
    you happen to draw a different N.

    We resample N candidates with replacement from the N actually drawn, re-apply the
    protocol's own rule (highest worked-example score, first at the maximum), and take
    the 5th/95th percentile of the selected program's test accuracy. Cheap, and it uses
    only data the run already recorded.
    """
    pool = [(ws, a) for ws, a in cands if a is not None and ws is not None]
    if len(pool) < 3:
        return None
    rng, n, sims = random.Random(seed), len(pool), []
    for _ in range(reps):
        d = [pool[rng.randrange(n)] for _ in range(n)]
        sims.append(d[max(range(n), key=lambda i: (d[i][0], -i))][1])
    sims.sort()
    return sims[int(0.05 * reps)], sims[int(0.95 * reps)]


def cell(r):
    f, p = r["per_case_form"], r["per_case_plain"]
    b = sum(1 for i in range(len(f)) if f[i] and not p[i])
    c = sum(1 for i in range(len(f)) if p[i] and not f[i])
    return (b - c) / len(f), mcnemar(b, c)


def branch_split():
    """Fidelity on education-credit cases vs the rest, for the models measured under
    the first-five draw. Hand-typed once and drifted; generated now."""
    import sys as _s
    _s.path.insert(0, str(ROOT / "src"))
    from symbol_scramble.rulearena.core_experiment import DOMAINS
    test = DOMAINS["tax"].loader(1, limit=95)[5:95]
    ed = [i for i, v in enumerate(test)
          if v.entities.get("education_credits") not in (None, 0, "")
          or v.entities.get("american_opportunity_credit") not in (None, 0, "")]
    rest = [i for i in range(len(test)) if i not in ed]
    on, off = [], []
    for line in (RES / "tax_L1_core.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("n") != 90 or sum(r.get("per_case_plain_err", [])) or \
                r["model"] not in ("gpt55", "claude_sonnet5"):
            continue
        f = r["per_case_form"]
        on.append(sum(f[i] for i in ed) / len(ed))
        off.append(sum(f[i] for i in rest) / len(rest))
    return (len(ed), min(on), max(on), min(off), max(off)) if on else None


def main():
    orig, strat = load("tax_L1_core.jsonl"), load("tax_L1strat_core.jsonl")
    # splice in the non-binding-budget formalize results, keeping each cell's own
    # plain run (which the budget change does not touch)
    fb = load_formbudget()
    for mid, r in fb.items():
        if mid in strat and r.get("per_case_form"):
            # the probe records [worked_score, test_acc] pairs; older rows stored a
            # flat list of accuracies, so accept either
            cands = r["candidates"]
            if cands and not isinstance(cands[0], list):
                cands = [[None, a] for a in cands]
            strat[mid] = dict(strat[mid],
                              per_case_form=r["per_case_form"],
                              formalize_once_acc=r["formalize_once_acc"],
                              candidates=cands,
                              form_budget=r["reasoning"])
    # A model needs the branch-covering draw to appear at all. The first-five draw is
    # the comparison arm; where it was never run on raw inputs (Opus), that half of the
    # row is blank rather than borrowed from a different input treatment.
    present = [m for m in ORDER if m in strat]
    both = [m for m in present if m in orig]
    if not both:
        print("REFUSING to emit -- no model has both draws")
        return 1
    if len(present) > len(both):
        print("partial rows (branch-covering only):",
              [m for m in present if m not in both])

    out, body = [], []
    for mid in present:
        st = STEM[mid]
        for tag, src in (("Naive", orig), ("Strat", strat)):
            if mid not in src:
                continue
            r = src[mid]
            g, p = cell(r)
            out += [(f"raTxo{st}{tag}Form", f"{r['formalize_once_acc']:.2f}"),
                    (f"raTxo{st}{tag}Plain", f"{r['plain_acc']:.2f}"),
                    (f"raTxo{st}{tag}Gap", f"{g:+.2f}{stars(p)}"),
                    (f"raTxo{st}{tag}P", f"{p:.4f}")]
        b_ = strat[mid]
        a = orig.get(mid)
        cands = [x for _, x in b_.get("candidates", []) if x is not None]
        out += [(f"raTxo{st}Best", f"{max(cands):.2f}" if cands else "---")]
        ci = draw_interval(b_.get("candidates", []))
        out += [(f"raTxo{st}DrawLo", f"{ci[0]:.2f}" if ci else "---"),
                (f"raTxo{st}DrawHi", f"{ci[1]:.2f}" if ci else "---"),
                (f"raTxo{st}DrawW", f"{ci[1] - ci[0]:.2f}" if ci else "---")]
        if a is not None:
            out += [(f"raTxo{st}FormLift",
                     f"{b_['formalize_once_acc'] - a['formalize_once_acc']:+.2f}"),
                    (f"raTxo{st}PlainLift",
                     f"{b_['plain_acc'] - a['plain_acc']:+.2f}")]
        gs, ps = cell(b_)
        if a is None:
            left = "--- & --- & --- & "
        else:
            go, po = cell(a)
            left = (f"{a['formalize_once_acc']:.2f} & {a['plain_acc']:.2f} & "
                    f"${go:+.2f}$$^{{{stars(po)}}}$ & ")
        body.append(
            f"    {NAME[mid]} & " + left +
            f"{b_['formalize_once_acc']:.2f} & "
            + (f"[{ci[0]:.2f},\\,{ci[1]:.2f}]" if ci else "---")
            + f" & {b_['plain_acc']:.2f} & "
            f"${gs:+.2f}$$^{{{stars(ps)}}}$ \\\\")

    d = dict(out)
    best = {m: float(d[f"raTxo{STEM[m]}Best"]) for m in present
            if d.get(f"raTxo{STEM[m]}Best", "---") != "---"}
    CAN = 0.75                       # best candidate drawn clears this => can formalize
    able = [m for m in present if best.get(m, 0) >= CAN]
    gated = [m for m in present if best.get(m, 0) < CAN]
    lifts = [float(d[f"raTxo{STEM[m]}FormLift"]) for m in able
             if f"raTxo{STEM[m]}FormLift" in d]
    plifts = [abs(float(d[f"raTxo{STEM[m]}PlainLift"])) for m in present
              if f"raTxo{STEM[m]}PlainLift" in d]
    out += [("raTxoLiftLo", f"{min(lifts):+.2f}"), ("raTxoLiftHi", f"{max(lifts):+.2f}"),
            ("raTxoPlainLiftHi", f"{max(plifts):.2f}"),
            ("raTxoNmodels", str(len(present))),
            ("raTxoDrawWidest", max((d[f"raTxo{STEM[m]}DrawW"] for m in present
                                     if d.get(f"raTxo{STEM[m]}DrawW", "---") != "---"),
                                    default="---")),
            ("raTxoNable", str(len(able))),
            ("raTxoNgated", str(len(gated))),
            ("raTxoAbleFormLo", f"{min(float(d[f'raTxo{STEM[m]}StratForm']) for m in able):.2f}"),
            ("raTxoAbleFormHi", f"{max(float(d[f'raTxo{STEM[m]}StratForm']) for m in able):.2f}"),
            ("raTxoGatedBest", f"{min(best[m] for m in gated):.2f}" if gated else "---"),
            ("raTxoAbleBestLo", f"{min(best[m] for m in able):.2f}")]
    bs = branch_split()
    if bs:
        n_ed, on_lo, on_hi, off_lo, off_hi = bs
        out += [("raTxoEdN", str(n_ed)),
                ("raTxoEdAcc", f"{on_lo:.2f}" if abs(on_hi - on_lo) < 0.005
                 else f"{on_lo:.2f}--{on_hi:.2f}"),
                ("raTxoRestLo", f"{off_lo:.2f}"), ("raTxoRestHi", f"{off_hi:.2f}")]
    MACROS.write_text("% generated by tools/fill_ra_taxl1.py -- do not edit\n"
                      + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in out))
    TABLE.write_text(
        "% generated by tools/fill_ra_taxl1.py -- do not edit\n"
        "\\small\\begin{tabular}{lccccccc}\n    \\toprule\n"
        "    & \\multicolumn{3}{c}{Examples: first five records} & "
        "\\multicolumn{4}{c}{Examples: branch-covering}\\\\\n"
        "    \\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\n"
        "    Model & Formalize & Plain & Gap & Formalize & 90\\% draw & Plain & Gap \\\\\n"
        "    \\midrule\n" + "\n".join(body) + "\n    \\bottomrule\n\\end{tabular}\n")
    print(f"wrote {MACROS.name} ({len(out)} macros), tables/{TABLE.name}")
    print(f"fidelity lift {min(lifts):+.2f} to {max(lifts):+.2f}; "
          f"plain moves at most {max(plifts):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
