"""The cost advantage is a curve in the number of cases, not a ratio at n=90.

Formalize-once pays a fixed price to write the program and then almost nothing per
case; per-case reasoning pays the same price on every case. So the advantage is not a
constant -- it is negative at n=1, crosses zero at a break-even point, and grows after
that. Reporting a single ratio at whatever n the benchmark happens to supply hides both
the break-even (the number a practitioner actually needs) and the shape.

The two input settings have qualitatively different curves, which is the part worth
stating:

  structured inputs   formalize pays C_prog once, then 0 per case (execution is local).
                      Ratio = n*p / C_prog, linear in n and unbounded.

  free text           formalize pays C_prog once, then an extraction call per case.
                      Ratio = n*p / (C_prog + n*e), which asymptotes to p/e.

So unstructured input does not just shrink the advantage, it changes it from a growing
one to a bounded one. That is a more useful claim than either single ratio.

Run:  .venv/bin/python tools/cost_curve.py
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from price_extract import rates                                     # noqa: E402

RES = ROOT / "data" / "results" / "rulearena"
OUT = ROOT / "paper" / "ra_cost_macros.tex"
NAME = {"claude_opus48": "Opus~4.8", "claude_sonnet5": "Sonnet~5", "gpt55": "GPT-5.5",
        "gemini35_flash": "Gemini~3.5~Flash", "gemini31_pro": "Gemini~3.1~Pro"}
STEM = {"claude_opus48": "Opus", "claude_sonnet5": "Sonnet", "gpt55": "GPT",
        "gemini35_flash": "Gemini", "gemini31_pro": "GeminiPro"}
MAC = {m.group(1): m.group(2) for m in
       (re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", l.strip())
        for l in (ROOT / "paper" / "ra_macros.tex").read_text().splitlines()) if m}


def main():
    out, rows = [], []
    for mid, stem in STEM.items():
        # structured airline cell, as priced in Table 1
        c_prog = float(MAC[f"ra{stem}CostForm"])
        c_plain = float(MAC[f"ra{stem}CostPlain"])
        p = c_plain / 90.0                       # per case, reasoning
        # the first whole case count at which formalize-once is actually cheaper.
        # Rounding the continuous crossing understates this by one whenever the
        # crossing falls just above an integer, which is three of five models here.
        breakeven = math.ceil(c_prog / p)
        if breakeven * p <= c_prog:
            breakeven += 1
        out += [(f"raCv{stem}Break", f"{breakeven:.0f}"),
                (f"raCv{stem}PerCase", f"{p:.3f}")]
        rows.append((NAME[mid], c_prog, p, breakeven))

    # free text: extraction is per case, so the ratio is bounded
    asym = []
    for line in (RES / "airline_extract.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("n") != 90:
            continue
        rt = rates(r["model"])
        if not rt:
            continue
        a, b = rt
        ept, ect, ppt, pct = r["tokens"]
        e = (ept * a + ect * b) / 90.0           # extraction, per case
        pl = (ppt * a + pct * b) / 90.0          # read-and-reason, per case
        asym.append((NAME[r["model"]], pl / e))
        out += [(f"raCv{STEM[r['model']]}Asym", f"{pl / e:.1f}")]

    # Break-even on airline only understates how fast tax pays off, and a reader who
    # divides the tax columns gets a smaller number than the one we print. Report the
    # range over both rulesets, and keep the per-ruleset ranges so prose can be exact.
    tax_breaks = []
    for mid, stem in STEM.items():
        ktf, ktp = f"raTax{stem}CostForm", f"raTax{stem}CostPlain"
        if ktf not in MAC or ktp not in MAC:
            continue
        cf, pc = float(MAC[ktf]), float(MAC[ktp]) / 90.0
        n = math.ceil(cf / pc)
        if n * pc <= cf:
            n += 1
        tax_breaks.append(n)
    air_lo, air_hi = min(r[3] for r in rows), max(r[3] for r in rows)
    out += [("raCvBreakAirLo", f"{air_lo:.0f}"), ("raCvBreakAirHi", f"{air_hi:.0f}")]
    if tax_breaks:
        out += [("raCvBreakTaxLo", f"{min(tax_breaks)}"),
                ("raCvBreakTaxHi", f"{max(tax_breaks)}")]
    lo = min([air_lo] + tax_breaks)
    hi = max([air_hi] + tax_breaks)

    # The obvious objection to a 10-50x headline is that per-case reasoning re-sends the
    # same ruleset prefix on every call and every vendor sells cached input cheaply. It
    # does not bite, because completion tokens are most of that bill: quantify it here
    # rather than leave the reader to wonder.
    cin = cout = 0.0
    for line in (RES / "airline_core.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("level", 0) != 0 or "plain_tokens" not in r:
            continue
        rt = rates(r["model"])
        cin += r["plain_tokens"][0] * rt[0] / 1000
        cout += r["plain_tokens"][1] * rt[1] / 1000
    share = cout / (cin + cout)
    out += [("raCvPlainOutShare", f"{share * 100:.0f}"),
            ("raCvCacheShrink", f"{1 / share:.2f}")]
    alo, ahi = min(x[1] for x in asym), max(x[1] for x in asym)
    out += [("raCvBreakLo", f"{lo:.0f}"), ("raCvBreakHi", f"{hi:.0f}"),
            ("raCvAsymLo", f"{alo:.1f}"), ("raCvAsymHi", f"{ahi:.1f}")]

    print("STRUCTURED INPUT -- ratio grows linearly with n, no ceiling")
    print(f"  {'model':18s}{'program $':>11s}{'plain $/case':>14s}{'break-even n':>14s}"
          f"{'ratio n=90':>12s}{'ratio n=1000':>14s}")
    for nm, cp, p, be in rows:
        print(f"  {nm.replace('~',' '):18s}{cp:11.2f}{p:14.3f}{be:14.0f}"
              f"{90 * p / cp:12.0f}x{1000 * p / cp:13.0f}x")
    print(f"\n  break-even: {lo:.0f} to {hi:.0f} cases")

    print("\nFREE TEXT -- extraction is per case, so the ratio is bounded")
    print(f"  {'model':18s}{'asymptote':>11s}")
    for nm, r in asym:
        print(f"  {nm.replace('~',' '):18s}{r:10.1f}x")
    print(f"\n  ceiling: {alo:.1f}x to {ahi:.1f}x however many cases you run")

    OUT.write_text("% generated by tools/cost_curve.py -- do not edit\n"
                   + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in out))
    print(f"\nwrote {OUT.name} ({len(out)} macros)")


if __name__ == "__main__":
    main()
