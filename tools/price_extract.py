"""Price the extraction run before buying it, from tokens this project already paid for.

Per-token rates are not hardcoded here. Each model's airline and tax Level-0 plain
cells give two equations in the two unknowns ($/prompt token, $/completion token),
using the published cost macros and the token totals stored in the result rows; the
pilot supplies this run's tokens per case. Solving beats guessing a rate card that
may have moved since the runs.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "results" / "rulearena"
STEM = {"claude_opus48": "Opus", "claude_sonnet5": "Sonnet", "gpt55": "GPT",
        "gemini35_flash": "Gemini", "gemini31_pro": "GeminiPro"}
MAC = {m.group(1): m.group(2) for m in
       (re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", l.strip())
        for l in (ROOT / "paper" / "ra_macros.tex").read_text().splitlines()) if m}


def plain_row(fname, mid):
    best = None
    for l in (RES / fname).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if (r.get("model") == mid and r.get("n") == 90
                    and r.get("level", 0) == 0 and r.get("plain_sc", 1) == 1):
                best = r
    return best


def rates(mid):
    """($/prompt token, $/completion token) solved from the two published cells."""
    eqs = []
    for fname, pre in (("airline_core.jsonl", "ra"), ("tax_core.jsonl", "raTax")):
        r, key = plain_row(fname, mid), f"{pre}{STEM[mid]}CostPlain"
        if r and key in MAC:
            eqs.append((r["plain_tokens"][0], r["plain_tokens"][1], float(MAC[key])))
    if len(eqs) < 2:
        return None
    (p1, c1, v1), (p2, c2, v2) = eqs[0], eqs[1]
    det = p1 * c2 - p2 * c1
    if abs(det) < 1e-9:
        return None
    a = (v1 * c2 - v2 * c1) / det
    b = (p1 * v2 - p2 * v1) / det
    return (a, b) if a > 0 and b > 0 else None


def main():
    pilot = {}
    for l in (RES / "airline_extract.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r["n"] == 3:
                pilot[r["model"]] = r["tokens"]          # [ept, ect, ppt, pct]
    if not pilot:
        sys.exit("no pilot rows (n=3) -- run tools/run_extract.py --pilot first")
    ref = [sum(v[i] for v in pilot.values()) / len(pilot) for i in range(4)]
    print(f"{'model':16s} {'extract $':>10s} {'read+reason $':>14s} {'cell $':>9s}")
    tot = 0.0
    for mid in STEM:
        rt = rates(mid)
        if rt is None:
            print(f"{mid:16s} {'-- no rate solution --':>36s}")
            continue
        a, b = rt
        tk = pilot.get(mid, ref)                          # measured, else pilot mean
        scale = 90 / 3
        ex = (tk[0] * a + tk[1] * b) * scale
        pl = (tk[2] * a + tk[3] * b) * scale
        meas = "" if mid in pilot else "  (extrapolated from pilot mean)"
        print(f"{mid:16s} {ex:10.2f} {pl:14.2f} {ex + pl:9.2f}{meas}")
        tot += ex + pl
    print(f"{'TOTAL':16s} {'':>10s} {'':>14s} {tot:9.2f}")


if __name__ == "__main__":
    main()
