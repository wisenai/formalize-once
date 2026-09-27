"""Macros and the table for the extraction experiment (paper/ra_extract_macros.tex,
paper/tables/extract.tex).

The experiment answers the objection that the main study hands both branches a
structured dict. Behind prose, each case gives four numbers per model: the program's
accuracy on the true dict (the ceiling), how often extraction recovers the dict
exactly, the end-to-end accuracy of extract-then-run, and per-case reasoning read
straight off the prose. The paired McNemar test is between the last two, which are
the two pipelines a deployment would actually choose between.

Intervals are Tango (1998) score intervals for a paired difference of proportions,
the same estimator Figure 2 uses, so the two agree by construction.

Run:  .venv/bin/python tools/fill_ra_extract.py
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "results" / "rulearena" / "airline_extract.jsonl"
MACROS = ROOT / "paper" / "ra_extract_macros.tex"
TABLE = ROOT / "paper" / "tables" / "extract.tex"

ORDER = ["claude_opus48", "claude_sonnet5", "gpt55", "gemini35_flash", "gemini31_pro"]
NAME = {"claude_opus48": "Opus~4.8", "claude_sonnet5": "Sonnet~5", "gpt55": "GPT-5.5",
        "gemini35_flash": "Gemini~3.5~Flash", "gemini31_pro": "Gemini~3.1~Pro"}
STEM = {"claude_opus48": "Opus", "claude_sonnet5": "Sonnet", "gpt55": "GPT",
        "gemini35_flash": "Gemini", "gemini31_pro": "GeminiPro"}
# fields the fee rules never read are not in oracle_view at all, so every label here
# is a field that CAN change the answer
FIELD_LABEL = {"base_price": "fare", "customer_class": "cabin", "routine": "region",
               "direction": "direction", "bag_count": "bag count",
               "bag_size": "bag size", "bag_weight": "bag weight",
               "unparseable": "unparseable"}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


def tango(b, c, n, z=1.96):
    def score(d):
        A, B = 2 * n, -b - c + (2 * n - b + c) * d
        Cc = -c * d * (1 - d)
        q = (-B + math.sqrt(max(B * B - 4 * A * Cc, 0.0))) / (2 * A)
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


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def unit_costs(mid):
    """($/prompt token, $/completion token) solved from this project's own paid cells."""
    sys.path.insert(0, str(ROOT / "tools"))
    from price_extract import rates
    return rates(mid)


def load():
    rows = {}
    for line in SRC.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("n") == 90:            # the reported cell; 3 == pilot
                rows[r["model"]] = r
    return rows


def main():
    rows = load()
    missing = [m for m in ORDER if m not in rows]
    if missing:
        print("WARNING: no 90-case row yet for", ", ".join(missing))
    out, body = [], []
    tot = {}
    for mid in ORDER:
        r = rows.get(mid)
        if r is None:
            continue
        e2e, pln = r["per_case_e2e"], r["per_case_plaintext"]
        n = len(e2e)
        b = sum(1 for i in range(n) if e2e[i] and not pln[i])
        c = sum(1 for i in range(n) if pln[i] and not e2e[i])
        p = mcnemar(b, c)
        gap, lo, hi = tango(b, c, n)
        st = STEM[mid]
        out += [(f"raEx{st}Struct", f"{r['form_struct_acc']:.2f}"),
                (f"raEx{st}Fid", f"{r['extract_exact_acc']:.2f}"),
                (f"raEx{st}Keeps", f"{r['extract_keeps_acc']:.2f}"),
                (f"raEx{st}Etoe", f"{r['e2e_acc']:.2f}"),
                (f"raEx{st}Plain", f"{r['plaintext_acc']:.2f}"),
                (f"raEx{st}Gap", f"{gap:+.2f}{stars(p)}"),
                (f"raEx{st}Lo", f"{lo:+.2f}"), (f"raEx{st}Hi", f"{hi:+.2f}"),
                (f"raEx{st}P", f"{p:.3f}")]
        body.append(f"    {NAME[mid]} & {r['form_struct_acc']:.2f} & "
                    f"{r['extract_keeps_acc']:.2f} & {r['e2e_acc']:.2f} & "
                    f"{r['plaintext_acc']:.2f} & ${gap:+.2f}$$^{{{stars(p)}}}$ \\\\")
        rt = unit_costs(mid)
        if rt:
            a, b_ = rt
            ept, ect, ppt, pct = r["tokens"]
            fpt, fct = r.get("form_tokens", [0, 0])
            # extract-then-run pays one extraction call per case plus the one-time
            # program; read-and-reason pays one reasoning call per case. Program
            # execution is local and free, which is the whole point.
            c_ex = ept * a + ect * b_ + (fpt * a + fct * b_)
            c_pl = ppt * a + pct * b_
            out += [(f"raEx{st}CostExtract", f"{c_ex:.2f}"),
                    (f"raEx{st}CostPlain", f"{c_pl:.2f}"),
                    (f"raEx{st}CostRatio", f"{c_pl / c_ex:.1f}")]
            body[-1] = body[-1].replace(" \\\\", f" & {c_pl / c_ex:.1f}$\\times$ \\\\")
        for d in r["diffs"]:
            for f in d:
                tot[f] = tot.get(f, 0) + 1

    if body:
        drops = [float(dict(out)[f"raEx{STEM[m]}Struct"])
                 - float(dict(out)[f"raEx{STEM[m]}Etoe"])
                 for m in ORDER if m in rows]
        out += [("raExDropHi", f"{max(drops):.2f}")]
        gaps = [float(v.rstrip("*")) for k, v in out if k.endswith("Gap")]
        fids = [float(v) for k, v in out if k.endswith("Fid")]
        e2es = [float(v) for k, v in out if k.endswith("Etoe")]
        ratios = [float(v) for k, v in out if k.endswith("CostRatio")]
        if ratios:
            out += [("raExRatioLo", f"{min(ratios):.1f}"),
                    ("raExRatioHi", f"{max(ratios):.1f}")]
        out += [("raExGapLo", f"{min(gaps):+.2f}"), ("raExGapHi", f"{max(gaps):+.2f}"),
                ("raExFidLo", f"{min(fids):.2f}"), ("raExFidHi", f"{max(fids):.2f}"),
                ("raExEtoeLo", f"{min(e2es):.2f}"), ("raExEtoeHi", f"{max(e2es):.2f}"),
                ("raExNmodels", str(len(body))),
                ("raExNwin", str(sum(1 for g in gaps if g > 0))),
                ("raExNsig", str(sum(1 for k, v in out
                                     if k.endswith("Gap") and "*" in v)))]
        # which fields extraction actually gets wrong, most common first
        rank = sorted(tot.items(), key=lambda kv: -kv[1])
        for i, (f, n) in enumerate(rank[:3], 1):
            out += [(f"raExErr{'One' if i == 1 else 'Two' if i == 2 else 'Three'}",
                     FIELD_LABEL.get(f, f)),
                    (f"raExErr{'One' if i == 1 else 'Two' if i == 2 else 'Three'}N",
                     str(n))]
        print("extraction error fields:", rank)

    MACROS.write_text("% generated by tools/fill_ra_extract.py -- do not edit\n"
                      + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in out))
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(
        "% generated by tools/fill_ra_extract.py -- do not edit\n"
        "\\begin{tabular}{lcccccc}\n    \\toprule\n"
        "    & \\multicolumn{2}{c}{Components} & \\multicolumn{4}{c}{End to end}\\\\\n"
        "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-7}\n"
        "    Model & Program & Extraction & Extract & Read & Gap & Cost \\\\\n"
        "     & on true input & exact & $\\rightarrow$ run & $+$ reason & & ratio \\\\\n"
        "    \\midrule\n" + "\n".join(body) + "\n    \\bottomrule\n\\end{tabular}\n")
    print(f"wrote {MACROS.name} ({len(out)} macros) and tables/{TABLE.name} "
          f"({len(body)} rows)")


if __name__ == "__main__":
    main()
