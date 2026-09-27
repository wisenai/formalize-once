"""Fresh-input cells against the public ones they mirror, with paired tests.

The question is whether the formalize-once edge measured on RuleArena's public
inputs survives on inputs no model can have memorized. That is only readable if the
generated cases match the public ones in difficulty, which tools/check_fresh_match.py
verifies, and if every cell actually ran, which is checked here -- an abandoned cell
reports as a confident zero.

Run:  .venv/bin/python tools/fresh_report.py
"""
import json
import math
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "tools")

from ra_cells import rows                                           # noqa: E402

SRC = "data/results/rulearena/airline_fresh_core.jsonl"
NAME = {"claude_opus48": "Opus 4.8", "claude_sonnet5": "Sonnet 5", "gpt55": "GPT-5.5",
        "gemini35_flash": "Flash 3.5", "gemini31_pro": "Pro 3.1"}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n * 2)


def tango(b, c, n, z=1.96):
    def score(d):
        A, B = 2 * n, -b - c + (2 * n - b + c) * d
        Cc = -c * d * (1 - d)
        q = (-B + math.sqrt(max(B * B - 4 * A * Cc, 0.0))) / (2 * A)
        var = n * (2 * q + d * (1 - d))
        return (b - c - n * d) / math.sqrt(var) if var > 0 else math.inf

    def solve(lo, hi, t):
        for _ in range(200):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if score(m) - t > 0 else (lo, m)
        return (lo + hi) / 2

    pt = (b - c) / n
    return pt, solve(-0.999, pt, z), solve(pt, 0.999, -z)


def paired(r):
    f, p = r["per_case_form"], r["per_case_plain"]
    err = r.get("per_case_plain_err", [0] * len(p))
    keep = [i for i in range(len(p)) if not err[i]]
    b = sum(1 for i in keep if f[i] and not p[i])
    c = sum(1 for i in keep if p[i] and not f[i])
    return b, c, len(keep)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main():
    fresh = {}
    for line in open(SRC):
        if line.strip():
            r = json.loads(line)
            if r.get("n") == 90 and r.get("code_extracted"):
                fresh[r["model"]] = r
    pub = rows("airline", 0)

    incomplete = [m for m, r in fresh.items() if sum(r.get("per_case_plain_err", []))]
    if incomplete:
        print("REFUSING to report -- these cells never finished:", incomplete)
        return 1

    print(f"{'model':10s} | {'PUBLIC (memorizable)':^26s} | "
          f"{'FRESH (generated)':^26s}")
    print(f"{'':10s} | {'form':>5s}{'plain':>6s}{'gap':>7s}{'p':>8s} | "
          f"{'form':>5s}{'plain':>6s}{'gap':>7s}{'p':>8s}")
    print("-" * 70)
    n_pub_pos = n_fresh_pos = n_fresh_sig = 0
    for mid in NAME:
        if mid not in fresh:
            print(f"{NAME[mid]:10s} |  -- fresh cell missing --")
            continue
        pr, fr = pub[mid], fresh[mid]
        pb, pc, pn = paired(pr)
        fb, fc, fn = paired(fr)
        pg, _, _ = tango(pb, pc, pn)
        fg, lo, hi = tango(fb, fc, fn)
        pp, fp = mcnemar(pb, pc), mcnemar(fb, fc)
        n_pub_pos += pg > 0
        n_fresh_pos += fg > 0
        n_fresh_sig += fp < 0.05
        print(f"{NAME[mid]:10s} | {pr['formalize_once_acc']:5.2f}"
              f"{pr['plain_acc']:6.2f}{pg:+7.2f}{pp:8.3f}{stars(pp):3s}| "
              f"{fr['formalize_once_acc']:5.2f}{fr['plain_acc']:6.2f}"
              f"{fg:+7.2f}{fp:8.3f}{stars(fp)}  [{lo:+.2f},{hi:+.2f}]")
    print("-" * 70)
    print(f"gap > 0:  public {n_pub_pos}/5   fresh {n_fresh_pos}/5"
          f"   (fresh significant: {n_fresh_sig}/5)")
    ff = [fresh[m]["formalize_once_acc"] for m in NAME if m in fresh]
    pf = [pub[m]["formalize_once_acc"] for m in NAME]
    print(f"formalization fidelity: public {min(pf):.2f}-{max(pf):.2f}   "
          f"fresh {min(ff):.2f}-{max(ff):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
