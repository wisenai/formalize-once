"""Does the fresh generator reproduce the public set's difficulty?

The contamination control is only interpretable if the generated cases differ from
the public ones in identity and not in difficulty. This compares the marginals that
drive how hard a case is -- threshold proximity above all -- and checks that no
generated case is a public one.

Run:  .venv/bin/python tools/check_fresh_match.py
"""
import statistics as st
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "tools")

from gen_fresh import fresh_cases                                   # noqa: E402
from symbol_scramble.rulearena.core_experiment import DOMAINS       # noqa: E402

SIZE_EDGE, WT_EDGES = 62, (50, 70, 100)


def stats(cases):
    bags = [b for v in cases for b in v.entities["bag_list"][1:]]
    li = [sum(b["size"]) for b in bags]
    wt = [b["weight"] for b in bags]
    near = sum(1 for b in bags
               if abs(sum(b["size"]) - SIZE_EDGE) <= 3
               or any(abs(b["weight"] - e) <= 3 for e in WT_EDGES))
    return {
        "linear_in_median": st.median(li),
        "linear_in_p90": sorted(li)[int(0.9 * len(li))],
        "weight_median": st.median(wt),
        "base_price_median": st.median([v.entities["base_price"] for v in cases]),
        "domestic_share": sum(v.entities["routine"] == "U.S." for v in cases) / len(cases),
        "near_threshold": near / len(bags),
        "fee_median": st.median([v.gt_answer - v.entities["base_price"] for v in cases]),
    }


def main():
    pub = DOMAINS["airline"].loader(0, limit=95)[5:95]
    new = fresh_cases("airline", 95)[5:95]
    old = fresh_cases("airline", 95, matched=False)[5:95]
    a, b, c = stats(pub), stats(new), stats(old)
    print(f"{'feature':20s} {'PUBLIC':>9s} {'MATCHED':>9s} {'ratio':>7s} "
          f"{'| SUPERSEDED':>13s} {'ratio':>7s}")
    worst = 0.0
    for k in a:
        r = b[k] / a[k] if a[k] else float("nan")
        r2 = c[k] / a[k] if a[k] else float("nan")
        worst = max(worst, abs(r - 1))
        print(f"{k:20s} {a[k]:9.2f} {b[k]:9.2f} {r:7.2f} {c[k]:13.2f} {r2:7.2f}")
    pubset = {repr(v.entities) for v in DOMAINS["airline"].loader(0, limit=100)}
    dup = sum(1 for v in new if repr(v.entities) in pubset)
    print(f"\ngenerated cases identical to a public one: {dup}/90")
    print(f"worst marginal deviation from public: {worst:.0%}")
    return 0 if (dup == 0 and worst < 0.20) else 1


if __name__ == "__main__":
    sys.exit(main())
