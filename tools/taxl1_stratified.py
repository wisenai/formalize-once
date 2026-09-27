"""Is formalize-once's low comp_1 fidelity a property of the strategy, or of our
worked-example draw?

The reported comp_1 cells take the first five records as worked examples. That draw
happens to contain four self-employed returns, one basic, one with children, and
ZERO with education credits -- while 32 of the 90 test cases have them. The programs
score 0.22 on exactly those cases and 0.67-0.85 elsewhere, so most of the fidelity
deficit sits in a branch the model was shown no example of and the ruleset text
mentions once.

Per-case reasoning does not care: it meets each return on its own and recalls the
credit. Formalize-once has to cover every branch up front. That asymmetry is real, but
how much of the measured gap is it, versus an unlucky sample?

This tests it directly and cheaply. The test set is untouched (records 5-94). Only the
worked examples change: five drawn from records 0-4 and 95-99 so that all four comp_1
categories are represented. New programs are drawn with those examples and executed on
the SAME 90 cases. Execution is free; only the formalize calls cost anything.

If fidelity jumps, the reported number understates formalize-once and we say so. If it
does not, the ceiling is real.

Run:  .venv/bin/python tools/taxl1_stratified.py
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from symbol_scramble.llm.client import make_client                  # noqa: E402
from symbol_scramble.rulearena.core_experiment import (              # noqa: E402
    DOMAINS, _wrap, formalize_once)
from symbol_scramble.sandbox import run_solve                       # noqa: E402

OUT = ROOT / "data" / "results" / "rulearena" / "tax_L1_stratified.jsonl"
MODELS = ["gpt55", "claude_sonnet5"]


def feats(e):
    f = set()
    if (e.get("business_income") not in (None, 0, "")
            or e.get("self_employment_deductible") not in (None, 0, "")):
        f.add("self-employed")
    if e.get("num_qualifying_children"):
        f.add("children")
    if (e.get("education_credits") not in (None, 0, "")
            or e.get("american_opportunity_credit") not in (None, 0, "")):
        f.add("education")
    return f or {"basic"}


def pick_stratified(pool, k=5):
    """Greedy: repeatedly take the record adding the most uncovered categories."""
    chosen, covered = [], set()
    while len(chosen) < k and pool:
        best = max(pool, key=lambda p: (len(feats(p.entities) - covered),
                                        -len(feats(p.entities))))
        chosen.append(best)
        covered |= feats(best.entities)
        pool = [p for p in pool if p is not best]
    return chosen


def accuracy(code, test, dom):
    hits = [bool(code) and (lambda sb: sb.ok and dom.correct(sb.value, v.gt_answer))(
        run_solve(_wrap(code, v.entities), timeout=8)) for v in test]
    return hits


def main():
    dom = DOMAINS["tax"]
    probs = dom.loader(1, limit=500)
    test = probs[5:95]                            # unchanged from the reported cells
    pool = probs[0:5] + probs[95:]                # never overlaps the test set
    worked = pick_stratified(pool)
    orig = probs[:5]
    print("worked examples, original :",
          dict(collections.Counter(x for p in orig for x in feats(p.entities))))
    print("worked examples, stratified:",
          dict(collections.Counter(x for p in worked for x in feats(p.entities))))
    print(f"test set unchanged: {len(test)} cases, "
          f"{sum(1 for v in test if 'education' in feats(v.entities))} with education "
          f"credits\n")

    reported = {}
    for line in (ROOT / "data/results/rulearena/tax_L1_core.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("n") == 90 and not sum(r.get("per_case_plain_err", [])):
                reported[r["model"]] = r

    for mid in MODELS:
        code, pt, ct, cands = formalize_once(make_client(mid), mid, dom, worked,
                                             n_candidates=5, full_sweep=True)
        hits = accuracy(code, test, dom)
        acc = sum(hits) / len(test)
        cand_accs = [round(sum(accuracy(c, test, dom)) / len(test), 3)
                     for _, c, _, _ in cands]
        old = reported.get(mid, {}).get("formalize_once_acc")
        print(f"{mid:16s} stratified fidelity {acc:.2f}   reported {old:.2f}   "
              f"delta {acc - old:+.2f}")
        print(f"{'':16s} candidates {cand_accs}")
        byf = collections.defaultdict(lambda: [0, 0])
        for i, v in enumerate(test):
            for f in feats(v.entities):
                byf[f][0] += hits[i]
                byf[f][1] += 1
        print(f"{'':16s} by branch: "
              + "  ".join(f"{f} {a}/{n}" for f, (a, n) in sorted(byf.items())))
        with open(OUT, "a") as fh:
            fh.write(json.dumps({
                "model": mid, "domain": "tax", "level": 1, "n": len(test),
                "worked": "stratified", "formalize_once_acc": round(acc, 3),
                "reported_acc": old, "candidates": cand_accs,
                "per_case_form": [int(x) for x in hits],
                "form_tokens": [pt, ct]}) + "\n")


if __name__ == "__main__":
    main()
