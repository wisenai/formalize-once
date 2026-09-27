"""Separate "answered wrong" from "never answered" in the plain branch.

A plain call that exhausts its completion ceiling while reasoning returns no text.
_plain_num() yields None and grading counts it WRONG -- indistinguishable from a
wrong dollar amount. At a 16,000-token ceiling that is a harness artifact (pilot 2
showed the same case answers in ~5k tokens once the ceiling is lifted). At 60,000,
four times the formalize budget, it is a real failure of per-case reasoning. Either
way it must be reported, not folded silently into the denominator.

The cache holds the raw response per case and the runner preserves case order, so
this is recoverable after the fact with no re-spend.

Run:  python tools/noanswer_report.py
"""
import json
import math
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "tools")
from preflight import key_path                                      # noqa: E402
from symbol_scramble import paths                                   # noqa: E402
from symbol_scramble.rulearena.core_experiment import DOMAINS       # noqa: E402

RES = paths.RESULTS / "rulearena"
CELLS = [("gemini35_flash", 1, None, "airline_L1_core.jsonl"),
         ("claude_opus48", 2, 60000, "airline_L2_core.jsonl"),
         ("claude_sonnet5", 1, 60000, "airline_L1_core.jsonl"),
         ("claude_sonnet5", 2, 60000, "airline_L2_core.jsonl"),
         ("claude_opus48", 1, 60000, "airline_L1_core.jsonl"),
         ("claude_opus48", 2, 60000, "airline_L2_core.jsonl"),
         ("gemini35_flash", 1, None, "airline_L1_core.jsonl"),
         ("claude_sonnet5", 0, 60000, "airline_ceil60k_core.jsonl"),
         ("claude_opus48", 0, 60000, "airline_ceil60k_core.jsonl")]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


def row_for(fname, mid, level):
    fp = RES / fname
    if not fp.exists():
        return None
    best = None
    for line in fp.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if (r.get("model") == mid and r.get("n") == 90
                    and r.get("level", 0) == level and r.get("plain_sc", 1) == 1):
                best = r
    return best


def no_answer_flags(mid, level, ceiling, domain="airline", plain_budget="3000"):
    dom = DOMAINS[domain]
    probs = dom.loader(level, limit=95)
    worked, test = probs[:5], probs[5:95]
    rules, cap = dom.rules(), (ceiling or max(16000, int(plain_budget) + 4000))
    shots = ("Worked examples (info -> correct answer):\n" + "\n".join(
        f"info={p.entities!r}  ->  {dom.fmt_answer(p.gt_answer)}" for p in worked) + "\n\n")
    flags = []
    for v in test:
        fp = key_path(mid, f"You compute {dom.noun} by applying the given rules.",
                      f"Rules:\n{rules}\n\n{shots}Case (structured): "
                      f"{json.dumps(v.entities)}\n\nCompute exactly {dom.noun}. Reason "
                      f"step by step, then last line exactly: FINAL ANSWER: <number>",
                      ("plain", json.dumps(v.entities), 0),
                      max_tokens=cap, temperature=0, reasoning=plain_budget)
        if not fp.exists():
            flags.append(None)
            continue
        d = json.loads(fp.read_text())
        flags.append(bool(d.get("no_answer") or not d.get("text", "").strip()))
    return flags


if __name__ == "__main__":
    print(f"{'cell':26s} {'no-ans':>7s} {'err':>6s} {'plain(all)':>11s} "
          f"{'plain(answered)':>16s} {'gap(answered)':>14s} {'p':>8s}")
    for mid, lv, ceil, fname in CELLS:
        r = row_for(fname, mid, lv)
        if not r:
            print(f"{mid + ' L' + str(lv):26s} {'-- not run yet --':>7s}")
            continue
        flags = no_answer_flags(mid, lv, ceil)
        f_, p_ = r["per_case_form"], r["per_case_plain"]
        err = r.get("per_case_plain_err", [0] * len(p_))
        na = sum(1 for x in flags if x)
        nerr = sum(err)
        keep = [i for i in range(len(p_)) if not flags[i] and not err[i]]
        pa_all = sum(p_) / len(p_)
        pa_ans = (sum(p_[i] for i in keep) / len(keep)) if keep else float("nan")
        b = sum(1 for i in keep if f_[i] and not p_[i])
        c = sum(1 for i in keep if p_[i] and not f_[i])
        gap = (b - c) / len(keep) if keep else float("nan")
        warn = "   <-- INCOMPLETE" if nerr else ""
        print(f"{mid + ' L' + str(lv):26s} {na:7d} {nerr:6d} {pa_all:11.3f} "
              f"{pa_ans:16.3f} {gap:+14.3f} {mcnemar(b, c):8.4f}{warn}")
