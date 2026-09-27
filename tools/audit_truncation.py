"""Did any reported plain call get truncated instead of answered?

A call that exhausts its completion ceiling while reasoning emits no FINAL ANSWER
and is graded wrong. That is a harness artifact, not a model outcome, and it is
what invalidated the withdrawn fresh-input run. Every cell that reaches a table
should be audited for it.

Reads the on-disk call cache only -- no network, no spend.

Run:  python tools/audit_truncation.py
"""
import json
import re
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "tools")
from preflight import key_path                                      # noqa: E402
from symbol_scramble.rulearena.core_experiment import DOMAINS       # noqa: E402

# (model, level, plain_ceiling) -- ceiling must match what the cell was run with,
# because it is part of the cache key.
CELLS = [
    ("gpt55", 1, None), ("gemini31_pro", 1, None), ("gemini35_flash", 1, None),
    ("gpt55", 2, None), ("gemini31_pro", 2, None), ("gemini35_flash", 2, None),
    ("claude_sonnet5", 1, 60000), ("claude_sonnet5", 2, 60000),
    ("claude_opus48", 1, 60000), ("claude_opus48", 2, 60000),
]


def audit(mid, level, ceiling, domain="airline", n_test=90, n_worked=5,
          plain_budget="3000"):
    dom = DOMAINS[domain]
    probs = dom.loader(level, limit=n_test + n_worked)
    worked, test = probs[:n_worked], probs[n_worked:n_worked + n_test]
    rules = dom.rules()
    shots = ("Worked examples (info -> correct answer):\n" + "\n".join(
        f"info={p.entities!r}  ->  {dom.fmt_answer(p.gt_answer)}" for p in worked) + "\n\n")
    cap = ceiling if ceiling else max(16000, int(plain_budget) + 4000)
    found = empty = atcap = nofinal = 0
    for v in test:
        fp = key_path(mid, f"You compute {dom.noun} by applying the given rules.",
                      f"Rules:\n{rules}\n\n{shots}Case (structured): "
                      f"{json.dumps(v.entities)}\n\nCompute exactly {dom.noun}. Reason "
                      f"step by step, then last line exactly: FINAL ANSWER: <number>",
                      ("plain", json.dumps(v.entities), 0),
                      max_tokens=cap, temperature=0, reasoning=plain_budget)
        if not fp.exists():
            continue
        found += 1
        d = json.loads(fp.read_text())
        t = d.get("text", "")
        if d.get("no_answer") or not t.strip():
            empty += 1
        if d.get("ct", 0) >= cap - 100:
            atcap += 1
        if t.strip() and not re.search(r"FINAL ANSWER:", t):
            nofinal += 1
    return found, empty, atcap, nofinal, cap


if __name__ == "__main__":
    print(f"{'cell':34s} {'cap':>6s} {'found':>6s} {'empty':>6s} {'atcap':>6s} {'noFINAL':>8s}")
    bad = 0
    for mid, lv, ceil in CELLS:
        f, e, a, nf, cap = audit(mid, lv, ceil)
        flag = (f"  <-- {e} no-answer(s): exclude from the paired test and report "
                f"the count") if (e or a or nf) else ""
        bad += bool(e or a or nf)
        print(f"{mid + ' L' + str(lv):34s} {cap:6d} {f:6d} {e:6d} {a:6d} {nf:8d}{flag}")
    print("\nNo truncation anywhere." if not bad else
          f"\n{bad} cell(s) contain no-answers. Reportable only if excluded from the\n"
          f"paired test and disclosed -- tools/noanswer_report.py does the former,\n"
          f"and fill_ra_sweep.py emits a NoAns count per cell for the latter.")
