"""Raise the formalize reasoning budget until it does not bind, and check it.

The plain branch already has this rule: report every cell at a completion ceiling that
does not bind it, because a ceiling that truncates one model and not another measures
the ceiling. The formalize branch needs the same rule and did not have it.

Under our nominal reasoning="8000", the actual effort models spend on the identical
formalize call differs by an order of magnitude -- Gemini 3.5 Flash 40-60k completion
tokens, GPT-5.5 11-16k, Gemini 3.1 Pro 6-16k. Pro's comp_1 fidelity looked like a
capability ceiling at 0.47; at reasoning="24000" it writes 0.79-0.89 programs, beside
Sonnet's 0.89 and Flash's 0.90. It was under-reasoning, not unable.

The fix has to be a rule rather than a knob turned until one model improves, so this
raises the budget for a model that was ALREADY good (GPT-5.5 at 0.87) as well as the
one that was not. If the raise only ever helps the model it was introduced for, it is
tuning; if it leaves an already-good model where it was, it is a ceiling correction.

Formalize only -- the plain branch is untouched, and program execution is local.

Run:  .venv/bin/python tools/probe_form_budget.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from taxl1_stratified import accuracy, pick_stratified                # noqa: E402
import symbol_scramble.rulearena.core_experiment as C                 # noqa: E402
from symbol_scramble.llm.client import make_client                    # noqa: E402

OUT = ROOT / "data" / "results" / "rulearena" / "tax_L1_formbudget.jsonl"
BUDGET = "24000"
PLAN = [("gemini31_pro", 5), ("gpt55", 3)]   # the gated one, and a control already high


def main():
    dom = C.DOMAINS["tax"]
    probs = dom.loader(1, limit=500)
    test = probs[5:95]
    worked = pick_stratified(probs[0:5] + probs[95:])
    ex = "\n".join(f"info={p.entities!r}  ->  {dom.fmt_answer(p.gt_answer)}"
                   for p in worked)
    user = (f"Ruleset:\n{dom.rules()}\n\n{dom.schema}\n\nWorked examples "
            f"(your program MUST reproduce all of these):\n{ex}\n\nWrite compute(info) "
            f"implementing the ENTIRE ruleset, returning the numeric answer. stdlib "
            f"only. Return ONLY a ```python code block.")
    sysmsg = ("You formalize a fixed ruleset into one correct reusable Python "
              "program.")

    for mid, n in PLAN:
        model = make_client(mid)
        best, best_score, accs, toks = None, -1, [], []
        worked_scores = []
        for ci in range(n):
            r = C._cached_complete(model, mid, sysmsg, user,
                                   ("formalize", BUDGET, ci),
                                   max_tokens=60000, temperature=0.4,
                                   reasoning=BUDGET)
            code = C.extract_code(r.text)
            toks.append(r.completion_tokens)
            accs.append(round(sum(accuracy(code, test, dom)) / len(test), 3)
                        if code else None)
            sc = C._worked_score(code, worked, dom)   # the protocol's own selector
            worked_scores.append(sc)
            if sc > best_score:
                best, best_score = code, sc
        sel = round(sum(accuracy(best, test, dom)) / len(test), 3) if best else None
        print(f"{mid:16s} budget={BUDGET} selected={sel} candidates={accs} "
              f"tokens={toks}", flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps({"model": mid, "domain": "tax", "level": 1,
                                "worked": "stratified", "reasoning": BUDGET,
                                "n": len(test), "formalize_once_acc": sel,
                                "candidates": [[w, a] for w, a in
                                               zip(worked_scores, accs)],
                                "cand_accs": accs, "completion_tokens": toks,
                                "per_case_form": [int(x) for x in
                                                  accuracy(best, test, dom)]}) + "\n")


if __name__ == "__main__":
    main()
