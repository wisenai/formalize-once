"""Price a planned RuleArena run BEFORE spending anything.

Rebuilds every prompt exactly as core_experiment.run() would, hashes it with the
same key recipe, and checks the call cache on disk. Cache hits are free on a real
run; only the misses cost money. Estimated cost per miss comes from the token
spend actually observed in the published cells (paper/ra_macros.tex), so it is an
empirical estimate, not a price list.

Touches no network and constructs no client.

Run:  python tools/preflight.py
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from symbol_scramble import paths                                   # noqa: E402
from symbol_scramble.rulearena.core_experiment import DOMAINS       # noqa: E402

CALLCACHE = paths.RESULTS / "rulearena" / "callcache"
ROOT = Path(__file__).resolve().parent.parent

MAC = {}
for _line in (ROOT / "paper" / "ra_macros.tex").read_text().splitlines():
    _m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", _line.strip())
    if _m:
        MAC[_m.group(1)] = _m.group(2)

# model id -> macro stem used in the paper's cost macros
STEM = {"claude_opus48": "Opus", "claude_sonnet5": "Sonnet", "gpt55": "GPT",
        "gemini35_flash": "Gemini", "gemini31_pro": "GeminiPro"}


def observed_unit_costs(mid, domain):
    """($ per formalize candidate, $ per plain case) from the published L0 cells."""
    stem = STEM.get(mid)
    pre = "ra" if domain == "airline" else "raTax"
    try:
        form = float(MAC[f"{pre}{stem}CostForm"]) / 3.0
        plain = float(MAC[f"{pre}{stem}CostPlain"]) / 90.0
    except KeyError:
        return None, None
    return form, plain


def key_path(mid, system, user, sample_key, **kw):
    key = json.dumps({"mid": mid, "sys": system, "user": user, "kw": kw,
                      "s": list(sample_key)}, sort_keys=True)
    return CALLCACHE / (hashlib.sha256(key.encode()).hexdigest()[:32] + ".json")


def already_done(domain, out_tag, level, n_test, plain_fewshot, plain_budget, plain_sc):
    """Models run() will skip outright -- its row-level resume, which matters more
    than the call cache: a completed cell is never re-bought."""
    fp = paths.RESULTS / "rulearena" / f"{domain}{out_tag}_core.jsonl"
    done = set()
    if fp.exists():
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("level", 0) == level and r.get("n") == n_test
                    and r.get("plain_fewshot") == plain_fewshot
                    and r.get("plain_budget") == plain_budget
                    and r.get("plain_sc", 1) == plain_sc
                    and r.get("code_extracted")):
                done.add(r["model"])
    return done


def preflight(models, domain="airline", level=0, n_test=90, n_worked=5,
              n_candidates=3, plain_fewshot=True, plain_budget="3000", plain_sc=1,
              out_tag="", plain_ceiling=None):
    dom = DOMAINS[domain]
    done = already_done(domain, out_tag, level, n_test, plain_fewshot,
                        plain_budget, plain_sc)
    probs = dom.loader(level, limit=n_test + n_worked)
    worked, test = probs[:n_worked], probs[n_worked:n_worked + n_test]
    rules = dom.rules()
    shots = ("Worked examples (info -> correct answer):\n"
             + "\n".join(f"info={p.entities!r}  ->  {dom.fmt_answer(p.gt_answer)}"
                         for p in worked) + "\n\n") if plain_fewshot else ""
    ex = "\n".join(f"info={p.entities!r}  ->  {dom.fmt_answer(p.gt_answer)}" for p in worked)
    fuser = (f"Ruleset:\n{dom.rules()}\n\n{dom.schema}\n\nWorked examples "
             f"(your program MUST reproduce all of these):\n{ex}\n\nWrite compute(info) "
             f"implementing the ENTIRE ruleset, returning the numeric answer. stdlib only. "
             f"Return ONLY a ```python code block.")
    rows, tot_miss_cost = [], 0.0
    for mid in models:
        if mid in done:
            rows.append((mid, None, None, None, None, 0.0))
            continue
        fhit = sum(key_path(mid, "You formalize a fixed ruleset into one correct "
                            "reusable Python program.", fuser, ("formalize", ci),
                            max_tokens=60000, temperature=0.4,
                            reasoning="8000").exists() for ci in range(n_candidates))
        phit = 0
        for v in test:
            sysm = f"You compute {dom.noun} by applying the given rules."
            usr = (f"Rules:\n{rules}\n\n{shots}Case (structured): "
                   f"{json.dumps(v.entities)}\n\nCompute exactly {dom.noun}. Reason "
                   f"step by step, then last line exactly: FINAL ANSWER: <number>")
            for si in range(plain_sc):
                if key_path(mid, sysm, usr, ("plain", json.dumps(v.entities), si),
                            max_tokens=(plain_ceiling if plain_ceiling
                                        else max(16000, int(plain_budget) + 4000)),
                            temperature=(0.7 if plain_sc > 1 else 0),
                            reasoning=plain_budget).exists():
                    phit += 1
        fmiss, pmiss = n_candidates - fhit, n_test * plain_sc - phit
        cf, cp = observed_unit_costs(mid, domain)
        est = (fmiss * cf + pmiss * cp) if cf else float("nan")
        tot_miss_cost += 0.0 if cf is None else est
        rows.append((mid, fhit, fmiss, phit, pmiss, est))
    print(f"\n=== {domain} L{level}  n_test={n_test} sc={plain_sc} budget={plain_budget}")
    print(f"{'model':16s} {'formalize':>16s} {'plain':>18s} {'est. $ to run':>14s}")
    for mid, fh, fm, ph, pm, est in rows:
        if fh is None:
            print(f"{mid:16s} {'cell already recorded -- run() skips it':>36s} "
                  f"{0.0:13.2f}")
        else:
            print(f"{mid:16s} {fh:3d} hit /{fm:3d} miss {ph:6d} hit /{pm:4d} miss "
                  f"{est:13.2f}")
    print(f"{'TOTAL':16s} {'':>16s} {'':>18s} {tot_miss_cost:13.2f}")
    return tot_miss_cost


if __name__ == "__main__":
    grand = 0.0
    # Sonnet and Opus truncate at the default ceiling, so they run with it raised.
    for lv in (1, 2):
        grand += preflight(["claude_opus48", "claude_sonnet5"], "airline", level=lv,
                           out_tag=f"_L{lv}", plain_ceiling=60000)
    # Flash never approaches its ceiling; keep the default so its cached calls count.
    grand += preflight(["gemini35_flash"], "airline", level=1, out_tag="_L1")
    print(f"\nGRAND TOTAL of misses: ${grand:.2f}")
