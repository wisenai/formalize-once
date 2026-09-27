"""One definition of "which rows count, and which cases within them".

Four generators (macros, stats, CIs, sweep) each had their own copy of this
logic and drifted apart, which is how a cell corrected in one table stayed
stale in another. They all call this now.

Two rules, both learned the hard way:

1. A cell re-run with the plain completion ceiling raised supersedes the
   original at airline L0. At the default 16,000 ceiling Sonnet spent the whole
   ceiling reasoning on many cases and returned no answer, which grading scored
   as wrong. That measures the cap, not the model. Cells that never approached
   the ceiling are unaffected either way.

2. A case that never produced an answer is excluded from the paired test rather
   than counted against per-case reasoning. Two ways that happens: the call
   errored and never executed (per_case_plain_err), or the model exhausted its
   completion ceiling while reasoning and returned nothing. Counting either as a
   wrong answer credits formalize-once for a harness outcome -- the defect that
   invalidated the withdrawn fresh-input run.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "results" / "rulearena"

# cells whose plain branch ran with the ceiling raised (see tools/run_sweep2.py)
RAISED = {("claude_sonnet5", "airline", 0): 60000,
          ("claude_opus48", "airline", 0): 60000,
          ("claude_sonnet5", "airline", 1): 60000,
          ("claude_sonnet5", "airline", 2): 60000,
          ("claude_opus48", "airline", 1): 60000,
          ("claude_opus48", "airline", 2): 60000}


def files_for(domain, level):
    base = {0: f"{domain}_core.jsonl", 1: f"{domain}_L1_core.jsonl",
            2: f"{domain}_L2_core.jsonl"}[level]
    out = [base]
    if domain == "airline" and level == 0:
        out.append("airline_ceil60k_core.jsonl")     # supersedes
    return out


def rows(domain="airline", level=0, n=None):
    """Usable rows per model: single-pass plain, program parsed, cell complete."""
    best = {}
    for fname in files_for(domain, level):
        fp = RES / fname
        if not fp.exists():
            continue
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("level", 0) != level or not r.get("code_extracted"):
                continue
            if r.get("plain_sc") not in (None, 1):
                continue
            if sum(r.get("per_case_plain_err", [])):
                continue                              # incomplete: cases never ran
            if n is not None and r.get("n") != n:
                continue
            if r["model"] not in best or r["n"] >= best[r["model"]]["n"]:
                best[r["model"]] = r
    return best


def paired(row, domain="airline", level=0):
    """(b, c, n_used): discordant counts over cases that actually produced an answer."""
    # Prefer the flags stored in the record. They are written once by
    # tools/backfill_noanswer.py so that reading a result does not require replaying
    # the call cache; without them a cell silently scores its truncated cases as wrong
    # answers, which changes significance. Fall back to the cache for older records.
    mid = row["model"]
    stored = row.get("per_case_no_answer")
    if stored is not None:
        na = [bool(x) for x in stored]
    else:
        from noanswer_report import no_answer_flags
        na = no_answer_flags(mid, level, RAISED.get((mid, domain, level)),
                             domain=domain)
    err = row.get("per_case_plain_err", [0] * len(row["per_case_plain"]))
    f, p = row["per_case_form"], row["per_case_plain"]
    keep = [i for i in range(len(p))
            if not (i < len(na) and na[i]) and not err[i]
            and not isinstance(f[i], str) and not isinstance(p[i], str)]
    b = sum(1 for i in keep if f[i] and not p[i])
    c = sum(1 for i in keep if p[i] and not f[i])
    return b, c, len(keep)
