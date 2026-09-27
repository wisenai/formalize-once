"""Does formalize-once survive contact with unstructured input?

The main study hands both branches a structured `info` dict. A deployment does not
get one; it gets a message. That gap is the sharpest objection to the paper, because
the program cannot read prose -- something has to recover the dict first, and if that
step is unreliable it can eat the whole accuracy edge.

So we put the same 90 airline cases behind a narrator (tools/ra_narrate.py) and run
two end-to-end pipelines on the prose, one call per case each:

  extract -> run      one extraction call recovers the dict, then the SAME program
                      the model already wrote (replayed from cache, free) executes
  read -> reason      per-case chain-of-thought straight off the prose

Both are graded by the oracle against the label of the original structured case, so
the comparison is paired and the ceiling is known: the program's accuracy on the TRUE
dict is measured too, which splits an end-to-end miss into an extraction error and a
program error instead of leaving them fused.

Every call goes through the shared content-addressed cache, and the narrator is
deterministic, so a resume replays for free and re-running is exactly reproducible.
The plain ceiling is set to 60,000 rather than the default 16,000: at the default a
model can exhaust the ceiling while reasoning and return nothing, which grading would
score as a wrong answer. That defect invalidated an earlier cell of this paper; it is
checked for here, not assumed away.

Run:  .venv/bin/python tools/run_extract.py --pilot     # 3 cases, 2 models
      .venv/bin/python tools/run_extract.py             # full: 90 cases, 5 models
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "src")
sys.path.insert(0, "tools")

from ra_narrate import narrate                                      # noqa: E402
from symbol_scramble import paths                                   # noqa: E402
from symbol_scramble.rulearena.core_experiment import (              # noqa: E402
    DOMAINS, _cached_complete, _plain_num, _wrap, formalize_once)
from symbol_scramble.llm.client import make_client                  # noqa: E402
from symbol_scramble.rulearena import oracle                        # noqa: E402
from symbol_scramble.sandbox import run_solve                       # noqa: E402

OUT = paths.RESULTS / "rulearena" / "airline_extract.jsonl"
SEED = 1
MODELS = ["claude_opus48", "claude_sonnet5", "gpt55", "gemini35_flash", "gemini31_pro"]
ROUTINES = ["U.S.", "Puerto Rico", "Canada", "Mexico", "Europe", "China", "Japan",
            "India", "South Korea", "Australia", "New Zealand", "Colombia", "Peru",
            "Ecuador", "Panama", "Cuba", "Israel", "Qatar", "South America"]

CLASSES = ["Basic Economy", "Main Cabin", "Main Plus", "Premium Economy",
           "Business", "First"]
EXTRACT_SYS = ("You convert a traveler's message into the structured record a booking "
               "system stores. You output JSON only.")
EXTRACT_USER = """Message from the traveler:
\"\"\"{text}\"\"\"

Fill in the booking record. Return a JSON object whose TOP-LEVEL keys are exactly
these five and nothing else:

{{
  "base_price": <int, the fare in dollars>,
  "customer_class": <one of: {classes}>,
  "routine": <one of: {routines}>,
  "direction": <0 if the trip departs the U.S., 1 if it arrives in the U.S.>,
  "bag_list": [ {{"name": <str>, "size": [<length>, <width>, <height>], "weight": <int>}} ]
}}

- routine is the non-U.S. endpoint of the trip as a region label, or "U.S." if both
  endpoints are in the U.S.
- bag_list[0] is the bag kept in the cabin; every later entry is a checked bag, in
  the order the traveler lists them.
- size is in inches, weight in pounds, both integers.

Do not nest the record under another key. Do not compute any fee or total. Return
ONLY the JSON object, no commentary."""


def parse_json(t):
    """The extracted record, or None. Models fence JSON, prefix it, or emit it bare."""
    if not t:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    body = m.group(1) if m else t
    start = body.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(body)):
        ch = body[i]
        if in_str:
            # order matters: a backslash-escaped quote must not close the string,
            # and the escape must be consumed before this char is classified
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(body[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


KEYS = ("base_price", "customer_class", "routine", "direction", "bag_list")


def unwrap(obj):
    """Recover the record from however the model packaged it.

    Models that read every field correctly still nest the result under "info" or
    "record", or hoist bag_list beside the wrapper instead of inside it. A deployment
    pins the shape with a structured-output schema, so envelope compliance is not the
    variable this experiment is about -- fact recovery is. We normalize the envelope
    and count separately how often normalization was needed, rather than scoring a
    correctly-read case as a miss.

    Returns (record, rewrapped) or (None, False)."""
    if not isinstance(obj, dict):
        return None, False
    if all(k in obj for k in KEYS):
        return obj, False
    for k, v in obj.items():                       # nested under some wrapper key
        if isinstance(v, dict):
            merged = {**v, **{kk: obj[kk] for kk in KEYS if kk in obj}}
            if all(kk in merged for kk in KEYS):
                return merged, True
    return None, False


def oracle_view(info):
    """Just the parts the reference computation reads. A differential test confirms
    bag 'name' and 'id' change no answer, so a model that renames a bag is not
    penalized for an error the ruleset cannot see."""
    if not isinstance(info, dict):
        return None
    try:
        return {
            "base_price": int(info["base_price"]),
            "customer_class": str(info["customer_class"]).strip(),
            "routine": str(info["routine"]).strip(),
            "direction": int(info["direction"]),
            "bags": [[[int(x) for x in b["size"]], int(b["weight"])]
                     for b in info["bag_list"]],
        }
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def field_diff(got, want, routine):
    """Which oracle-read fields differ. Drives the error breakdown, not the grade.

    direction is skipped on a U.S.-domestic trip. It encodes whether the journey
    departs or arrives the U.S., and when both endpoints are domestic it is not
    determined by the message and the fee rules do not read it: flipping it changes
    the oracle's answer on 0 of the 35 domestic cases. Counting it found 88 errors,
    every one domestic and none international, which says the metric was measuring an
    unanswerable field rather than the model."""
    if got is None:
        return ["unparseable"]
    keys = ["base_price", "customer_class", "routine"]
    if routine != "U.S.":
        keys.append("direction")
    bad = [k for k in keys if got.get(k) != want[k]]
    if got.get("bags") != want["bags"]:
        gb, wb = got.get("bags") or [], want["bags"]
        bad.append("bag_count" if len(gb) != len(wb) else
                   ("bag_size" if [b[0] for b in gb] != [b[0] for b in wb]
                    else "bag_weight"))
    return bad


def answer_preserving(got_raw, truth):
    """Did extraction keep everything the RULES read?

    The strict field match is the honest strict number but it is hostage to which
    fields one decides to compare. This is the same question asked without a
    hand-maintained list: run the reference computation on the extracted record and
    on the true one and see whether they agree. It isolates the extraction step from
    the program step -- the program is not involved."""
    if not isinstance(got_raw, dict):
        return False
    try:
        return oracle.airline_answer(got_raw) == oracle.airline_answer(truth)
    except Exception:                      # noqa: BLE001 - a malformed record
        return False


def run_model(mid, test, code, dom, rules, workers, ceiling):
    model = make_client(mid)
    T = len(test)
    res = [None] * T

    def one(i):
        v = test[i]
        text = narrate(v.entities, seed=SEED)
        # --- arm 1: extract, then run the program the model already wrote
        r1 = _cached_complete(
            model, mid, EXTRACT_SYS,
            EXTRACT_USER.format(text=text, classes=", ".join(f'"{c}"' for c in CLASSES),
                                routines=", ".join(f'"{r}"' for r in ROUTINES)),
            ("extract", text, 0), max_tokens=ceiling, temperature=0, reasoning="3000")
        got, rewrapped = unwrap(parse_json(r1.text))
        want = oracle_view(v.entities)
        diff = field_diff(oracle_view(got) if got else None, want,
                          v.entities["routine"])
        keeps = answer_preserving(got, v.entities)
        e2e = False
        if got is not None and code:
            sb = run_solve(_wrap(code, got), timeout=8)
            e2e = bool(sb.ok and dom.correct(sb.value, v.gt_answer))
        # --- arm 2: per-case reasoning straight off the prose
        r2 = _cached_complete(
            model, mid, f"You compute {dom.noun} by applying the given rules.",
            f"Rules:\n{rules}\n\nA traveler writes:\n\"\"\"{text}\"\"\"\n\nCompute "
            f"exactly {dom.noun}. Reason step by step, then last line exactly: "
            f"FINAL ANSWER: <number>",
            ("plaintext", text, 0), max_tokens=ceiling, temperature=0,
            reasoning="3000")
        pt_ok = dom.correct(_plain_num(r2.text), v.gt_answer)
        return i, {
            "extract_exact": not diff, "diff": diff, "e2e": e2e,
            "keeps": keeps,
            "rewrapped": rewrapped,
            "plaintext": bool(pt_ok), "pt_noanswer": not r2.text.strip(),
            "tok": [r1.prompt_tokens, r1.completion_tokens,
                    r2.prompt_tokens, r2.completion_tokens]}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, i) for i in range(T)]
        for f in as_completed(futs):
            i, d = f.result()
            res[i] = d
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ceiling", type=int, default=60000)
    a = ap.parse_args()

    dom = DOMAINS["airline"]
    rules = dom.rules()
    probs = dom.loader(0, limit=95)
    worked, test = probs[:5], probs[5:95]
    models = MODELS[:2] if a.pilot else MODELS
    if a.pilot:
        test = test[:3]

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("n") == len(test):
                    done.add(r["model"])

    lock = threading.Lock()
    for mid in models:
        if mid in done:
            print(f"{mid:16s} already done (resume) -- skipping", flush=True)
            continue
        # replayed from the cache: identical prompt to the reported Level-0 cell, so
        # this is the SAME program the main table scores, at no cost
        code, fpt, fct, _ = formalize_once(model=make_client(mid), mid=mid,
                                           dom=dom, worked=worked,
                                           n_candidates=3)
        # the program's ceiling on THIS test set, from the true dicts -- the number an
        # end-to-end miss has to be compared against
        struct = [bool(code) and (lambda sb: sb.ok and dom.correct(sb.value, v.gt_answer))(
            run_solve(_wrap(code, v.entities), timeout=8)) for v in test]
        res = run_model(mid, test, code, dom, rules, a.workers, a.ceiling)
        T = len(test)
        row = {
            "model": mid, "domain": "airline", "level": 0, "n": T, "seed": SEED,
            "ceiling": a.ceiling, "code_extracted": bool(code),
            "form_struct_acc": round(sum(struct) / T, 3),
            "extract_exact_acc": round(sum(r["extract_exact"] for r in res) / T, 3),
            # extraction that preserves every field the rules read -- the number the
            # end-to-end accuracy actually depends on
            "extract_keeps_acc": round(sum(r["keeps"] for r in res) / T, 3),
            "e2e_acc": round(sum(r["e2e"] for r in res) / T, 3),
            "plaintext_acc": round(sum(r["plaintext"] for r in res) / T, 3),
            "pt_noanswer": sum(r["pt_noanswer"] for r in res),
            "rewrapped": sum(r["rewrapped"] for r in res),
            "gap": round((sum(r["e2e"] for r in res)
                          - sum(r["plaintext"] for r in res)) / T, 3),
            "tokens": [sum(r["tok"][j] for r in res) for j in range(4)],
            "form_tokens": [fpt, fct],
            "per_case_struct": [int(x) for x in struct],
            "per_case_extract": [int(r["extract_exact"]) for r in res],
            "per_case_keeps": [int(r["keeps"]) for r in res],
            "per_case_e2e": [int(r["e2e"]) for r in res],
            "per_case_plaintext": [int(r["plaintext"]) for r in res],
            "diffs": [r["diff"] for r in res],
        }
        with lock, open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{mid:16s} prog(struct)={row['form_struct_acc']:.2f} "
              f"extract_exact={row['extract_exact_acc']:.2f} "
              f"keeps={row['extract_keeps_acc']:.2f} "
              f"e2e={row['e2e_acc']:.2f} read+reason={row['plaintext_acc']:.2f} "
              f"gap={row['gap']:+.2f} noans={row['pt_noanswer']} "
              f"rewrap={row['rewrapped']} n={T}", flush=True)


if __name__ == "__main__":
    main()
