"""RuleArena core experiment: formalize-once (neuro-symbolic) vs plain per-case,
across a spread of CAPABLE models. Concurrent + cached/resumable. Domain-aware:
airline (exact-integer cost) and tax (float Form-1040 amount owed/refunded).

formalize-once: the model writes ONE compute(info) program for the entire ruleset,
given a few worked (info->answer) examples; that single program is executed on every
test case. plain: per-case CoT on the same structured info. Both graded by the
deterministic oracle.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .. import paths
from ..branches.base import extract_code
from ..llm.client import make_client
from ..sandbox import run_solve
from . import oracle
from .loader import load_airline, load_tax

# Errors that mean "stop the run", as opposed to "this one call failed". A run that
# continues past these fills every remaining cell with errors instead of halting with
# its progress intact.
TERMINAL = ("limit exceeded", "quota", "permission", "401", "403", "insufficient",
            "empty response", "requires more credits", "can only afford")
# ...but not every 402 is terminal. OpenRouter returns 402 both for an empty balance
# and for in_flight_budget_exhausted, which means too many concurrent requests are
# each reserving credit at once. The second is transient, carries a Retry-After, and
# clears on its own as requests settle; halting on it throws away a cell that would
# have completed.
IN_FLIGHT = "in_flight_budget_exhausted"

RESULTS = paths.RESULTS / "rulearena"
CALLCACHE = RESULTS / "callcache"  # per-LLM-call disk cache: interrupt-safe, no re-spend


class _Resp:
    __slots__ = ("text", "prompt_tokens", "completion_tokens")

    def __init__(self, text, pt, ct):
        self.text, self.prompt_tokens, self.completion_tokens = text, pt, ct


def _complete_waiting_out_in_flight(model, system, user, kw, tries=5):
    """model.complete(), but a concurrency-budget 402 is waited out, not raised.

    OpenRouter reserves credit for every in-flight request, so enough parallel calls
    can exhaust the reservation pool while the balance is fine. The response carries
    Retry-After and the condition clears as calls settle."""
    for attempt in range(tries):
        try:
            return model.complete(system, user, **kw)
        except Exception as e:                        # noqa: BLE001
            if IN_FLIGHT not in str(e) or attempt == tries - 1:
                raise
            m = re.search(r"'Retry-After': '(\d+)'", str(e))
            wait = min(int(m.group(1)) if m else 60, 180) * (attempt + 1)
            print(f"    in-flight credit budget exhausted; waiting {wait}s "
                  f"(attempt {attempt + 1}/{tries})", flush=True)
            time.sleep(wait)


def _cached_complete(model, mid, system, user, sample_key, **kw):
    """model.complete() wrapped in a content-addressed disk cache. The key includes a
    sample_key so repeated same-prompt samples (self-consistency, best-of-n candidates)
    are distinct cache entries. On restart after any interruption, completed calls are
    replayed from disk for free; only never-completed calls hit the API and cost money."""
    CALLCACHE.mkdir(parents=True, exist_ok=True)
    key = json.dumps({"mid": mid, "sys": system, "user": user, "kw": kw,
                      "s": sample_key}, sort_keys=True)
    fp = CALLCACHE / (hashlib.sha256(key.encode()).hexdigest()[:32] + ".json")
    if fp.exists():
        try:
            d = json.loads(fp.read_text())
            # trust a non-empty response, or an explicitly recorded no-answer (the
            # model exhausted its completion ceiling without emitting text — a real,
            # deterministic model outcome, not an infra failure worth re-buying)
            if d.get("text", "").strip() or d.get("no_answer"):
                return _Resp(d["text"], d["pt"], d["ct"])
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt/partial cache file: fall through and re-fetch
    # Retry on an empty body (a dropped connection can return HTTP 200 with no text;
    # such a response must NOT be cached or it would poison every future resume).
    # Exception: empty text with finish_reason "length" is the MODEL's outcome, not
    # infra — it exhausted the completion ceiling reasoning without emitting an
    # answer. Deterministic at temperature 0, so retrying only re-buys the same
    # failure; record it as a no-answer and let grading score it unanswered.
    r = None
    for _ in range(4):
        r = _complete_waiting_out_in_flight(model, system, user, kw)
        if r.text and r.text.strip():
            break
        if getattr(r, "finish_reason", "") == "length":
            print(f"    [{mid}] no-answer: completion ceiling exhausted while "
                  f"reasoning (finish_reason=length)", flush=True)
            tmp = fp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"text": "", "pt": r.prompt_tokens,
                                       "ct": r.completion_tokens, "no_answer": True,
                                       "finish_reason": "length"}))
            tmp.replace(fp)
            return _Resp("", r.prompt_tokens, r.completion_tokens)
    if not (r.text and r.text.strip()):
        # persistent emptiness => systematic failure (quota/outage). Halt loudly so
        # partial progress is kept, rather than silently scoring the cell 0.
        raise RuntimeError("empty response after retries")
    tmp = fp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"text": r.text, "pt": r.prompt_tokens,
                               "ct": r.completion_tokens}))
    tmp.replace(fp)  # atomic: a half-written file never looks like a cache hit
    return _Resp(r.text, r.prompt_tokens, r.completion_tokens)

# --- domain descriptions of the `info` dict the compute(info) program receives ---
AIRLINE_SCHEMA = (
    'info dict: base_price(int), customer_class(str e.g. "Basic Economy","Main Cabin",'
    '"Main Plus","Premium Economy","Business","First"), routine(str non-US endpoint '
    'region/country), direction(int: 0=departing US, 1=arriving US), bag_list(list; '
    'bag_list[0]=carry-on/personal item = free, bag_list[1:]=CHECKED bags each '
    '{"name","size":[L,W,H] inches,"weight" lbs}). Return int total = ticket + all '
    "checked-bag fees (incl. oversize by linear inches and overweight by lbs).")

TAX_SCHEMA = (
    "info dict: the taxpayer's Form 1040 input line items. Keys are self-descriptive "
    "field names (e.g. filing_status, age, spouse_age, blind, num_qualifying_children, "
    "num_other_dependents, wage_tip_compensation, taxable_interest, ordinary_dividends, "
    "taxable_pensions, taxable_social_security_benefits, qualified_business_income, "
    "federal_income_tax_withheld, ...). Compute the Form 1040 result and return a float: "
    "the amount OWED (positive) or OVERPAID/refunded (negative). comp_0 payers do not "
    "itemize, are not self-employed, and claim no education/child credits.")


class Domain:
    def __init__(self, key, loader, schema, noun, tol):
        self.key = key
        self.loader = loader
        self.schema = schema
        self.noun = noun      # human phrase for the plain prompt
        self.tol = tol        # abs tolerance for "correct" (0 => exact)

    def rules(self):
        return oracle.reference_rules(self.key)

    def fmt_answer(self, a):
        return int(a) if self.tol == 0 else round(float(a), 2)

    def correct(self, pred, gt):
        if pred is None:
            return False
        try:
            return abs(float(pred) - float(gt)) <= self.tol
        except (TypeError, ValueError):
            return False

    def band(self, pred, gt):
        """Approximately-correct: within max($1, 1% of |gt|). Distinguishes a
        structurally-correct-but-imprecise answer from a genuinely wrong one."""
        if pred is None:
            return False
        try:
            return abs(float(pred) - float(gt)) <= max(1.0, 0.01 * abs(float(gt)))
        except (TypeError, ValueError):
            return False


DOMAINS = {
    "airline": Domain("airline", load_airline, AIRLINE_SCHEMA,
                      "airline costs (ticket + checked-bag fees incl oversize/overweight)",
                      tol=0),
    "tax": Domain("tax", load_tax, TAX_SCHEMA,
                  "the Form 1040 amount owed (positive) or overpaid/refunded (negative)",
                  tol=1.0),
}


def _wrap(code, info):
    return code + f"\ndef solve():\n    return compute({info!r})"


def _plain_num(t):
    """Parse the final numeric answer (supports negatives, decimals, $ and commas)."""
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL)
    pat = r"-?\$?\s*[\d,]+(?:\.\d+)?"
    m = re.findall(r"FINAL ANSWER:\s*(" + pat + r")", t)
    cand = m[-1] if m else None
    if cand is None:
        n = re.findall(pat, t)
        cand = n[-1] if n else None
    if cand is None:
        return None
    cand = cand.replace("$", "").replace(",", "").replace(" ", "")
    try:
        return float(cand)
    except ValueError:
        return None


def _majority(preds, dom):
    """Self-consistency vote over parsed numeric answers. Bucket to the grading
    granularity (integer for airline; nearest $1 for tax, matching the tolerance),
    pick the most common bucket, and return an actual prediction from it."""
    from collections import Counter
    vals = [p for p in preds if p is not None]
    if not vals:
        return None
    keyf = (lambda x: int(round(x))) if dom.tol == 0 else (lambda x: round(float(x)))
    win_key, _ = Counter(keyf(v) for v in vals).most_common(1)[0]
    for v in vals:  # representative actual value from the winning bucket
        if keyf(v) == win_key:
            return v
    return vals[0]


def _worked_score(code, worked, dom):
    """How many labeled worked examples this program gets right (0..len)."""
    if not code:
        return -1
    s = 0
    for p in worked:
        sb = run_solve(_wrap(code, p.entities), timeout=8)
        if sb.ok and dom.correct(sb.value, p.gt_answer):
            s += 1
    return s


def _one_program(model, mid, user, max_tokens, cand_idx):
    r = _cached_complete(
        model, mid,
        "You formalize a fixed ruleset into one correct reusable Python program.",
        user, ("formalize", cand_idx),
        max_tokens=max_tokens, temperature=0.4, reasoning="8000")
    return extract_code(r.text), r.prompt_tokens, r.completion_tokens


def formalize_once(model, mid, dom, worked, n_candidates=3, max_tokens=60000,
                   full_sweep=False):
    """Generate several candidate programs and select the one that best matches the
    labeled worked examples (principled variance reduction, using few-shot answers).

    full_sweep draws all n_candidates instead of stopping at the first program that
    reproduces every worked example. It does NOT change which program is selected:
    the winner is chosen with a strict `sc > best_score`, so the first candidate at
    the maximum keeps the slot whether or not later ones are drawn. What it buys is
    the spread -- how good the candidates this protocol did NOT pick were.

    That spread matters on the harder tiers, where the selection signal is only five
    worked examples and is too weak to separate programs: two candidates can both
    reproduce all five while one scores 0.52 on the test set and the other 0.00. A
    cell reported without the spread hides the fact that its number was partly a draw.

    Returns (code, prompt_tokens, completion_tokens, cands), where cands is a list of
    (worked_score, code, cum_prompt_tokens, cum_completion_tokens) in draw order. The
    entry at which early stopping would have fired carries the protocol's cost, so
    the sweep's extra candidates never inflate the reported formalize cost."""
    ex = "\n".join(f"info={p.entities!r}  ->  {dom.fmt_answer(p.gt_answer)}" for p in worked)
    user = (f"Ruleset:\n{dom.rules()}\n\n{dom.schema}\n\nWorked examples "
            f"(your program MUST reproduce all of these):\n{ex}\n\nWrite compute(info) "
            f"implementing the ENTIRE ruleset, returning the numeric answer. stdlib only. "
            f"Return ONLY a ```python code block.")
    best, best_score, pt, ct = None, -1, 0, 0
    cands, stop_at = [], None
    for ci in range(n_candidates):
        try:
            code, a, b = _one_program(model, mid, user, max_tokens, ci)
        except Exception as e:
            # transient network/timeout after the client's own retries: skip this
            # candidate rather than crash the whole run (a quota error still halts).
            if any(s in str(e).lower() for s in TERMINAL):
                raise
            continue
        pt += a
        ct += b
        sc = _worked_score(code, worked, dom)
        cands.append((sc, code, pt, ct))
        if sc > best_score:
            best, best_score = code, sc
        if best_score == len(worked):  # perfect on worked examples
            if stop_at is None:
                stop_at = len(cands)   # where the protocol would have stopped
            if not full_sweep:
                break
    if stop_at is not None and stop_at < len(cands):
        pt, ct = cands[stop_at - 1][2], cands[stop_at - 1][3]  # protocol cost
    return best, pt, ct, cands




def run(models, domain="airline", n_test=40, n_worked=5, level=0, max_workers=16,
        n_candidates=3, plain_fewshot=False, plain_budget="3000", plain_sc=1,
        problems=None, out_tag="", plain_ceiling=None,
        record_candidates=False):
    """plain_fewshot: give the plain branch the same worked examples formalize gets
    (fair, few-shot). plain_budget: plain per-call reasoning budget (match to the
    formalize budget of 8000 for a budget-matched comparison). plain_sc: number of
    self-consistency samples for the plain branch (1 = single pass; set to n_candidates
    with plain_budget=8000 to match formalize's best-of-N sampling AND budget, making the
    accuracy comparison airtight against the sampling/budget asymmetry).
    record_candidates: draw every formalize candidate rather than stopping at the
    first perfect one, and store each one's test accuracy alongside the selected
    program's. Selection is unchanged (see formalize_once); this only records how
    much of the cell's number was the draw. Executing the extra candidates is free --
    only drawing them costs, and the reported formalize cost stays on the protocol.
    plain_ceiling: override the plain completion ceiling. The default 16,000 is not
    always enough: a model whose reasoning cap the gateway does not honor can spend
    the whole ceiling reasoning and emit no answer, which grading would score wrong.
    That is a harness artifact, not a model outcome, so raise the ceiling for any
    cell where it binds. Raising a non-binding ceiling cannot change a result."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    dom = DOMAINS[domain]
    # `problems` (e.g. freshly generated cases) overrides the loader; `out_tag` isolates
    # the output file so an alternate run never touches the main results.
    probs = problems if problems is not None else dom.loader(level, limit=n_test + n_worked)
    worked, test = probs[:n_worked], probs[n_worked:n_worked + n_test]
    rules = dom.rules()
    # optional few-shot block for the plain branch (same info->answer examples as
    # formalize-once), so both strategies see the same labeled examples.
    plain_shots = ("Worked examples (info -> correct answer):\n"
                   + "\n".join(f"info={p.entities!r}  ->  {dom.fmt_answer(p.gt_answer)}"
                               for p in worked) + "\n\n") if plain_fewshot else ""
    out_path = RESULTS / f"{domain}{out_tag}_core.jsonl"
    lock = threading.Lock()

    def record(row):
        with lock, open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    # Resume: skip models already completed for this exact (domain, level, n,
    # fewshot, budget) config so a budget halt never loses or repeats work.
    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if (r.get("level", 0) == level and r.get("n") == n_test
                    and r.get("plain_fewshot") == plain_fewshot
                    and r.get("plain_budget") == plain_budget
                    and r.get("plain_sc", 1) == plain_sc
                    and r.get("code_extracted")          # skip degraded/no-code rows
                    # ...and skip rows whose plain branch never actually ran. A cell
                    # abandoned mid-way (credit cap, outage) is recorded with its
                    # unrun cases flagged; treating it as done would resume straight
                    # past the one cell that still needs buying.
                    and not sum(r.get("per_case_plain_err", []))):
                done.add(r["model"])

    summary = {}
    for mid in models:
        if mid in done:
            print(f"{mid:16s} [{domain}] already done (resume) -- skipping", flush=True)
            continue
        model = make_client(mid)
        # 1) formalize once
        code, fpt, fct, cands = formalize_once(
            model, mid, dom, worked, n_candidates=n_candidates,
            full_sweep=record_candidates)
        prog_correct = prog_err = 0

        def eval_prog(v):
            if not code:
                return ("err", False, False)
            sb = run_solve(_wrap(code, v.entities), timeout=8)
            if sb.ok:
                return ("ok", dom.correct(sb.value, v.gt_answer),
                        dom.band(sb.value, v.gt_answer))
            return ("err", False, False)

        # 2) plain per-case (concurrent). Generous ceiling so reasoning tokens
        # (whose cap OpenRouter doesn't always honor) can't truncate the answer.
        def plain_case(v):
            try:
                # self-consistency: sample plain_sc answers (temperature>0 for
                # diversity when sampling more than once) and majority-vote. Tokens
                # accumulate across all samples, so the plain cost reflects them.
                preds, tpt, tct = [], 0, 0
                temp = 0.7 if plain_sc > 1 else 0
                sys = f"You compute {dom.noun} by applying the given rules."
                usr = (f"Rules:\n{rules}\n\n{plain_shots}Case (structured): "
                       f"{json.dumps(v.entities)}\n\nCompute exactly {dom.noun}. Reason "
                       f"step by step, then last line exactly: FINAL ANSWER: <number>")
                for si in range(plain_sc):
                    r = _cached_complete(
                        model, mid, sys, usr, ("plain", json.dumps(v.entities), si),
                        max_tokens=(plain_ceiling if plain_ceiling
                                    else max(16000, int(plain_budget) + 4000)),
                        temperature=temp, reasoning=plain_budget)
                    preds.append(_plain_num(r.text))
                    tpt += r.prompt_tokens
                    tct += r.completion_tokens
                pred = _majority(preds, dom)
                return (dom.correct(pred, v.gt_answer), dom.band(pred, v.gt_answer),
                        tpt, tct)
            except Exception as e:
                # a quota/auth failure is NOT a wrong answer — it would silently zero
                # the accuracy for every remaining case. Let it halt loudly. Only
                # isolate genuine single-case content failures.
                msg = str(e).lower()
                if any(s in msg for s in TERMINAL):
                    raise
                print(f"    [{mid}] plain case ERRORED (scored as not-run, excluded "
                      f"from accuracy): {type(e).__name__}: {e}", flush=True)
                return ("err", False, 0, 0)

        T = len(test)
        plain_hit = [False] * T  # per-case plain correctness (indexed by test position)
        plain_err = [False] * T  # per-case transient-error flag (call never completed)
        ppt = pct = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            prog_res = list(ex.map(eval_prog, test))
            plain_futs = {ex.submit(plain_case, v): i for i, v in enumerate(test)}
            for fut in as_completed(plain_futs):
                i = plain_futs[fut]
                ok, bd, a, b = fut.result()
                if ok == "err":
                    plain_err[i] = True   # recorded, never conflated with a wrong answer
                else:
                    plain_hit[i] = bool(ok)
                ppt += a
                pct += b
        if any(plain_err):
            print(f"    [{mid}] WARNING: {sum(plain_err)}/{T} plain cases errored and "
                  f"did not run; per_case_plain_err marks them. Re-run to complete "
                  f"the cell before reporting it.", flush=True)
        cand_accs = []
        if record_candidates:
            for sc, ccode, _, _ in cands:
                if not ccode:
                    cand_accs.append([sc, None])
                    continue
                n_ok = sum(1 for v in test
                           if (lambda sb: sb.ok and dom.correct(sb.value, v.gt_answer))(
                               run_solve(_wrap(ccode, v.entities), timeout=8)))
                cand_accs.append([sc, round(n_ok / len(test), 3)])
        prog_hit = [status == "ok" and bool(ok) for status, ok, bd in prog_res]
        prog_err = sum(1 for status, _, _ in prog_res if status == "err")
        prog_correct = sum(prog_hit)
        prog_band = sum(bd for _, _, bd in prog_res)
        plain_correct = sum(plain_hit)
        plain_band = 0  # band no longer reported; kept for schema stability

        row = {
            "model": mid, "domain": domain, "level": level, "n": T,
            "n_candidates": n_candidates,
            "plain_fewshot": plain_fewshot, "plain_budget": plain_budget,
            "plain_sc": plain_sc,
            "formalize_once_acc": round(prog_correct / T, 3),
            "formalize_band_acc": round(prog_band / T, 3),
            "formalize_exec_err": prog_err,
            "plain_acc": round(plain_correct / T, 3),
            "plain_band_acc": round(plain_band / T, 3),
            "gap": round((prog_correct - plain_correct) / T, 3),
            "code_extracted": bool(code),
            "form_tokens": [fpt, fct], "plain_tokens": [ppt, pct],
            "per_case_form": [int(x) for x in prog_hit],
            "per_case_plain": [int(x) for x in plain_hit],
            "per_case_plain_err": [int(x) for x in plain_err],
            # [worked_score, test_accuracy] per candidate, in draw order. The selected
            # program is the first at the maximum worked score.
            "candidates": cand_accs,
        }
        summary[mid] = row
        record(row)
        print(f"{mid:16s} [{domain}] formalize_once={row['formalize_once_acc']:.2f} "
              f"(band={row['formalize_band_acc']:.2f} exec_err={prog_err}) "
              f"plain={row['plain_acc']:.2f} (band={row['plain_band_acc']:.2f}) "
              f"gap={row['gap']:+.2f} n={T} nc={n_candidates}"
              + (f" cand={[a for _, a in cand_accs]}" if cand_accs else ""),
              flush=True)
    return summary
