"""MedQA boundary tier (M5): the low-formalizability end of the gradient.

Diagnosis MCQA has no computable formula to offload to, so the neuro-symbolic
branch degenerates to natural-language reasoning wrapped in code and confers no
robustness advantage over plain prompting. Showing gap(MedQA) ~ 0 against the
large gap on MedCalc traces the formalizability gradient (proposal Contribution 3).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from . import paths
from .config import load_config
from .llm.client import make_client
from .llm.prompts import PROMPT_VERSION
from .runner import CostTracker
from .sandbox import run_solve

MEDQA_PLAIN_SYS = (
    "You are a careful physician answering a USMLE-style multiple-choice question. "
    "Reason step by step, then give the single best answer letter."
)
MEDQA_PLAIN_USER = """{question}

Options:
{options}

Think step by step, then on the LAST line write exactly:
FINAL ANSWER: <letter> (one of A, B, C, D)."""

MEDQA_NS_SYS = (
    "You solve a USMLE-style question by writing Python. Encode the decision logic "
    "explicitly and return the chosen option letter."
)
MEDQA_NS_USER = """{question}

Options:
{options}

Write a Python function `solve()` that returns the best option as a string, one of
"A", "B", "C", or "D". Encode your clinical decision logic in the code. Use only
the standard library. Return ONLY a fenced ```python code block."""


def load_medqa(n: int = 60) -> list[dict]:
    src = paths.MEDQA_DIR / "medqa_test_60.json"
    data = json.load(open(src))
    items = []
    for r in data["rows"][:n]:
        row = r["row"]
        items.append({
            "id": f"medqa-{r['row_idx']}",
            "question": row["question"],
            "options": row["options"],
            "answer": row["answer_idx"],
        })
    return items


def _fmt_options(opts: dict) -> str:
    return "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))


def _distractor(question: str) -> str:
    """L3-style: insert a plausible but decision-irrelevant clinical sentence."""
    s = "The patient mentions that a distant relative was recently diagnosed with seasonal allergies."
    parts = re.split(r"(?<=[.!?])\s+", question.strip(), maxsplit=1)
    return parts[0] + " " + s + " " + parts[1] if len(parts) == 2 else question + " " + s


def _parse_letter(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.findall(r"FINAL ANSWER:\s*\*?\*?\s*([ABCD])", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([ABCD])\b", text)
    return m[-1].upper() if m else None


def _cache_key(item_id, level, branch, model_id):
    h = hashlib.sha256(
        f"medqa|{item_id}|{level}|{branch}|{model_id}|{PROMPT_VERSION}".encode()
    ).hexdigest()[:20]
    return h


def _solve_plain(model, question, options):
    resp = model.complete(
        system=MEDQA_PLAIN_SYS,
        user=MEDQA_PLAIN_USER.format(question=question, options=_fmt_options(options)),
        max_tokens=1000, temperature=0.0,
    )
    return _parse_letter(resp.text), resp, resp.text


def _solve_ns(model, question, options):
    from .branches.base import extract_code

    resp = model.complete(
        system=MEDQA_NS_SYS,
        user=MEDQA_NS_USER.format(question=question, options=_fmt_options(options)),
        max_tokens=1200, temperature=0.0,
    )
    code = extract_code(resp.text)
    letter, exec_ok = None, False
    if code:
        sb = run_solve(code, timeout=8.0)
        if sb.ok and isinstance(sb.value, str) and sb.value.strip().upper()[:1] in "ABCD":
            letter, exec_ok = sb.value.strip().upper()[:1], True
    if letter is None:  # code failed to yield a letter -> fall back to text
        letter = _parse_letter(resp.text)
    return letter, resp, code, exec_ok


def run_medqa(models=None, levels=("L0", "L3"), n: int = 60, config_name="default") -> str:
    cfg = load_config(config_name)
    models = models or [m["id"] for m in cfg["models"]]
    items = load_medqa(n)
    clients = {mid: make_client(mid) for mid in models}
    tracker = CostTracker(budget_usd=cfg["budget_usd"])
    paths.ensure_dirs()

    work = []
    for it in items:
        for level in levels:
            q = _distractor(it["question"]) if level == "L3" else it["question"]
            for mid in models:
                for branch in ("plain_cot", "neurosymbolic"):
                    key = _cache_key(it["id"], level, branch, mid)
                    if not (paths.CACHE / f"{key}.json").exists():
                        work.append((it, q, level, mid, branch, key))
    print(f"MedQA: {len(items)} items x {len(levels)} levels x {len(models)} models "
          f"x 2 branches; {len(work)} cells to run", flush=True)

    lock = threading.Lock()

    def worker(job):
        it, q, level, mid, branch, key = job
        model = clients[mid]
        if branch == "plain_cot":
            letter, resp, raw = _solve_plain(model, q, it["options"])
            code, exec_ok = None, None
        else:
            letter, resp, code, exec_ok = _solve_ns(model, q, it["options"])
            raw = resp.text
        correct = (letter == it["answer"])
        tracker.add(mid, resp.prompt_tokens, resp.completion_tokens)
        payload = {
            "variant_id": f"{it['id']}:{level}", "branch": branch, "model_id": mid,
            "raw_response": raw[:4000], "parsed_answer": letter, "generated_code": code,
            "exec_ok": exec_ok, "n_refine": 0,
            "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
            "latency_s": resp.latency_s, "correct": correct,
            "fail_mode": None, "ref_answer": it["answer"],
            "level": level, "calc_id": "medqa", "calc_name": "MedQA-USMLE",
            "family": "diagnosis", "answer_invariant": True,
        }
        (paths.CACHE / f"{key}.json").write_text(json.dumps(payload))
        with lock:
            with open(paths.RESULTS / "graded.jsonl", "a") as f:
                f.write(json.dumps(payload) + "\n")
        return correct

    done = 0
    with ThreadPoolExecutor(max_workers=cfg.get("max_workers", 8)) as ex:
        for fut in as_completed([ex.submit(worker, j) for j in work]):
            fut.result()
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(work)} | spent ${tracker.spent:.2f}", flush=True)
    print(f"MedQA done: {done} cells, spent ${tracker.spent:.2f}", flush=True)
    return str(paths.RESULTS / "graded.jsonl")
