"""Versioned prompt templates. The hash is stored in every result for repro."""
from __future__ import annotations

import hashlib

RESKIN_SYSTEM = (
    "You rewrite clinical patient notes to break verbatim memorization while "
    "preserving every clinical fact. You are meticulous: no numeric value, unit, "
    "sex, or documented condition may change."
)

RESKIN_USER = """Rewrite the patient note below so it reads as a different case report while keeping ALL clinical facts identical.

Rules:
- Change the patient's name, the hospital, dates, and pronoun-level phrasing where natural.
- Paraphrase the prose (reorder sentences, vary wording).
- DO NOT change any numeric value, any unit, the patient's sex, age, or any documented medical condition / history.
- Keep the patient's sex EXPLICIT using a clear gendered word (man/woman or male/female) — never make it gender-neutral (do not write "individual"/"patient" in place of the sex).
- Keep every laboratory value and vital sign exactly as stated (same number, same unit).

Return ONLY the rewritten note, no preamble.

ORIGINAL NOTE:
{note}
"""

# ---- solver branches ---------------------------------------------------------

PLAIN_SYSTEM = (
    "You are a careful clinician solving a medical calculation from a patient note. "
    "Reason step by step, then give the final answer."
)

PLAIN_USER = """Patient note:
{note}

Question: {question}

Work through the calculation step by step. Then on the LAST line write exactly:
FINAL ANSWER: <number>
(a single number, no units)."""

NEUROSYM_SYSTEM = (
    "You convert a medical calculation problem into executable Python. You extract "
    "the needed values from the note and implement the exact clinical formula/score."
)

NEUROSYM_USER = """Patient note:
{note}

Question: {question}

Write a single Python function `solve()` that returns the numeric answer.
Requirements:
- Read every needed value directly from the note (hard-code the extracted numbers inside solve()).
- Implement the clinical formula or scoring rule yourself in code.
- `solve()` must take no arguments and `return` a single number (float or int).
- Use only the Python standard library (`math` is available). No I/O, no network, no input().

Return ONLY a fenced ```python code block containing the function."""

NEUROSYM_REFINE = """Your previous code failed with:
{error}

Here is your previous code:
```python
{code}
```

Fix it. Return ONLY a corrected ```python code block defining `solve()` that returns a single number."""


OPENBOOK_SYSTEM = (
    "You are a careful clinician solving a medical calculation. You are given the "
    "calculator's official specification (formula / scoring rule). Apply it exactly "
    "to the values in the note."
)

OPENBOOK_USER = """Calculator specification (use this exact formula/rule):
{spec}

Patient note:
{note}

Question: {question}

Extract the needed values from the note, apply the specification step by step, then
on the LAST line write exactly:
FINAL ANSWER: <number>
(a single number, no units)."""


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
    return h.hexdigest()[:12]


PROMPT_VERSION = _hash(
    RESKIN_SYSTEM, RESKIN_USER, PLAIN_SYSTEM, PLAIN_USER,
    NEUROSYM_SYSTEM, NEUROSYM_USER, NEUROSYM_REFINE,
    OPENBOOK_SYSTEM, OPENBOOK_USER,
)
