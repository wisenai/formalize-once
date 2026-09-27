"""Load RuleArena problems into the shared Variant schema.

Each problem becomes a Variant whose ``note`` is the reference rules + the problem
prompt (rules are given in-context — this is a rule-APPLICATION/formation task, not
recall), ``entities`` is the structured ``info`` dict, and ``gt_answer`` is the
oracle's computed answer. This lets the existing branches/runner/grading run
unchanged over RuleArena.
"""
from __future__ import annotations

import json

from .. import paths
from ..schemas import Variant
from . import oracle

_QUESTION = {
    "airline": ("What is the TOTAL cost this passenger must pay (ticket price plus all "
                "baggage fees), in US dollars? Give a single integer number."),
    "tax": ("What is this taxpayer's Form 1040 amount owed (positive) or overpaid / "
            "refunded (negative), in US dollars? Give a single number."),
}


def load_airline(level: int = 0, limit: int | None = None) -> list[Variant]:
    path = paths.RAW / "rulearena" / "airline" / f"comp_{level}.jsonl"
    rules = oracle.reference_rules("airline")
    out: list[Variant] = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit and len(out) >= limit:
                break
            rec = json.loads(line)
            info = rec["info"]
            try:
                gt = oracle.airline_answer(info)
            except Exception:  # noqa: BLE001 - skip any problem the oracle can't label
                continue
            note = (
                "You are given the airline's baggage-fee rules:\n\n"
                + rules
                + "\n\n--- Problem ---\n"
                + rec["prompt"]
            )
            out.append(Variant(
                variant_id=f"airline{level}:{i}:L0",
                seed_row_id=i,
                calc_id="airline",
                calc_name="RuleArena-Airline",
                family="equation",
                output_type="integer",
                level="L0",
                note=note,
                question=_QUESTION["airline"],
                entities=info,
                gt_answer=gt,
                lower=None,
                upper=None,
                answer_invariant=True,
                op_params={"domain": "airline", "level_difficulty": level},
                qc_passed=True,
            ))
    return out


# Unfilled slots on RuleArena's harder tax forms. The records are blank-form
# encodings, so a line the taxpayer left empty arrives as the literal placeholder
# rather than as 0 or a missing key.
TAX_BLANKS = ("$TBD", "[__]", "[  ]", "N/A", "--", "")


def _fill_blanks(payer: dict) -> dict:
    """Turn unfilled numeric slots into 0. NOT APPLIED -- kept for reproducibility.

    We tried this and it was wrong, in a way worth recording because the check that
    cleared it looked sound.

    The motivation was real: a generated program does arithmetic on "$TBD" and raises
    TypeError, which grading scored as 32 wrong answers out of 90. A differential test
    said normalizing was free -- replacing every placeholder with 0 leaves all 77
    affected ground-truth answers unchanged, because the reference computation derives
    these lines rather than reading them.

    That test asked whether the ORACLE was indifferent. It never asked whether the
    MODEL's reading was unchanged, and it is not. "[__]" says the line is blank and
    must be derived; 0 says the credit is zero. Per-case accuracy on returns with
    qualifying children collapsed from 0.71 and 0.75 to 0.00 and 0.00 for the two
    models measured both ways: told the credit was zero, they applied no credit and
    were wrong on every such return.

    So the placeholders stay. They are part of the task, the worked examples in the
    formalize prompt contain them, and a program that crashes on one is a real
    formalization failure -- the strategy committing to a type assumption it was shown
    evidence against, where per-case reasoning reads the blank correctly. That is a
    result about the two strategies, not a defect to normalize away.

    The lesson generalizes past this field: an input transformation that provably
    preserves the LABEL can still change what the input MEANS to a model, and
    label-invariance is not evidence of meaning-invariance.
    """
    return {k: (0 if (isinstance(v, str) and v.strip() in TAX_BLANKS
                      and k not in ("name", "filing_status")) else v)
            for k, v in payer.items()}


def load_tax(level: int = 0, limit: int | None = None) -> list[Variant]:
    """RuleArena tax problems. Each record's ``pydantic`` field is the structured
    (input-only) Form 1040 payer; the oracle computes the amount owed/refunded.
    comp_0 is pure basic 1040 (no itemizing/self-employment/credits)."""
    path = paths.RAW / "rulearena" / "tax" / f"comp_{level}.json"
    rules = oracle.reference_rules("tax")
    out: list[Variant] = []
    records = json.load(open(path))
    for i, rec in enumerate(records):
        if limit and len(out) >= limit:
            break
        # Strip fields the reference computation never reads, so the model and the
        # oracle see an identical spec (avoids a benchmark artifact where a model
        # correctly applies an input field the oracle silently omits).
        payer = {k: v for k, v in rec["pydantic"].items()
                 if k not in oracle.TAX_IGNORED_FIELDS}
        try:
            gt = oracle.tax_answer(payer)
        except Exception:  # noqa: BLE001 - skip any problem the oracle can't label
            continue
        note = (
            "You are given the U.S. Form 1040 tax rules:\n\n"
            + rules
            + "\n\n--- Taxpayer (structured) ---\n"
            + json.dumps(payer, indent=1)
        )
        out.append(Variant(
            variant_id=f"tax{level}:{i}:L0",
            seed_row_id=i,
            calc_id="tax",
            calc_name="RuleArena-Tax",
            family="equation",
            output_type="decimal",
            level="L0",
            note=note,
            question=_QUESTION["tax"],
            entities=payer,
            gt_answer=gt,
            lower=None,
            upper=None,
            answer_invariant=True,
            op_params={"domain": "tax", "level_difficulty": level},
            qc_passed=True,
        ))
    return out
