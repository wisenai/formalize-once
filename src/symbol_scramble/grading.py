"""Answer parsing + correctness grading (IMPLEMENTATION_SPEC §9)."""
from __future__ import annotations

import re
from typing import Any, Optional

from .schemas import GradedRecord, Variant

_NUM = r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d*\.\d+|[-+]?\d+"


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def parse_answer(raw: str, output_type: str) -> Optional[Any]:
    """Robustly extract the final number/integer from a model response.

    Priority: explicit ``\\boxed{}`` / ``**answer**`` markers, then
    "answer is/=" phrasings, then the last number in the text.
    """
    if raw is None:
        return None
    text = str(raw)
    # strip any leaked chain-of-thought block so we parse the final answer, not the trace
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)

    # 1. boxed / fenced final-answer markers
    for pat in (
        r"\\boxed\{\s*(" + _NUM + r")",
        r"[Ff]inal\s+[Aa]nswer\s*[:=]?\s*\**\s*(" + _NUM + r")",
        r"[Aa]nswer\s*(?:is|:|=)\s*\**\s*(" + _NUM + r")",
        r"\*\*\s*(" + _NUM + r")\s*\*\*",
    ):
        m = re.findall(pat, text)
        if m:
            val = _to_float(m[-1])
            if val is not None:
                return int(round(val)) if output_type == "integer" else val

    # 2. fall back to the last standalone number in the text
    nums = re.findall(_NUM, text)
    if nums:
        val = _to_float(nums[-1])
        if val is not None:
            return int(round(val)) if output_type == "integer" else val
    return None


def is_correct(parsed: Optional[Any], variant: Variant) -> bool:
    if parsed is None:
        return False
    if variant.output_type == "integer":
        try:
            return int(round(float(parsed))) == int(round(float(variant.gt_answer)))
        except (ValueError, TypeError):
            return False
    if variant.output_type == "decimal":
        try:
            p = float(parsed)
        except (ValueError, TypeError):
            return False
        lo = variant.lower if variant.lower is not None else float(variant.gt_answer)
        hi = variant.upper if variant.upper is not None else float(variant.gt_answer)
        if lo > hi:
            lo, hi = hi, lo
        return lo - 1e-6 <= p <= hi + 1e-6
    # date family is out of scope for v1
    return str(parsed).strip() == str(variant.gt_answer).strip()


def fail_mode(
    record: GradedRecord,
    variant: Variant,
) -> str:
    """Attribute a neuro-symbolic failure (IMPLEMENTATION_SPEC §9).

    - none: the answer is correct.
    - execution: the generated code never produced a value.
    - arithmetic: code ran and matched the oracle answer on these entities, but the
      graded answer is wrong (rounding / band edge / parse) -> right program, wrong number.
    - formalization: code ran but disagrees with the oracle -> wrong program logic/inputs.
    """
    if record.correct:
        return "none"
    if record.exec_ok is False or record.parsed_answer is None:
        return "execution"
    ref = record.ref_answer
    if ref is None:
        return "formalization"
    try:
        if variant.output_type == "integer":
            agrees = int(round(float(record.parsed_answer))) == int(round(float(ref)))
        else:
            agrees = abs(float(record.parsed_answer) - float(ref)) <= max(
                1e-6, 0.02 * abs(float(ref))
            )
    except (ValueError, TypeError):
        agrees = False
    return "arithmetic" if agrees else "formalization"
