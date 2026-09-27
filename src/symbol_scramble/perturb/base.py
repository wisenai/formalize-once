"""Operator protocol + shared Variant construction with oracle-recomputed GT."""
from __future__ import annotations

from typing import Any, Optional, Protocol

from .. import calc_runner
from ..schemas import SeedItem, Variant

# MedCalc-Bench uses a +/-5% tolerance band for decimal answers.
TOL = 0.05


def recompute_band(answer: float) -> tuple[float, float]:
    lo, hi = answer * (1 - TOL), answer * (1 + TOL)
    return (lo, hi) if lo <= hi else (hi, lo)


def make_variant(
    seed: SeedItem,
    level: str,
    k: int,
    entities: dict[str, Any],
    note: str,
    answer_invariant: bool,
    op_params: dict[str, Any],
    *,
    question: Optional[str] = None,
) -> Optional[Variant]:
    """Build a Variant, recomputing GT from ``entities`` via the oracle.

    Returns ``None`` if the oracle cannot compute an answer for these entities.
    """
    try:
        ans = calc_runner.answer(seed.calc_id, entities)
    except calc_runner.CalcError:
        return None

    if seed.output_type == "integer":
        gt: Any = int(round(float(ans)))
        lower = upper = None
    elif seed.output_type == "decimal":
        gt = float(ans)
        lower, upper = recompute_band(gt)
    else:
        gt = ans
        lower = upper = None

    return Variant(
        variant_id=f"{seed.row_id}:{level}:{k}",
        seed_row_id=seed.row_id,
        calc_id=seed.calc_id,
        calc_name=seed.calc_name,
        family=seed.family,
        output_type=seed.output_type,
        level=level,  # type: ignore[arg-type]
        note=note,
        question=question if question is not None else seed.question,
        entities=entities,
        gt_answer=gt,
        lower=lower,
        upper=upper,
        answer_invariant=answer_invariant,
        op_params=op_params,
        qc_passed=True,
    )


class Operator(Protocol):
    level: str
    applies_to: set[str]
    answer_invariant: bool
    needs_llm: bool

    def generate(
        self, seed: SeedItem, rng, n: int, rewriter=None
    ) -> list[Variant]: ...
