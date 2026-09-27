"""Oracle healthcheck: the calc_runner must reproduce ground truth (SPEC §6, M1 gate)."""
from __future__ import annotations

from dataclasses import dataclass

from . import calc_runner
from .data_loader import load_seed_items
from .grading import is_correct
from .schemas import Variant


@dataclass
class HealthReport:
    total: int
    reproduced: int
    failed: int
    per_calc_fail: dict[str, int]

    @property
    def rate(self) -> float:
        return self.reproduced / self.total if self.total else 0.0


def _as_variant(item) -> Variant:
    return Variant(
        variant_id=f"{item.row_id}:L0:0",
        seed_row_id=item.row_id,
        calc_id=item.calc_id,
        calc_name=item.calc_name,
        family=item.family,
        output_type=item.output_type,
        level="L0",
        note=item.note,
        question=item.question,
        entities=item.entities,
        gt_answer=item.gt_answer,
        lower=item.lower,
        upper=item.upper,
        answer_invariant=True,
    )


def run_healthcheck(families=("equation", "score")) -> HealthReport:
    items = load_seed_items(families=families)
    total = reproduced = failed = 0
    per_calc_fail: dict[str, int] = {}
    for item in items:
        total += 1
        try:
            ans = calc_runner.answer(item.calc_id, item.entities)
        except calc_runner.CalcError:
            failed += 1
            per_calc_fail[item.calc_id] = per_calc_fail.get(item.calc_id, 0) + 1
            continue
        v = _as_variant(item)
        # grade the recomputed oracle answer against the dataset's own GT/band
        if is_correct(ans if item.output_type != "integer" else int(round(float(ans))), v):
            reproduced += 1
        else:
            failed += 1
            per_calc_fail[item.calc_id] = per_calc_fail.get(item.calc_id, 0) + 1
    return HealthReport(total, reproduced, failed, per_calc_fail)
