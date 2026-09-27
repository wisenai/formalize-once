"""Frontier-model perturbation generation (clinically realistic, QC-gated).

The templated operators produced stilted text (fixed distractor bank, robotic
temporal clause, grammar bugs — see PERTURBATION_REVIEW.md). A frontier model
writes clinically credible perturbations instead, while the same hard QC gates
keep the label correct:
  - distractor / temporal (answer-invariant): entities are NOT changed, so the
    oracle answer is unchanged by construction; we only verify the original entity
    values still appear in the note (the model didn't quietly alter them).
  - flip: the boolean entity is flipped and the oracle recomputes the new score;
    the model renders a coherent, internally consistent note for that change.
"""
from __future__ import annotations

from typing import Any, Optional

from .. import calc_runner
from . import note_edit

DISTRACTOR_SYS = (
    "You edit clinical notes for a research benchmark. You insert one realistic, "
    "clinically plausible finding that a clinician might genuinely record, while "
    "changing nothing else."
)
DISTRACTOR_USER = """Here is a patient note. Insert exactly ONE additional clinical finding that:
- is realistic and plausible for this patient in a real hospital note, and
- is NOT relevant to computing "{calc_name}" (it must not be any of the values that calculator uses: {inputs}), and
- does not change, remove, or restate any existing number, unit, sex, age, or documented condition.

Return the FULL edited note only (no commentary).

NOTE:
{note}"""

TEMPORAL_SYS = (
    "You edit clinical notes for a research benchmark. You add a realistic later "
    "follow-up measurement as a natural clinical addendum, changing nothing else."
)
TEMPORAL_USER = """Here is a patient note. The presenting/admission {entity} is stated in the note. Add ONE sentence reporting a DIFFERENT {entity} value measured LATER during the stay (e.g., after initial treatment), phrased as a natural clinical follow-up.
Rules:
- Do NOT state or imply which value should be used for any calculation.
- Do NOT change the original admission value or any other value.
- The later value should be clinically plausible (a realistic change over a few days).

Return the FULL edited note only (no commentary).

NOTE:
{note}"""

FLIP_SYS = (
    "You edit clinical notes for a research benchmark. You revise the note so a "
    "specific risk factor's status changes, keeping the note internally consistent."
)
FLIP_USER = """Here is a patient note. Revise it so that the patient's status for "{condition}" is {status}.
- If making it PRESENT: state the history/finding naturally where a clinician would.
- If making it ABSENT: remove or correct any existing mention so the note does not contradict itself (do not simply append a contradictory sentence).
- Change nothing else: keep all other values, units, sex, age, and conditions identical.

Return the FULL edited note only (no commentary).

NOTE:
{note}"""


def _entities_preserved(note: str, entities: dict[str, Any]) -> bool:
    for v in entities.values():
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            if not note_edit.note_contains_number(note, float(v[0]), tol=0.05):
                return False
    return True


class LLMPerturber:
    def __init__(self, client, retries: int = 2):
        self.client = client
        self.retries = retries
        self.stats = {"ok": 0, "dropped": 0, "errors": 0}

    def _call(self, system: str, user: str) -> Optional[str]:
        for _ in range(self.retries):
            try:
                r = self.client.complete(system=system, user=user, max_tokens=2200,
                                         temperature=0.7, reasoning="off")
            except Exception:  # noqa: BLE001
                self.stats["errors"] += 1
                continue
            t = r.text.strip()
            if t:
                return t
        return None

    def distractor(self, note: str, entities: dict, calc_id: str, calc_name: str) -> Optional[str]:
        inputs = ", ".join(sorted(entities.keys())) or "the calculator inputs"
        for _ in range(self.retries):
            new = self._call(DISTRACTOR_SYS,
                             DISTRACTOR_USER.format(calc_name=calc_name, inputs=inputs, note=note))
            if new and new != note and _entities_preserved(new, entities):
                self.stats["ok"] += 1
                return new
        self.stats["dropped"] += 1
        return None

    def temporal(self, note: str, entities: dict, entity_key: str) -> Optional[str]:
        for _ in range(self.retries):
            new = self._call(TEMPORAL_SYS, TEMPORAL_USER.format(entity=entity_key.lower(), note=note))
            if new and new != note and _entities_preserved(new, entities):
                self.stats["ok"] += 1
                return new
        self.stats["dropped"] += 1
        return None

    def flip(self, note: str, condition: str, present: bool) -> Optional[str]:
        status = "PRESENT (the patient HAS this)" if present else "ABSENT (the patient does NOT have this)"
        new = self._call(FLIP_SYS, FLIP_USER.format(condition=condition, status=status, note=note))
        if new and new != note:
            self.stats["ok"] += 1
            return new
        self.stats["dropped"] += 1
        return None
