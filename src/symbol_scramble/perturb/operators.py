"""The perturbation ladder: L0, L1, L2, Lu, L3, L4, Flip (SPEC §7).

Ground truth for every variant is recomputed by the oracle on the resulting
entities (in ``base.make_variant``). Operators that claim answer-invariance
additionally verify the recomputed answer matches the seed answer.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..config import load_distractors, load_entity_ranges
from ..schemas import SeedItem, Variant
from . import base, note_edit

# unit aliases so a note's unit string matches the canonical range unit
_UNIT_ALIASES = {
    "mm hg": "mm hg", "mmhg": "mm hg",
    "bpm": "bpm", "beats per minute": "bpm",
    "meq/l": "meq/l", "mmol/l": "mmol/l",
}


def _norm_unit(u: str) -> str:
    u = u.lower().strip()
    return _UNIT_ALIASES.get(u, u)


def _answers_match(seed: SeedItem, v: Variant) -> bool:
    if seed.output_type == "integer":
        return int(round(float(v.gt_answer))) == int(round(float(seed.gt_answer)))
    try:
        return abs(float(v.gt_answer) - float(seed.gt_answer)) <= 1e-6 + 0.01 * abs(
            float(seed.gt_answer)
        )
    except (TypeError, ValueError):
        return str(v.gt_answer) == str(seed.gt_answer)


# ---- L0 ----------------------------------------------------------------------
class Identity:
    level, applies_to, answer_invariant, needs_llm = "L0", {"equation", "score"}, True, False

    def generate(self, seed, rng, n, rewriter=None, llm_perturber=None):
        v = base.make_variant(seed, "L0", 0, dict(seed.entities), seed.note, True,
                              {"op": "identity"})
        return [v] if v else []


# ---- L1 reskin ---------------------------------------------------------------
class Reskin:
    level, applies_to, answer_invariant, needs_llm = "L1", {"equation", "score"}, True, True

    def generate(self, seed, rng, n, rewriter=None, llm_perturber=None):
        if rewriter is None:
            return []
        out: list[Variant] = []
        for k in range(n):
            new_note = rewriter.reskin(seed.note, seed.entities)
            if new_note is None:
                continue
            v = base.make_variant(seed, "L1", k, dict(seed.entities), new_note, True,
                                  {"op": "reskin"})
            if v and _answers_match(seed, v):
                out.append(v)
        return out


# ---- L2 value shift ----------------------------------------------------------
class ValueShift:
    level, applies_to, answer_invariant, needs_llm = "L2", {"equation", "score"}, False, False

    # `age` defines the patient's clinical context; shifting it by decades while the
    # rest of the narrative (comorbidities, presentation) stays fixed reads as
    # incongruous to a clinician. Shift labs/vitals/anthropometrics only.
    _NO_SHIFT = {"age"}

    def _shiftable(self, seed):
        ranges = load_entity_ranges()
        cand = []
        for key, val in seed.entities.items():
            if not (isinstance(val, list) and len(val) == 2 and isinstance(val[0], (int, float))):
                continue
            if key in self._NO_SHIFT or key not in ranges:
                continue
            if _norm_unit(str(val[1])) != _norm_unit(str(ranges[key]["unit"])):
                continue
            if note_edit.find_number_in_note(seed.note, float(val[0])) is None:
                continue
            cand.append(key)
        return cand

    def generate(self, seed, rng, n, rewriter=None, llm_perturber=None):
        ranges = load_entity_ranges()
        cand = self._shiftable(seed)
        if not cand:
            return []
        out: list[Variant] = []
        seen_answers = set()
        attempts = 0
        while len(out) < n and attempts < n * 6:
            attempts += 1
            key = cand[rng.randrange(len(cand))]
            spec = ranges[key]
            old_val = float(seed.entities[key][0])
            new_val = round(rng.uniform(spec["min"], spec["max"]), int(spec["round"]))
            if int(spec["round"]) == 0:
                new_val = int(new_val)
            if abs(new_val - old_val) < 1e-9:
                continue
            new_note = note_edit.replace_number(seed.note, old_val, note_edit.fmt_num(new_val))
            if new_note is None:
                continue
            new_entities = dict(seed.entities)
            new_entities[key] = [new_val, seed.entities[key][1]]
            v = base.make_variant(seed, "L2", len(out), new_entities, new_note, False,
                                  {"op": "value_shift", "entity": key,
                                   "old": old_val, "new": new_val})
            if not v:
                continue
            # verify the note round-trips to the new value and NOT the old one
            if not note_edit.note_contains_number(new_note, float(new_val), tol=0.5):
                continue
            akey = str(v.gt_answer)
            if akey in seen_answers:
                continue
            seen_answers.add(akey)
            out.append(v)
        return out


# ---- Lu units ----------------------------------------------------------------
class Units:
    level, applies_to, answer_invariant, needs_llm = "Lu", {"equation", "score"}, True, False

    def generate(self, seed, rng, n, rewriter=None, llm_perturber=None):
        out: list[Variant] = []
        for key, val in seed.entities.items():
            if len(out) >= n:
                break
            if not (isinstance(val, list) and len(val) == 2 and isinstance(val[0], (int, float))):
                continue
            conv = note_edit.convert_unit(key, float(val[0]), str(val[1]))
            if conv is None:
                continue
            new_val, new_unit = conv
            new_note = self._rewrite_unit(seed.note, float(val[0]), str(val[1]), new_val, new_unit)
            if new_note is None:
                continue
            new_entities = dict(seed.entities)
            new_entities[key] = [new_val, new_unit]
            v = base.make_variant(seed, "Lu", len(out), new_entities, new_note, True,
                                  {"op": "units", "entity": key,
                                   "from": val[1], "to": new_unit})
            # invariant: oracle answer must be preserved within tolerance
            if v and _answers_match(seed, v):
                out.append(v)
        return out

    @staticmethod
    def _rewrite_unit(note, old_val, old_unit, new_val, new_unit):
        cand = note_edit.find_number_in_note(note, old_val)
        if cand is None:
            return None
        new_num = note_edit.fmt_num(new_val)
        # try "<num><ws><unit>" (unit may be abbreviated / cased differently)
        unit_pat = re.escape(old_unit)
        m = re.search(r"(?<![\d.])" + re.escape(cand) + r"(\s*)(" + unit_pat + r")",
                      note, flags=re.IGNORECASE)
        if m:
            return note[:m.start()] + f"{new_num}{m.group(1)}{new_unit}" + note[m.end():]
        return None  # require the unit to be co-located so the note stays consistent


# ---- L3 distractor -----------------------------------------------------------
class Distractor:
    level, applies_to, answer_invariant, needs_llm = "L3", {"equation", "score"}, True, False

    def _bank(self, seed):
        from .. import calc_runner

        banks = load_distractors()
        pool = list(banks.get("common", [])) + list(banks.get(seed.family, []))
        # verified non-inputs: mentioned tokens must not collide with real inputs
        real_tokens = {k.lower() for k in seed.entities}
        real_tokens |= {a.lower() for a in calc_runner.entity_arg_names(seed.calc_id)}
        ok = []
        for d in pool:
            if any(any(tok in rt or rt in tok for rt in real_tokens)
                   for tok in d["mentions"]):
                continue
            ok.append(d["text"])
        return ok

    def generate(self, seed, rng, n, rewriter=None, llm_perturber=None):
        out: list[Variant] = []
        if llm_perturber is not None:
            for k in range(n):
                new_note = llm_perturber.distractor(seed.note, seed.entities,
                                                    seed.calc_id, seed.calc_name)
                if new_note is None:
                    continue
                v = base.make_variant(seed, "L3", k, dict(seed.entities), new_note, True,
                                      {"op": "distractor", "gen": "llm"})
                if v and _answers_match(seed, v):
                    out.append(v)
            return out
        # templated fallback
        pool = self._bank(seed)
        if not pool:
            return []
        rng.shuffle(pool)
        for k in range(min(n, len(pool))):
            new_note = _insert_sentence(seed.note, pool[k])
            v = base.make_variant(seed, "L3", k, dict(seed.entities), new_note, True,
                                  {"op": "distractor", "text": pool[k]})
            if v and _answers_match(seed, v):
                out.append(v)
        return out


# Entities whose value plausibly changes over a hospital stay (for L4 temporal).
# Static demographics (age, weight, height) are excluded — a "post-treatment age"
# is clinically nonsensical.
_TIME_VARYING = {
    "creatinine", "Pre-operative creatinine", "Sodium", "Chloride", "Bicarbonate",
    "Albumin", "Glucose", "Bilirubin", "Calcium", "Heart Rate or Pulse",
    "Systolic Blood Pressure", "Diastolic Blood Pressure", "QT Interval",
    "Temperature", "Blood Urea Nitrogen (BUN)", "respiratory rate",
    "Total cholesterol", "high-density lipoprotein cholesterol", "Triglycerides",
    "Hematocrit", "O₂ saturation percentage",
}


# ---- L4 compositional / temporal depth --------------------------------------
class Depth:
    level, applies_to, answer_invariant, needs_llm = "L4", {"equation"}, True, False

    def generate(self, seed, rng, n, rewriter=None, llm_perturber=None):
        ranges = load_entity_ranges()
        # time-varying numeric entities that ARE real inputs and appear in the note
        cand = []
        for key, val in seed.entities.items():
            if not (isinstance(val, list) and len(val) == 2 and isinstance(val[0], (int, float))):
                continue
            if key not in ranges or key not in _TIME_VARYING:
                continue
            if note_edit.find_number_in_note(seed.note, float(val[0])) is None:
                continue
            cand.append(key)
        if not cand:
            return []
        out: list[Variant] = []
        attempts = 0
        while len(out) < n and attempts < n * 6:
            attempts += 1
            key = cand[rng.randrange(len(cand))]
            spec = ranges[key]
            admit_val = float(seed.entities[key][0])
            later_val = round(rng.uniform(spec["min"], spec["max"]), int(spec["round"]))
            if int(spec["round"]) == 0:
                later_val = int(later_val)
            if abs(later_val - admit_val) < 1e-9:
                continue
            unit = seed.entities[key][1]
            if llm_perturber is not None:
                new_note = llm_perturber.temporal(seed.note, seed.entities, key)
                if new_note is None:
                    continue
            else:
                # templated fallback: natural addendum, no spoon-feeding instruction
                sentence = (
                    f"A repeat {key.lower()} measured on hospital day 3, after initial "
                    f"treatment, was {note_edit.fmt_num(later_val)} {unit}."
                )
                new_note = _insert_sentence(seed.note, sentence)
            # admission entities unchanged -> answer invariant, verified below
            v = base.make_variant(seed, "L4", len(out), dict(seed.entities), new_note, True,
                                  {"op": "temporal_depth", "entity": key,
                                   "admission": admit_val, "post_treatment": later_val})
            if v and _answers_match(seed, v):
                out.append(v)
        return out


# ---- Flip (score only) -------------------------------------------------------
class Flip:
    level, applies_to, answer_invariant, needs_llm = "Flip", {"score"}, False, False

    def generate(self, seed, rng, n, rewriter=None, llm_perturber=None):
        bool_keys = [k for k, v in seed.entities.items() if isinstance(v, bool)]
        if not bool_keys:
            return []
        rng.shuffle(bool_keys)
        out: list[Variant] = []
        seen = set()
        for key in bool_keys:
            if len(out) >= n:
                break
            new_val = not seed.entities[key]
            new_entities = dict(seed.entities)
            new_entities[key] = new_val
            cond = _clean_condition(key)
            if llm_perturber is not None:
                new_note = llm_perturber.flip(seed.note, cond, present=new_val)
                if new_note is None:
                    continue
            elif new_val:
                # add a genuinely new risk factor (no contradiction)
                new_note = _insert_sentence(
                    seed.note,
                    f"On further review of the records, the patient does have a documented history of {cond}.")
            else:
                # retract a previously noted factor — framed as a chart correction
                new_note = _insert_sentence(
                    seed.note,
                    f"On further review, the earlier note of {cond} was found to be in error; the patient has no history of {cond}.")
            v = base.make_variant(seed, "Flip", len(out), new_entities, new_note, False,
                                  {"op": "flip", "entity": key,
                                   "from": not new_val, "to": new_val})
            if not v or _answers_match(seed, v):
                continue  # require the flip to actually change the score
            if str(v.gt_answer) in seen:
                continue
            seen.add(str(v.gt_answer))
            out.append(v)
        return out


def _clean_condition(key: str) -> str:
    """Turn a boolean entity key into a natural condition phrase (no double words)."""
    c = key.strip()
    # drop templated suffixes like " history", " criteria for the HAS-BLED rule"
    c = re.sub(r"\s+criteria for the .*?rule$", "", c, flags=re.IGNORECASE)
    c = re.sub(r"\s+history$", "", c, flags=re.IGNORECASE)
    c = re.sub(r"\s+History$", "", c)
    return c.strip().lower()


def _insert_sentence(note: str, sentence: str) -> str:
    """Insert a distractor sentence at a natural position (after 1st sentence)."""
    parts = re.split(r"(?<=[.!?])\s+", note.strip(), maxsplit=1)
    if len(parts) == 2:
        return parts[0] + " " + sentence + " " + parts[1]
    return note.rstrip() + " " + sentence


ALL_OPERATORS: dict[str, Any] = {
    op.level: op
    for op in (Identity(), Reskin(), ValueShift(), Units(), Distractor(), Depth(), Flip())
}
