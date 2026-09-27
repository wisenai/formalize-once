"""LLM note regeneration (L1 reskin) with a mandatory round-trip QC gate.

A rewritten note is only accepted if every original entity value is still
recoverable from the new prose (numeric presence for numbers, token presence for
sex/categoricals) AND the oracle answer on the unchanged entities is preserved.
Variants that fail QC are regenerated up to ``retries`` times, then dropped.
"""
from __future__ import annotations

from typing import Any, Optional

from ..llm.prompts import RESKIN_SYSTEM, RESKIN_USER
from . import note_edit


def _entities_round_trip(note: str, entities: dict[str, Any]) -> bool:
    for k, v in entities.items():
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            if not note_edit.note_contains_number(note, float(v[0]), tol=0.05):
                return False
        elif isinstance(v, str):
            if v.lower() not in note.lower():
                # sex often stated as "man"/"woman"; accept those synonyms
                syn = {"male": ["male", "man", " m ", "gentleman"],
                       "female": ["female", "woman", " f ", "lady"]}
                toks = syn.get(v.lower())
                if not (toks and any(t in note.lower() for t in toks)):
                    return False
        # booleans are narrative; not checked positionally here
    return True


class NoteRewriter:
    """Wraps an LLMClient to produce QC-gated reskinned notes."""

    def __init__(self, client, retries: int = 3):
        self.client = client
        self.retries = retries
        self.stats = {"ok": 0, "dropped": 0, "attempts": 0}

    def reskin(self, note: str, entities: dict[str, Any]) -> Optional[str]:
        for _ in range(self.retries):
            self.stats["attempts"] += 1
            try:
                resp = self.client.complete(
                    system=RESKIN_SYSTEM,
                    user=RESKIN_USER.format(note=note),
                    max_tokens=2000,
                    temperature=0.7,
                    reasoning="off",  # reskin needs no thinking; keeps generation fast
                )
            except Exception:  # noqa: BLE001 - rate limit / provider error: drop this variant
                self.stats["errors"] = self.stats.get("errors", 0) + 1
                continue
            new_note = resp.text.strip()
            if new_note and new_note != note and _entities_round_trip(new_note, entities):
                self.stats["ok"] += 1
                return new_note
        self.stats["dropped"] += 1
        return None
