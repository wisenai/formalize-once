"""SolverBranch protocol + shared helpers (SPEC §8)."""
from __future__ import annotations

import re
import statistics
from typing import Optional, Protocol

from ..schemas import BranchOutput, Variant


class SolverBranch(Protocol):
    name: str

    def solve(self, variant: Variant, model) -> BranchOutput: ...


def extract_code(text: str) -> Optional[str]:
    """Pull the first ```python fenced block (or any fenced block) from a response."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # unclosed fence (e.g. response truncated mid-code): take everything after it
    m = re.search(r"```(?:python)?\s*\n(.*)", text, re.DOTALL)
    if m and ("def " in m.group(1)):
        return m.group(1).strip()
    # fall back: if the whole thing looks like code with a def
    if "def solve" in text or "def compute" in text:
        return text.strip()
    return None


def consensus(values, output_type: str):
    """Aggregate best-of-n answers: mode for integers, median for decimals."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if output_type == "integer":
        ints = [int(round(float(v))) for v in vals]
        return statistics.mode(ints)
    floats = [float(v) for v in vals]
    return statistics.median(floats)
