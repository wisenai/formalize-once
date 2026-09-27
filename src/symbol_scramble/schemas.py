"""Pydantic v2 schemas (IMPLEMENTATION_SPEC §4)."""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

Number = Union[float, int, str]
Level = Literal["L0", "L1", "L2", "Lu", "L3", "L4", "Flip"]
Family = Literal["equation", "score", "date"]
Branch = Literal["plain_cot", "scaled", "neurosymbolic", "open_book"]
FailMode = Literal["formalization", "execution", "arithmetic", "none"]


class SeedItem(BaseModel):
    row_id: int
    calc_id: str
    calc_name: str
    category: str
    output_type: Literal["decimal", "integer", "date"]
    family: Family
    note: str
    question: str
    entities: dict[str, Any]
    gt_answer: Number
    lower: Optional[float] = None
    upper: Optional[float] = None


class Variant(BaseModel):
    variant_id: str  # f"{row_id}:{level}:{k}"
    seed_row_id: int
    calc_id: str
    calc_name: str
    family: Family
    output_type: Literal["decimal", "integer", "date"]
    level: Level
    note: str
    question: str
    entities: dict[str, Any]
    gt_answer: Number
    lower: Optional[float] = None
    upper: Optional[float] = None
    answer_invariant: bool
    op_params: dict[str, Any] = Field(default_factory=dict)
    qc_passed: bool = True


class BranchOutput(BaseModel):
    variant_id: str
    branch: Branch
    model_id: str
    raw_response: str
    parsed_answer: Optional[Number] = None
    generated_code: Optional[str] = None
    exec_ok: Optional[bool] = None
    n_refine: Optional[int] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class GradedRecord(BranchOutput):
    correct: bool = False
    fail_mode: Optional[FailMode] = None
    ref_answer: Optional[Number] = None  # oracle answer on the variant's entities
