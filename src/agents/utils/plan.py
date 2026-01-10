from typing import List

from pydantic import BaseModel, Field


class PhysicsPlan(BaseModel):
    problem_restatement: str
    knowns: List[str]
    unknowns: List[str]
    steps: List[str] = Field(..., min_items=1, description="Must have at least one reasoning step.")
    final_note: str


class PlanStepV2(BaseModel):
    step: str
    rationale: str
    goal: str


class PhysicsPlanV2(BaseModel):
    knowns: List[str]
    unknowns: List[str] = Field(..., min_items=1, description="Must have at least one unknown.")
    plan: List[PlanStepV2] = Field(..., min_items=1, description="Must have at least one reasoning step.")
    verification: str
