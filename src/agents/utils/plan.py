from typing import List, Literal

from pydantic import BaseModel, Field, conint


class Step(BaseModel):
    id: conint(ge=1)
    action: Literal["reason", "act"]
    goal: str = Field(..., min_length=2)


class PlanModel(BaseModel):
    inputs: List[str]
    outputs: List[str]
    steps: List[Step]


def render_plan_text(plan: PlanModel | None) -> str:
    if not plan:
        return "Invalid plan"
    lines = ["PLAN SUMMARY"]
    if plan.inputs:
        lines.append("Inputs:")
        for s in plan.inputs:
            lines.append(f"- {s}")
    if plan.outputs:
        lines.append("Outputs:")
        for s in plan.outputs:
            lines.append(f"- {s}")
    lines.append("Steps:")
    for st in plan.steps:
        lines.append(f"- [{st.id}] {st.action}: {st.goal}")
    return "\n".join(lines)
