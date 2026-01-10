import polars as pl

from src.agents.baseline_react_agent import PhysicsReactAgent
from src.agents.cot_plan_react_agent_v2 import COTPhysicsAgent
from src.agents.plan_react_agent import PhysicsAgent
from src.agents.simple_planning_agent_cot import SimplePlanningPhysicsAgent


def run_solving(
    all_requests: list[str],
    agent: str = "plan_react_agent",
) -> pl.DataFrame:
    if agent == "plan_react_agent":
        agent = PhysicsAgent()
    elif agent == "baseline_react_agent":
        agent = PhysicsReactAgent()
    elif agent == "cot_plan_react_agent":
        agent = COTPhysicsAgent()
    elif agent == "simple_planning_agent":
        agent = SimplePlanningPhysicsAgent()
    else:
        raise ValueError(f"Unknown agent: {agent}")

    results = [agent.solve(problem) for problem in all_requests]
    return pl.DataFrame({"question": all_requests, "answer_agent": results})
