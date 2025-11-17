import polars as pl

from src.agents.baseline_react_agent import PhysicsReactAgent
from src.agents.plan_react_agent import PhysicsAgent


def run_solving(
    all_requests: list[str],
    agent: str = "plan_react_agent",
) -> pl.DataFrame:
    if agent == "plan_react_agent":
        agent = PhysicsAgent()
    elif agent == "baseline_react_agent":
        agent = PhysicsReactAgent()
    else:
        raise ValueError(f"Unknown agent: {agent}")

    results = [agent.solve(problem) for problem in all_requests]
    return pl.DataFrame({"question": all_requests, "answer": results})
