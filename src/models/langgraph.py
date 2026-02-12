import polars as pl

from src.agents.thinking_react_agent import PhysicsReactAgent


def run_solving(
    all_requests: list[str],
    agent: str = "plan_react_agent",
) -> pl.DataFrame:
    if agent == "thinking_react_agent":
        agent = PhysicsReactAgent()
    else:
        raise ValueError(f"Unknown agent: {agent}")

    results = [agent.solve(problem) for problem in all_requests]
    return pl.DataFrame({"question": all_requests, "answer_agent": results})
