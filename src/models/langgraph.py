import importlib
import inspect

import polars as pl


def _load_agent(name: str):
    module = importlib.import_module(f"src.agents.{name}")
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__ and callable(getattr(obj, "solve", None)):
            return obj()
    raise ValueError(f"No agent class with a 'solve' method found in src/agents/{name}.py")


def run_solving(
    all_requests: list[str],
    agent: str = "plan_react_agent",
) -> pl.DataFrame:
    agent_instance = _load_agent(agent)
    results = [agent_instance.solve(problem) for problem in all_requests]
    return pl.DataFrame({"question": all_requests, "answer_ai": results})
