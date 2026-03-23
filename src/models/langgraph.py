import polars as pl

from src.agents.thinking_react_agent import PhysicsReactAgent
from src.agents.thinking_react_agent_math_only import PhysicsReactAgent as MathOnlyAgent
from src.agents.thinking_react_agent_knowledge_only import PhysicsReactAgent as KnowledgeOnlyAgent
from src.agents.thinking_react_agent_sequential import PhysicsReactAgent as SequentialAgent
from src.agents.thinking_react_agent_knowledge_only_sequential import PhysicsReactAgent as SequentialKnowledgeOnlyAgent


def run_solving(
    all_requests: list[str],
    agent: str = "plan_react_agent",
) -> pl.DataFrame:
    if agent == "thinking_react_agent":
        agent = PhysicsReactAgent()
    elif agent == "thinking_react_agent_math_only":
        agent = MathOnlyAgent()
    elif agent == "thinking_react_agent_knowledge_only":
        agent = KnowledgeOnlyAgent()
    elif agent == "thinking_react_agent_sequential":
        agent = SequentialAgent()
    elif agent == "thinking_react_agent_knowledge_only_sequential":
        agent = SequentialKnowledgeOnlyAgent()
    else:
        raise ValueError(f"Unknown agent: {agent}")

    results = [agent.solve(problem) for problem in all_requests]
    return pl.DataFrame({"question": all_requests, "answer_ai": results})
