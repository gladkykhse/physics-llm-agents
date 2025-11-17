import logging as log

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from src.agents.utils.llm import make_llm
from src.agents.utils.tools import retriever, retriever_backend
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/baseline_react_agent.yaml")

log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PhysicsReactAgent:
    def __init__(self) -> None:
        self.graph = create_react_agent(
            make_llm(),
            tools=[retriever],
        )

    def solve(self, problem: str) -> str:
        # per-problem reset for your retriever
        retriever_backend.clear_memory()

        inputs = {
            "messages": [
                SystemMessage(content=agent_cfg["main_system_prompt"]),
                HumanMessage(content=f"Problem:\n{problem}"),
            ]
        }

        # Single shot, no streaming
        result = self.graph.invoke(inputs)

        messages = result.get("messages", [])
        if not messages:
            return ""

        # ---- Log the whole conversation in order ----
        for msg in messages:
            # LangChain messages have .type and .content
            mtype = getattr(msg, "type", None)
            content = getattr(msg, "content", "")

            if mtype == "system":
                log.info(f"[SYSTEM] {content}")
            elif mtype == "human":
                log.info(f"[USER] {content}")
            elif mtype == "tool" or isinstance(msg, ToolMessage):
                log.info(f"[TOOL RESULT] {content}")
            elif mtype == "ai" or isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    log.info(f"[AGENT TOOL CALLS] {tool_calls}")
                log.info(f"[AGENT] {content}")
            else:
                log.info(f"[MESSAGE:{mtype}] {content}")
        # ---------------------------------------------

        # Final answer = last AI message in the history
        final_ai = None
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "ai":
                final_ai = msg
                break

        if final_ai is None:
            # fallback: just return content of the last message
            return getattr(messages[-1], "content", "")

        return final_ai.content
