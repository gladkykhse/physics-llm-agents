import logging as log
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode

from src.agents.utils.llm import make_llm
from src.agents.utils.tools import retriever, retriever_backend
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/baseline_react_agent.yaml")

log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    react_iter: int


class PhysicsReactAgent:
    def __init__(self) -> None:
        self.llm = make_llm().bind_tools([retriever])

        self.tools = ToolNode([retriever])

        graph = StateGraph(State)

        graph.add_node("agent", self._agent)
        graph.add_node("tools", self.tools)

        graph.set_entry_point("agent")

        graph.add_conditional_edges(
            "agent",
            self._route_agent,
            {"tools": "tools", "end": END},
        )

        graph.add_edge("tools", "agent")

        self.graph = graph.compile()

    def _agent(self, state: State) -> State:
        ai = self.llm.invoke(state["messages"])

        log.info(f"[AGENT] Output: {ai.content}")

        tool_calls = getattr(ai, "tool_calls", None)
        if tool_calls:
            if len(tool_calls) > 1:
                log.info("[AGENT] - Multiple tool calls, truncating to first one")
                ai.tool_calls = [tool_calls[0]]

            log.info(f"[AGENT] - Tool Calls: {ai.tool_calls}")

        state["messages"] = [*state["messages"], ai]
        state["react_iter"] += 1
        return state

    def _route_agent(self, state: State) -> str:
        max_iters = agent_cfg["max_react_iters"]
        last = state["messages"][-1]
        tcs = getattr(last, "tool_calls", None)

        if state["react_iter"] >= max_iters:
            log.info("[ROUTER] Max react iterations reached")
            return "end"

        if tcs:
            log.info("[ROUTER] Going to tools")
            return "tools"

        log.info("[ROUTER] No tool calls, ending.")
        return "end"

    def solve(self, problem: str) -> str:
        retriever_backend.clear_memory()

        state: State = {
            "messages": [
                SystemMessage(content=agent_cfg["main_system_prompt"]),
                HumanMessage(content=f"Problem:\n{problem}"),
            ],
            "react_iter": 0,
        }

        final_state = self.graph.invoke(state, config={"recursion_limit": 200})
        msgs = final_state.get("messages", [])

        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                return msg.content

        return msgs[-1].content if msgs else ""
