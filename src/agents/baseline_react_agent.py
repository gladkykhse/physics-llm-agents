import logging as log
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode

from src.agents.utils.llm import make_llm
from src.agents.utils.tools import sympy_eval, sympy_solve, vector_math
from src.agents.utils.utils import scieval_split_problem_and_options
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/baseline_react_agent.yaml")

log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    problem: str
    react_iter: int


class PhysicsReactAgent:
    def __init__(self) -> None:
        tools_list = [sympy_eval, vector_math, sympy_solve]

        self.llm = make_llm(temperature=0.0).bind_tools(tools_list)
        self.base_llm = make_llm(temperature=0.0)

        graph = StateGraph(State)
        self.tools = ToolNode(tools_list)

        graph.add_node("agent", self._agent)
        graph.add_node("tools", self.tools)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("agent")
        graph.add_conditional_edges(
            "agent",
            self._route,
            {"tools": "tools", "finalize": "finalize"},
        )
        graph.add_edge("tools", "agent")
        graph.add_edge("finalize", END)

        self.graph = graph.compile()

    def _agent(self, state: State) -> State:
        ai = self.llm.invoke(state["messages"])

        tool_calls = getattr(ai, "tool_calls", None)
        if tool_calls:
            log.info(f"[AGENT] Tool calls: {tool_calls}")
            if len(tool_calls) > 1:
                log.warning(f"[AGENT] {len(tool_calls)} tool calls returned; truncating to 1.")
                ai.tool_calls = tool_calls[:1]
        else:
            log.info(f"[AGENT] Response (no tools): {ai.content[:200]}")

        state["messages"] = [ai]
        state["react_iter"] += 1
        return state

    def _finalize(self, state: State) -> State:
        prompt_content = agent_cfg["finalizer_prompt"].format(problem=state["problem"])
        messages = state["messages"] + [HumanMessage(content=prompt_content)]
        ai = self.base_llm.invoke(messages)
        log.info(f"[FINALIZE] {ai.content[:200]}")
        state["messages"] = [ai]
        return state

    def _route(self, state: State) -> str:
        if state["react_iter"] >= agent_cfg["max_react_iters"]:
            log.info("[ROUTE] Max iterations reached → finalize")
            return "finalize"

        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)

        if tool_calls:
            log.info("[ROUTE] → tools")
            return "tools"

        log.info("[ROUTE] No tool calls → finalize")
        return "finalize"

    def solve(self, problem: str) -> str:
        question, options = scieval_split_problem_and_options(full_text=problem)
        log.info(f"[QUESTION] - {question}")
        log.info(f"[OPTIONS] - {options}")

        state: State = {
            "messages": [
                SystemMessage(content=agent_cfg["main_system_prompt"]),
                HumanMessage(content=f"# Problem\n{problem}"),
            ],
            "problem": problem,
            "react_iter": 0,
        }

        final_state = self.graph.invoke(state, config={"recursion_limit": 200})
        msgs = final_state.get("messages", [])

        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                return msg.content

        return msgs[-1].content if msgs else ""
