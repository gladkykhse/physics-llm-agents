import json
import logging as log
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode

from src.agents.utils.llm import make_llm
from src.agents.utils.tools import retriever_backend, sympy_eval, vector_math
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/thinking_react_agent.yaml")

log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    problem: str
    react_iter: int


class PhysicsReactAgent:
    def __init__(self) -> None:
        tools_list = [sympy_eval, vector_math]

        self.base_llm = make_llm(temperature=0.0)
        self.tools_llm = make_llm(temperature=0.0).bind_tools(tools_list)

        self.memory = set()

        graph = StateGraph(State)
        self.tools = ToolNode(tools_list)

        graph.add_node("thought", self._thought)
        graph.add_node("agent", self._act)
        graph.add_node("tools", self.tools)
        graph.add_node("finalize", self._finalize)
        graph.add_node("thought_skip_tool", self._thought_skip_tool)

        graph.set_entry_point("thought")
        graph.add_edge("thought", "agent")
        graph.add_conditional_edges(
            "agent",
            self._route_act,
            {"tools": "tools", "thought": "thought", "skip_tool": "thought_skip_tool", "end": "finalize"},
        )
        graph.add_edge("thought_skip_tool", "agent")
        graph.add_edge("tools", "thought")
        graph.add_edge("finalize", END)

        self.graph = graph.compile()

    def _thought(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["thought_prompt"])
        msgs = state["messages"] + [prompt]
        ai = self.base_llm.invoke(msgs)
        log.info(f"[THOUGHT] - {ai.content}")

        state["messages"] = [ai]
        return state

    def _thought_skip_tool(self, state: State) -> dict:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)
        tool_call_dict = tool_calls[0]

        tool_msg = ToolMessage(
            content=agent_cfg["tool_invoked_prompt"].format(
                tool_name=tool_call_dict["name"], tool_args=tool_call_dict["args"]
            ),
            tool_call_id=tool_call_dict["id"],
        )

        prompt = HumanMessage(
            content=agent_cfg["thought_tool_invoked_prompt"].format(
                tool_name=tool_call_dict["name"], tool_args=tool_call_dict["args"]
            )
        )
        msgs = state["messages"] + [tool_msg, prompt]
        ai = self.base_llm.invoke(msgs)
        log.info(f"[THOUGHT_SKIP_TOOL] - {ai.content}")

        state["messages"] = [tool_msg, ai]
        return state

    def _act(self, state: State) -> State:
        ai = self.tools_llm.invoke(state["messages"])
        log.info(f"[ACT] Output: {ai.content}")

        tool_calls = getattr(ai, "tool_calls", None)

        if tool_calls and len(tool_calls) > 1:
            log.warning(f"[ACT] Model returned {len(tool_calls)} tool calls; truncating to 1.")
            ai.tool_calls = tool_calls[:1]

        if tool_calls:
            log.info(f"[ACT] - Tool Calls: {tool_calls}")

        state["messages"] = [ai]
        state["react_iter"] += 1
        return state

    def _finalize(self, state: State) -> State:
        messages = state["messages"] + [
            HumanMessage(content=agent_cfg["finalizer_prompt"].format(problem=state["problem"]))
        ]
        ai = self.base_llm.invoke(messages)
        log.info(f"[FINALIZE] LLM Response: {ai.content}")
        state["messages"] = [ai]
        return state

    def _route_act(self, state: State) -> str:
        if state["react_iter"] >= agent_cfg["max_react_iters"]:
            log.info("[ROUTE] Max react iterations reached")
            return "end"

        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)

        if not tool_calls:
            log.info("[ROUTE] No tool calls, ending.")
            return "end"

        tool_call_dict = tool_calls[0]

        args_key = json.dumps(tool_call_dict["args"], sort_keys=True)
        tool_call_tuple = (tool_call_dict["name"], args_key)

        if tool_call_tuple in self.memory:
            log.info("[ROUTE] Going to skip tool, tool has already been invoked earlier")
            return "skip_tool"

        self.memory.add(tool_call_tuple)
        log.info("[ROUTE] Going to tools")
        return "tools"

    def solve(self, problem: str) -> str:
        log.info(f"[PROBLEM] - {problem}")
        state: State = {
            "messages": [
                SystemMessage(content=agent_cfg["main_system_prompt"]),
                HumanMessage(content=f"## Current Problem\n{problem}"),
            ],
            "problem": problem,
            "react_iter": 0,
        }
        retriever_backend.clear_memory()
        self.memory = set()

        final_state = self.graph.invoke(state, config={"recursion_limit": 200})
        msgs = final_state.get("messages", [])

        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                return msg.content

        return msgs[-1].content if msgs else ""
