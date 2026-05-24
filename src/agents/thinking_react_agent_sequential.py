import json
import logging as log
from typing import Annotated, List, TypedDict

from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode

from src.agents.utils.llm import make_llm
from src.agents.utils.tools import (sympy_eval, sympy_solve, vector_math,
                                    wikipedia_multi_search)
from src.agents.utils.utils import scieval_split_problem_and_options
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/thinking_react_agent_sequential.yaml")

log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class State(TypedDict):
    knowledge_messages: Annotated[List[AnyMessage], add_messages]
    messages: Annotated[List[AnyMessage], add_messages]
    problem: str
    react_iter: int
    problem_type: str


class PhysicsReactAgent:
    def __init__(self) -> None:
        knowledge_tools_list = [wikipedia_multi_search]
        math_tools_list = [sympy_eval, vector_math, sympy_solve]

        self.base_llm = make_llm(temperature=0.0)
        self.knowledge_tools_llm = make_llm(temperature=0.0).bind_tools(knowledge_tools_list)
        self.math_tools_llm = make_llm(temperature=0.0).bind_tools(math_tools_list)

        graph = StateGraph(State)
        self.knowledge_tools = ToolNode(knowledge_tools_list, messages_key="knowledge_messages")
        self.math_tools = ToolNode(math_tools_list, messages_key="messages")

        graph.add_node("classify_problem", self._classify_problem)
        graph.add_node("cot_retrieve_knowledge", self._cot_retrieve_knowledge)
        graph.add_node("retrieve_knowledge", self._retrieve_knowledge)
        graph.add_node("knowledge_tools", self.knowledge_tools)
        graph.add_node("filter_knowledge", self._filter_knowledge)
        graph.add_node("solve_problem", self._solve_problem)

        graph.add_node("thought", self._thought)
        graph.add_node("agent", self._act)
        graph.add_node("tools", self.math_tools)
        graph.add_node("finalize", self._finalize)
        graph.add_node("thought_skip_tool", self._thought_skip_tool)

        graph.set_entry_point("classify_problem")
        graph.add_edge("classify_problem", "cot_retrieve_knowledge")
        graph.add_edge("cot_retrieve_knowledge", "retrieve_knowledge")
        graph.add_edge("retrieve_knowledge", "knowledge_tools")
        graph.add_edge("knowledge_tools", "filter_knowledge")
        graph.add_conditional_edges(
            "filter_knowledge",
            self._route_problem_type,
            {"math": "thought", "theory": "solve_problem"},
        )
        graph.add_conditional_edges(
            "thought",
            self._route_thought,
            {"agent": "agent", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "agent",
            self._route_act,
            {"tools": "tools", "skip_tool": "thought_skip_tool", "end": "finalize"},
        )
        graph.add_conditional_edges(
            "thought_skip_tool",
            self._route_thought,
            {"agent": "agent", "finalize": "finalize"},
        )
        graph.add_edge("tools", "thought")
        graph.add_edge("solve_problem", END)
        graph.add_edge("finalize", END)

        self.graph = graph.compile()

    def _classify_problem(self, state: State) -> State:
        prompt = agent_cfg["problem_type_router_prompt"].format(problem=state["problem"])
        ai = self.base_llm.invoke([HumanMessage(content=prompt)])
        text = (ai.content or "").upper()

        if "ANSWER: THEORETICAL" in text:
            problem_type = "theory"
        elif "ANSWER: MATHEMATICAL" in text:
            problem_type = "math"
        else:
            problem_type = "math"

        log.info(f"[CLASSIFY] - {problem_type} | raw: {ai.content}")
        state["problem_type"] = problem_type

        if problem_type == "math":
            state["messages"] = [
                SystemMessage(content=agent_cfg["math_system_prompt"]),
                HumanMessage(content=f"# Problem\n{state['problem']}"),
            ]
        elif problem_type == "theory":
            state["messages"] = [
                SystemMessage(content=agent_cfg["theory_system_prompt"]),
                HumanMessage(content=f"# Problem\n{state['problem']}"),
            ]

        return state

    def _cot_retrieve_knowledge(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["cot_retrieve_knowledge_prompt"].format(problem=state["problem"]))
        msgs = state["knowledge_messages"] + [prompt]
        ai = self.base_llm.invoke(msgs)
        log.info(f"[COT_RETRIEVE_KNOWLEDGE] - Output: {ai.content}")

        state["knowledge_messages"] = [ai]
        return state

    def _retrieve_knowledge(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["retrieve_knowledge_prompt"])
        msgs = state["knowledge_messages"] + [prompt]
        ai = self.knowledge_tools_llm.invoke(msgs)

        log.info(f"[RETRIEVE_KNOWLEDGE] - Output: {ai.content}")

        tool_calls = getattr(ai, "tool_calls", None)
        if tool_calls:
            log.info(f"[RETRIEVE_KNOWLEDGE] - Tool Calls: {tool_calls}")

        state["knowledge_messages"] = [ai]
        return state

    def _filter_knowledge(self, state: State) -> State:
        knowledge_results = []
        for msg in reversed(state["knowledge_messages"]):
            if isinstance(msg, ToolMessage):
                knowledge_results.append(msg.content)

        knowledge_str = "\n\n".join(reversed(knowledge_results))

        # =======================================================
        # Uncomment for LLM-based filtering of retrieved sections
        prompt = HumanMessage(
            content=agent_cfg["filter_knowledge_prompt"].format(problem=state["problem"], knowledge=knowledge_str)
        )
        ai = self.base_llm.invoke([prompt])

        filtered_knowledge = "# Wikipedia Search Results\n\n" + ai.content
        log.info(f"[FILTER_KNOWLEDGE] - Output: {filtered_knowledge}")
        # =======================================================

        # state["messages"] = [AIMessage(content=knowledge_str)]

        # =======================================================
        # Uncomment for LLM-based filtering of retrieved sections
        state["messages"] = [HumanMessage(content=filtered_knowledge)]
        # =======================================================

        return state

    def _solve_problem(self, state: State) -> State:
        prompt = HumanMessage(
            content=agent_cfg["solve_prompt"].format(problem=state["problem"])
        )  # , options=state["options"]))
        msgs = state["messages"] + [prompt]
        ai = self.base_llm.invoke(msgs)

        log.info(f"[SOLVE_PROBLEM] - Output: {ai.content}")
        state["messages"] = [ai]
        return state

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
        ai = self.math_tools_llm.invoke(state["messages"])
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
        prompt_content = agent_cfg["finalizer_prompt"].format(problem=state["problem"])  # , options=state["options"])
        messages = state["messages"] + [HumanMessage(content=prompt_content)]
        ai = self.base_llm.invoke(messages)
        log.info(f"[FINALIZE] LLM Response: {ai.content}")
        state["messages"] = [ai]
        return state

    def _route_problem_type(self, state: State) -> str:
        return state["problem_type"]

    def _route_thought(self, state: State) -> str:
        last = state["messages"][-1]
        content = (last.content or "").strip()

        if "READY FOR FINAL ANSWER" in content.upper():
            log.info("[ROUTE_THOUGHT] Completion marker detected, going to finalize.")
            return "finalize"

        log.info("[ROUTE_THOUGHT] No completion marker, going to agent.")
        return "agent"

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
        question, options = scieval_split_problem_and_options(full_text=problem)
        log.info(f"[QUESTION] - {question}")
        log.info(f"[OPTIONS] - {options}")

        state: State = {
            "knowledge_messages": [],
            "messages": [],
            "problem": problem,
            "react_iter": 0,
            "problem_type": "",
        }

        self.memory = set()

        final_state = self.graph.invoke(state, config={"recursion_limit": 200})
        msgs = final_state.get("messages", [])

        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                return msg.content

        return msgs[-1].content if msgs else ""
