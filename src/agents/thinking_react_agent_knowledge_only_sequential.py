import logging as log
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode

from src.agents.utils.llm import make_llm
from src.agents.utils.tools import wikipedia_multi_search
from src.agents.utils.utils import scieval_split_problem_and_options
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/thinking_react_agent_knowledge_only_sequential_v2.yaml")

log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class State(TypedDict):
    knowledge_messages: Annotated[List[AnyMessage], add_messages]
    messages: Annotated[List[AnyMessage], add_messages]
    problem: str
    # options: str
    react_iter: int


class PhysicsReactAgent:
    def __init__(self) -> None:
        knowledge_tools_list = [wikipedia_multi_search]

        self.base_llm = make_llm(temperature=0.0)
        self.knowledge_tools_llm = make_llm(temperature=0.0).bind_tools(knowledge_tools_list)

        graph = StateGraph(State)
        self.knowledge_tools = ToolNode(knowledge_tools_list, messages_key="knowledge_messages")

        graph.add_node("cot_retrieve_knowledge", self._cot_retrieve_knowledge)
        graph.add_node("retrieve_knowledge", self._retrieve_knowledge)
        graph.add_node("knowledge_tools", self.knowledge_tools)
        graph.add_node("filter_knowledge", self._filter_knowledge)
        graph.add_node("solve_problem", self._solve_problem)

        graph.set_entry_point("cot_retrieve_knowledge")
        graph.add_edge("cot_retrieve_knowledge", "retrieve_knowledge")
        graph.add_edge("retrieve_knowledge", "knowledge_tools")
        graph.add_edge("knowledge_tools", "filter_knowledge")
        graph.add_edge("filter_knowledge", "solve_problem")
        graph.add_edge("solve_problem", END)

        self.graph = graph.compile()

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

        # Uncomment for LLM-based filtering of retrieved sections
        # prompt = HumanMessage(
        #     content=agent_cfg["filter_knowledge_prompt"].format(problem=state["problem"], knowledge=knowledge_str))
        # ai = self.base_llm.invoke([prompt])
        #
        # filtered_knowledge = "# Wikipedia Search Results\n\n" + ai.content
        # log.info(f"[FILTER_KNOWLEDGE] - Output: {filtered_knowledge}")

        state["messages"] = [AIMessage(content=knowledge_str)]

        # Uncomment for LLM-based filtering of retrieved sections
        # state["messages"] = [AIMessage(content=filtered_knowledge)]

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

    def solve(self, problem: str) -> str:
        question, options = scieval_split_problem_and_options(full_text=problem)
        log.info(f"[QUESTION] - {question}")
        log.info(f"[OPTIONS] - {options}")

        state: State = {
            "knowledge_messages": [],
            "messages": [SystemMessage(content=agent_cfg["main_system_prompt"])],
            "problem": problem,
            # "options": options,
            "react_iter": 0,
        }

        final_state = self.graph.invoke(state, config={"recursion_limit": 200})
        msgs = final_state.get("messages", [])

        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                return msg.content

        return msgs[-1].content if msgs else ""
