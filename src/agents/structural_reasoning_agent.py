import logging as log
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages

from src.agents.utils.llm import make_llm
from src.agents.utils.utils import scieval_split_problem_and_options
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/structural_reasoning_agent.yaml")

log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    problem: str
    react_iter: int


class PhysicsReactAgent:
    """
    ReAct-style reasoning agent WITHOUT tools.
    Follows the same Thought -> Act -> Observation loop as the math agent,
    but all actions are internal reasoning steps performed by the LLM itself.
    """

    def __init__(self) -> None:
        self.llm = make_llm(temperature=0.0)

        graph = StateGraph(State)

        graph.add_node("thought", self._thought)
        graph.add_node("act", self._act)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("thought")
        graph.add_conditional_edges(
            "thought",
            self._route_thought,
            {"act": "act", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "act",
            self._route_act,
            {"thought": "thought", "finalize": "finalize"},
        )
        graph.add_edge("finalize", END)

        self.graph = graph.compile()

    # ── nodes ────────────────────────────────────────────────────────────

    def _thought(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["thought_prompt"])
        msgs = state["messages"] + [prompt]
        ai = self.llm.invoke(msgs)
        log.info(f"[THOUGHT] - {ai.content}")

        state["messages"] = [ai]
        return state

    def _act(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["act_prompt"])
        msgs = state["messages"] + [prompt]
        ai = self.llm.invoke(msgs)
        log.info(f"[ACT] - {ai.content}")

        state["messages"] = [ai]
        state["react_iter"] += 1
        return state

    def _finalize(self, state: State) -> State:
        prompt_content = agent_cfg["finalizer_prompt"].format(problem=state["problem"])
        messages = state["messages"] + [HumanMessage(content=prompt_content)]
        ai = self.llm.invoke(messages)
        log.info(f"[FINALIZE] LLM Response: {ai.content}")
        state["messages"] = [ai]
        return state

    # ── routing ──────────────────────────────────────────────────────────

    def _route_thought(self, state: State) -> str:
        last = state["messages"][-1]
        content = (last.content or "").strip()

        if "READY FOR FINAL ANSWER" in content.upper():
            log.info("[ROUTE_THOUGHT] Completion marker detected, going to finalize.")
            return "finalize"

        log.info("[ROUTE_THOUGHT] No completion marker, going to act.")
        return "act"

    def _route_act(self, state: State) -> str:
        if state["react_iter"] >= agent_cfg["max_react_iters"]:
            log.info("[ROUTE_ACT] Max react iterations reached, going to finalize.")
            return "finalize"

        log.info("[ROUTE_ACT] Continuing to next thought.")
        return "thought"

    # ── public api ───────────────────────────────────────────────────────

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
