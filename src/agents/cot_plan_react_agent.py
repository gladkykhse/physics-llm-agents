import logging as log
from typing import Annotated, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from src.agents.utils.llm import make_llm
from src.agents.utils.plan import PhysicsPlan
from src.agents.utils.tools import retriever, retriever_backend
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/cot_plan_react_agent.yaml")
log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    plan: PhysicsPlan | None
    plan_step: int
    react_iter: int

    # Planner fields
    plan_attempts: int
    plan_messages: Annotated[List[AnyMessage], add_messages]
    plan_last_raw: str
    plan_repair_prompt: str


class COTPhysicsAgent:
    def __init__(self):
        # LLMs
        self.base_llm = make_llm()
        self.tools_llm = make_llm().bind_tools([retriever])

        # Tool Nodes
        self._tool_node = ToolNode([retriever])

        # Graph
        graph = StateGraph(State)

        # Graph Nodes
        graph.add_node("cot_plan", self._cot_plan)
        graph.add_node("plan", self._plan)
        graph.add_node("convert", self._convert)
        graph.add_node("validate", self._validate)
        graph.add_node("execute", self._execute)
        graph.add_node("agent", self._agent)
        graph.add_node("tools", self._tool_node)
        graph.add_node("reflect", self._reflect)
        graph.add_node("finalize", self._finalize)

        # Graph Edges
        graph.set_entry_point("cot_plan")
        graph.add_edge("cot_plan", "plan")
        graph.add_edge("plan", "convert")
        graph.add_edge("convert", "validate")
        graph.add_conditional_edges(
            "validate",
            self._route_validate,
            {"ok": "execute", "repair": "convert"},
        )
        graph.add_conditional_edges(
            "execute",
            self._route_execute,
            {"continue": "agent", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "agent",
            self._route_agent,
            {"tools": "tools", "end": "execute"},
        )
        graph.add_edge("tools", "reflect")
        graph.add_edge("reflect", "agent")
        graph.add_edge("finalize", END)

        self.graph = graph.compile()

    def _cot_plan(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["cot_planner_prompt"])
        msgs = state["plan_messages"] + [prompt]
        ai = self.base_llm.invoke(msgs)
        log.info(f"[PLAN_COT] - {ai.content}")

        state["plan_messages"] = [prompt, ai]
        return state

    def _plan(self, state: State) -> State:
        msgs = state["plan_messages"] + [HumanMessage(content=agent_cfg["planner_prompt"])]
        ai = self.base_llm.invoke(msgs)
        log.info(f"[PLAN] - {ai.content}")

        state["messages"] = [ai]
        state["plan_messages"] = [HumanMessage(content=agent_cfg["planner_prompt"]), ai]
        return state

    def _convert(self, state: State) -> State:
        prompt = state["plan_repair_prompt"] if state["plan_attempts"] > 0 else agent_cfg["converter_prompt"]

        msgs = state["plan_messages"] + [HumanMessage(content=prompt)]
        ai = self.base_llm.invoke(msgs)

        log.info(f"[CONVERT] - {ai.content}")

        state["plan_messages"] = [HumanMessage(content=prompt)] + [ai]
        state["plan_last_raw"] = ai.content

        return state

    def _validate(self, state: State) -> State:
        try:
            state["plan"] = PhysicsPlan.model_validate_json(state["plan_last_raw"])
            log.info("[VALIDATE] - Succeeded")
        except ValidationError as e:
            log.info(f"[VALIDATE] - Failed: {e.json(indent=2)}")
            state["plan_repair_prompt"] = agent_cfg["planner_repair_prompt"].format(errors=e.json(indent=2))
            state["plan_attempts"] += 1

        return state

    def _execute(self, state: State) -> State:
        plan_step = state["plan"].steps[state["plan_step"]]
        log.info(f"[EXECUTE] - Plan step: {plan_step}")
        state["messages"] = [HumanMessage(content=agent_cfg["executor_prompt"].format(plan_step=plan_step))]
        state["plan_step"] += 1
        state["react_iter"] = 0
        return state

    def _agent(self, state: State) -> State:
        ai = self.tools_llm.invoke(state["messages"])
        log.info(f"[AGENT] - Output: {ai.content}")
        state["messages"] = [ai]
        state["react_iter"] += 1
        return state

    def _reflect(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["cot_reflect_prompt"])
        msgs = state["messages"] + [prompt]
        ai = self.base_llm.invoke(msgs)
        log.info(f"[REFLECT_COT] - {ai.content}")

        state["messages"] = [ai]
        return state

    def _finalize(self, state: State) -> State:
        messages = state["messages"] + [HumanMessage(content=agent_cfg["finalizer_prompt"])]
        ai = self.base_llm.invoke(messages)
        log.info(f"[FINALIZE] LLM Response: {ai.content}")
        state["messages"] = [ai]
        return state

    def _route_validate(self, state: State) -> str:
        if state["plan"] is not None:
            return "ok"
        if state["plan_attempts"] >= agent_cfg.get("max_plan_attempts", 5):
            raise RuntimeError("Max plan attempts reached")
        return "repair"

    def _route_execute(self, state: State) -> str:
        if not state["plan"] or state["plan_step"] >= len(state["plan"].steps):
            return "finalize"
        return "continue"

    def _route_agent(self, state: State) -> str:
        last = state["messages"][-1]
        tcs = getattr(last, "tool_calls", None)
        if state["react_iter"] >= agent_cfg["max_react_iters"]:
            return "end"
        elif tcs:
            log.info(f"[AGENT] - Tool Calls: {tcs}")
            return "tools"
        return "end"

    def solve(self, problem: str) -> str:
        state: State = {
            "messages": [
                SystemMessage(content=agent_cfg["main_system_prompt"]),
                HumanMessage(content=f"Problem:\n{problem}"),
            ],
            "plan": None,
            "plan_step": 0,
            "react_iter": 0,
            "plan_attempts": 0,
            "plan_messages": [HumanMessage(content=f"Problem:\n{problem}")],
            "plan_last_raw": "",
            "plan_repair_prompt": "",
        }
        retriever_backend.clear_memory()
        final_state = self.graph.invoke(state, config={"recursion_limit": 200})
        msgs = final_state.get("messages", [])
        return getattr(msgs[-1], "content", "") if msgs else ""
