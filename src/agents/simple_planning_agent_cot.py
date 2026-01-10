import logging as log
from typing import Annotated, List, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from pydantic import ValidationError

from src.agents.utils.llm import make_llm
from src.agents.utils.plan import PhysicsPlanV2
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/simple_planning_agent_cot_v2.yaml")
log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class State(TypedDict):
    problem: str
    messages: Annotated[List[AnyMessage], add_messages]
    plan: PhysicsPlanV2 | None
    plan_step: int
    agent_prompt: str | None

    # Planner fields
    plan_attempts: int
    plan_messages: Annotated[List[AnyMessage], add_messages]
    plan_last_raw: str
    plan_repair_prompt: str


class SimplePlanningPhysicsAgent:
    def __init__(self):
        # LLMs
        self.base_llm = make_llm()

        # Graph
        graph = StateGraph(State)

        # Graph Nodes
        graph.add_node("cot_plan", self._cot_plan)
        graph.add_node("plan", self._plan)
        graph.add_node("convert", self._convert)
        graph.add_node("validate", self._validate)
        graph.add_node("execute", self._execute)
        graph.add_node("agent", self._agent)
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
        graph.add_edge("execute", "agent")
        graph.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"continue": "execute", "finalize": "finalize"},
        )
        graph.add_edge("finalize", END)

        self.graph = graph.compile()

    def _cot_plan(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["cot_planner_prompt"])
        msgs = state["plan_messages"] + [prompt]
        ai = self.base_llm.invoke(msgs)
        log.info(f"[PLAN_COT] - {ai.content}")

        state["plan_messages"] = [ai]
        return state

    def _plan(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["planner_prompt"])
        msgs = state["plan_messages"] + [prompt]
        ai = self.base_llm.invoke(msgs)
        log.info(f"[PLAN] - {ai.content}")

        state["messages"] = [ai]
        state["plan_messages"] = [prompt, ai]
        return state

    def _convert(self, state: State) -> State:
        prompt = state["plan_repair_prompt"] if state["plan_attempts"] > 0 else agent_cfg["converter_prompt"]

        msgs = state["plan_messages"] + [HumanMessage(content=prompt)]
        ai = self.base_llm.invoke(msgs)

        log.info(f"[CONVERT] - {ai.content}")

        state["plan_messages"] = [prompt, ai]
        state["plan_last_raw"] = ai.content

        return state

    def _validate(self, state: State) -> State:
        try:
            state["plan"] = PhysicsPlanV2.model_validate_json(state["plan_last_raw"])
            log.info("[VALIDATE] - Succeeded")
        except ValidationError as e:
            log.info(f"[VALIDATE] - Failed: {e.json(indent=2)}")
            state["plan_repair_prompt"] = agent_cfg["planner_repair_prompt"]
            # .format(
            #     errors=e.json(indent=2),
            #     previous_json=state["plan_last_raw"],
            # )
            state["plan_attempts"] += 1

        return state

    def _execute(self, state: State) -> State:
        plan_element = state["plan"].plan[state["plan_step"]]
        log.info(f"[EXECUTE] - Plan step: {plan_element.step}")
        state["agent_prompt"] = agent_cfg["executor_prompt"].format(
            plan_step=plan_element.step,
            rationale=plan_element.rationale,
            goal=plan_element.goal,
        )
        return state

    def _agent(self, state: State) -> State:
        prompt = HumanMessage(content=state["agent_prompt"])
        msgs = state["messages"] + [prompt]
        ai = self.base_llm.invoke(msgs)
        log.info(f"[AGENT] - Output: {ai.content}")
        state["messages"] = [ai]
        state["plan_step"] += 1
        return state

    def _finalize(self, state: State) -> State:
        messages = state["messages"] + [
            HumanMessage(content=agent_cfg["finalizer_prompt"].format(problem=state["problem"]))
        ]
        ai = self.base_llm.invoke(messages)
        log.info(f"[FINALIZE] LLM Response: {ai.content}")
        state["messages"] = [ai]
        return state

    def _route_validate(self, state: State) -> str:
        if state["plan"] is not None:
            return "ok"
        if state["plan_attempts"] >= agent_cfg["max_plan_attempts"]:
            raise RuntimeError("Max plan attempts reached")
        return "repair"

    def _route_after_agent(self, state: State) -> str:
        if not state["plan"] or state["plan_step"] >= len(state["plan"].plan):
            return "finalize"
        return "continue"

    def solve(self, problem: str) -> str:
        state: State = {
            "problem": problem,
            "messages": [HumanMessage(content=problem)],
            "plan": None,
            "plan_step": 0,
            "agent_prompt": None,
            "plan_attempts": 0,
            "plan_messages": [HumanMessage(content=problem)],
            "plan_last_raw": "",
            "plan_repair_prompt": "",
        }
        final_state = self.graph.invoke(state, config={"recursion_limit": 300})
        msgs = final_state.get("messages", [])
        return getattr(msgs[-1], "content", "") if msgs else ""


a = [
    {
        "step": "Identify the fundamental aspects of velocity",
        "rationale": "Determine the basic characteristics of velocity that make it a useful quantity in physics.",
        "goal": "Understand the essential features of velocity.",
    },
    {
        "step": "Determine the relationship between velocity and motion",
        "rationale": "Analyze how velocity relates to an object's motion, including its speed and direction.",
        "goal": "Establish a connection between velocity and motion.",
    },
    {
        "step": "Identify the components of velocity",
        "rationale": "Based on the relationship between velocity and motion, determine the two fundamental aspects that make up velocity.",
        "goal": "Determine the two components of velocity.",
    },
]
