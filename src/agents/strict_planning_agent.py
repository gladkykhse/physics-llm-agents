import logging as log
from typing import Any, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from src.agents.utils.llm import make_llm
from src.agents.utils.plan import PlanModel, render_plan_text
from src.agents.utils.tools import finalize_solution, retriever
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/strict_planning_agent.yaml")
TOOLS = [retriever, finalize_solution]
log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class State(dict):
    # Main agent's fields
    messages: List[Any]
    plan: PlanModel | None
    iters: int

    # Planner fields
    repair_attempts: int
    plan_refinement_messages: List[Any]
    repair_prompt: str | None
    last_raw_plan: str | None


class PhysicsAgent:
    def __init__(self):
        self.base_llm = make_llm()
        self.tools_llm = make_llm().bind_tools(TOOLS)
        self.tool_node = ToolNode(TOOLS)

        g = StateGraph(State)
        g.add_node("plan", self._plan)
        g.add_node("validate", self._validate)
        g.add_node("execute", self._execute)
        g.add_node("tools", self.tool_node)
        g.add_node("finalize", self._finalize)

        g.set_entry_point("plan")
        g.add_edge("plan", "validate")
        g.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {"ok": "execute", "repair": "plan"},
        )
        g.add_conditional_edges(
            "execute", self._should_continue, {"tools": "tools", "finalize": "finalize", "continue": "execute"}
        )
        g.add_edge("tools", "execute")
        g.add_edge("finalize", END)

        self.app = g.compile()

    def _plan(self, state: State) -> State:
        """Generate or repair the plan (strict JSON)."""
        attempt = state.get("repair_attempts", 0)
        if attempt == 0 or not state.get("repair_prompt"):
            sys = SystemMessage(content=agent_cfg["planner_prompt"])
            log.info(f"[PLAN] Attempt {attempt + 1}")
        else:
            sys = SystemMessage(content=state["repair_prompt"])
            log.info(f"[PLAN] Repair attempt {attempt + 1}")

        msgs = state["plan_refinement_messages"] + [sys]
        ai = self.base_llm.invoke(msgs)
        raw = ai.content or ""
        log.info(f"[PLAN] Raw LLM output:\n{raw}")

        state["last_raw_plan"] = raw
        state["plan_refinement_messages"] = msgs + [ai]
        # DO NOT touch state["iters"] here.
        state["plan"] = None
        return state

    def _validate(self, state: State) -> State:
        """Parse with Pydantic; if invalid, craft a repair prompt that includes errors + previous JSON."""
        last_ai = [m for m in state["plan_refinement_messages"] if isinstance(m, AIMessage)][-1]
        raw = state.get("last_raw_plan") or last_ai.content or ""
        attempt = state.get("repair_attempts", 0)

        try:
            plan = PlanModel.model_validate_json(raw)
            state["plan"] = plan
            log.info("[VALIDATE] Plan validated successfully")

            summary = render_plan_text(plan)
            state["messages"] = state["messages"] + [SystemMessage(content=summary)]
            return state

        except ValidationError as e:
            log.warning(f"[VALIDATE] Plan invalid\n{e}")
            if attempt >= agent_cfg["plain_repair_max_iters"]:
                raise RuntimeError("Planner failed after multiple repairs.")

            # Build a helpful repair prompt with errors and the previous JSON
            repair_prompt = agent_cfg["planner_repair_prompt"].format(
                errors=e.json(indent=2), previous_json=raw.strip()
            )
            state["repair_prompt"] = repair_prompt
            state["repair_attempts"] = attempt + 1

            return state

    def _execute(self, state: State) -> State:
        steps = state["plan"].steps
        idx = state.get("iters", 0)

        if idx >= len(steps):
            log.info("[EXECUTE] All steps done; ready to finalize")
            return state

        current_plan_item = steps[idx]
        log.info(
            f"[EXECUTE] Step index {idx} (id={current_plan_item.id}, action={current_plan_item.action}, goal={current_plan_item.goal})"
        )

        if current_plan_item.action == "reason":
            reasoner_prompt = agent_cfg["reasoner_prompt"].format(
                plan_step=f"{current_plan_item.id}. {current_plan_item.action}: {current_plan_item.goal}"
            )
            messages = state["messages"] + [SystemMessage(content=reasoner_prompt)]
            ai_msg = self.base_llm.invoke(messages)
            log.info(f"[EXECUTE] LLM Response:\n{ai_msg.content}")

            state["messages"] = state["messages"] + [ai_msg]
            state["iters"] = idx + 1
            return state

        elif current_plan_item.action == "act":
            actor_prompt = agent_cfg["actor_prompt"].format(
                plan_step=f"{current_plan_item.id}. {current_plan_item.action}: {current_plan_item.goal}"
            )
            messages = state["messages"] + [SystemMessage(content=actor_prompt)]
            ai_msg = self.tools_llm.invoke(messages)
            log.info(f"[EXECUTE] LLM Response:\n{ai_msg.content}")

            # Robust tool-call logging across LangChain versions
            tcs = getattr(ai_msg, "tool_calls", None)
            if tcs:
                try:
                    names = [getattr(tc, "name", tc.get("name")) for tc in tcs]
                except Exception:
                    names = ["<unknown>"]
                log.info(f"[EXECUTE] Tool calls: {names}")

            state["messages"] = state["messages"] + [ai_msg]
            state["iters"] = idx + 1
            return state

        else:
            raise ValueError(f"Plan step action cannot be equal to `{current_plan_item.action}`.")

    def _finalize(self, state: State) -> State:
        log.info("[FINALIZE] Generating final answer")

        messages = state["messages"] + [SystemMessage(content=agent_cfg["finalizer_prompt"])]
        ai_msg = self.base_llm.invoke(messages)
        log.info(f"[FINALIZE] LLM Response:\n{ai_msg.content}")

        state["messages"] = state["messages"] + [ai_msg]
        return state

    def _should_continue(self, state: State) -> str:
        if state.get("iters", 0) >= len(state["plan"].steps):
            return "finalize"

        last_msg = state["messages"][-1]
        if getattr(last_msg, "tool_calls", None):
            return "tools"

        log.info("[ROUTE] No tool calls - continuing execution")
        return "continue"

    def _route_after_validate(self, state: State) -> str:
        return "ok" if state.get("plan") else "repair"

    def solve(self, user_text: str) -> str:
        log.info(f"[START] Planning for problem:\n{user_text}")
        first_msgs = [SystemMessage(content=agent_cfg["main_system_prompt"]), HumanMessage(content=user_text)]
        init = State(
            {
                "messages": first_msgs,
                "plan_refinement_messages": first_msgs,
                "iters": 0,
                "repair_attempts": 0,
                "plan": None,
                "repair_prompt": None,
                "last_raw_plan": None,
            }
        )
        final_state = self.app.invoke(init)

        final_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
        if final_messages:
            final_answer = final_messages[-1].content
            return final_answer
        return "Error: No solution generated"
