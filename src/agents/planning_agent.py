import logging as log
import re
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages

from src.agents.utils.llm import make_llm
from src.agents.utils.utils import scieval_split_problem_and_options
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/planning_agent.yaml")

log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    problem: str
    analysis: str
    plan: List[str]
    current_step: int
    plan_fix_iter: int
    last_plan_output: str


class PhysicsReactAgent:
    """
    Plan-and-Execute agent WITHOUT tools.

    Workflow:
      1. Analyze  – free-form reasoning about the problem (stored in state,
                     NOT pushed to main message history).
      2. Plan     – structured numbered plan built from the analysis.
                     Uses a separate planning context; only the final clean
                     plan is injected into main messages.
      3. Fix Plan – if the plan could not be parsed, retry with feedback.
      4. Execute  – carry out each plan step sequentially.
      5. Finalize – synthesize a final answer from step results.
    """

    def __init__(self) -> None:
        self.llm = make_llm(temperature=0.0)

        graph = StateGraph(State)

        graph.add_node("analyze", self._analyze)
        graph.add_node("plan", self._plan)
        graph.add_node("fix_plan", self._fix_plan)
        graph.add_node("execute", self._execute)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("analyze")
        graph.add_edge("analyze", "plan")
        graph.add_conditional_edges(
            "plan",
            self._route_plan,
            {"execute": "execute", "fix_plan": "fix_plan", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "fix_plan",
            self._route_plan,
            {"execute": "execute", "fix_plan": "fix_plan", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "execute",
            self._route_execute,
            {"execute": "execute", "finalize": "finalize"},
        )
        graph.add_edge("finalize", END)

        self.graph = graph.compile()

    # ── nodes ────────────────────────────────────────────────────────────

    def _analyze(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["analyze_prompt"])
        msgs = state["messages"] + [prompt]
        ai = self.llm.invoke(msgs)
        log.info(f"[ANALYZE] - {ai.content}")

        state["analysis"] = ai.content
        state["messages"] = []
        return state

    def _plan(self, state: State) -> State:
        prompt_content = agent_cfg["plan_prompt"].format(
            analysis=state["analysis"],
        )
        planning_msgs = [
            SystemMessage(content=agent_cfg["main_system_prompt"]),
            HumanMessage(content=f"# Problem\n{state['problem']}"),
            HumanMessage(content=prompt_content),
        ]

        ai = self.llm.invoke(planning_msgs)
        log.info(f"[PLAN] - {ai.content}")

        steps = self._parse_plan(ai.content)
        log.info(f"[PLAN] Parsed {len(steps)} steps: {steps}")

        state["plan"] = steps
        state["last_plan_output"] = ai.content
        state["plan_fix_iter"] = 0
        state["current_step"] = 0

        if steps:
            self._commit_plan(state, steps)

        return state

    def _fix_plan(self, state: State) -> State:
        prompt_content = agent_cfg["fix_plan_prompt"].format(
            analysis=state["analysis"],
            failed_output=state["last_plan_output"],
        )
        planning_msgs = [
            SystemMessage(content=agent_cfg["main_system_prompt"]),
            HumanMessage(content=f"# Problem\n{state['problem']}"),
            HumanMessage(content=prompt_content),
        ]

        ai = self.llm.invoke(planning_msgs)
        state["plan_fix_iter"] += 1
        log.info(f"[FIX_PLAN] Attempt {state['plan_fix_iter']} - {ai.content}")

        steps = self._parse_plan(ai.content)
        state["last_plan_output"] = ai.content

        if steps:
            state["plan"] = steps
            self._commit_plan(state, steps)
            log.info(f"[FIX_PLAN] Parsed {len(steps)} steps: {steps}")
        elif state["plan_fix_iter"] >= agent_cfg["max_plan_fix_iters"]:
            state["plan"] = [ai.content.strip()]
            self._commit_plan(state, state["plan"])
            log.warning("[FIX_PLAN] Max fix attempts reached, falling back to single step.")

        return state

    def _execute(self, state: State) -> State:
        step_idx = state["current_step"]
        step_desc = state["plan"][step_idx]

        prompt_content = agent_cfg["execute_prompt"].format(
            step_number=step_idx + 1,
            total_steps=len(state["plan"]),
            step_description=step_desc,
        )
        prompt = HumanMessage(content=prompt_content)
        msgs = state["messages"] + [prompt]
        ai = self.llm.invoke(msgs)
        log.info(f"[EXECUTE step {step_idx + 1}] - {ai.content}")

        state["messages"] = [ai]
        state["current_step"] = step_idx + 1
        return state

    def _finalize(self, state: State) -> State:
        prompt_content = agent_cfg["finalizer_prompt"].format(problem=state["problem"])
        messages = state["messages"] + [HumanMessage(content=prompt_content)]
        ai = self.llm.invoke(messages)
        log.info(f"[FINALIZE] - {ai.content}")
        state["messages"] = [ai]
        return state

    # ── routing ──────────────────────────────────────────────────────────

    def _route_plan(self, state: State) -> str:
        if state["plan"]:
            log.info(f"[ROUTE_PLAN] Plan ready with {len(state['plan'])} steps.")
            return "execute"

        if state["plan_fix_iter"] >= agent_cfg["max_plan_fix_iters"]:
            log.warning("[ROUTE_PLAN] Max fixes exhausted with empty plan, skipping to finalize.")
            return "finalize"

        log.info(f"[ROUTE_PLAN] Plan parsing failed, routing to fix attempt {state['plan_fix_iter'] + 1}.")
        return "fix_plan"

    def _route_execute(self, state: State) -> str:
        if state["current_step"] >= len(state["plan"]):
            log.info("[ROUTE_EXECUTE] All steps complete, going to finalize.")
            return "finalize"

        if state["current_step"] >= agent_cfg["max_execute_steps"]:
            log.info("[ROUTE_EXECUTE] Max execution steps reached, going to finalize.")
            return "finalize"

        log.info(f"[ROUTE_EXECUTE] Continuing to step {state['current_step'] + 1}.")
        return "execute"

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _commit_plan(state: State, steps: list) -> None:
        """Inject the clean plan into main message history."""
        plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        state["messages"] = [AIMessage(content=plan_text)]

    @staticmethod
    def _parse_plan(text: str) -> List[str]:
        """Parse numbered list from LLM output. Returns empty list on failure."""
        lines = re.findall(r"^\s*\d+\.\s*(.+)$", text, re.MULTILINE)
        return [line.strip() for line in lines if line.strip()]

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
            "analysis": "",
            "plan": [],
            "current_step": 0,
            "plan_fix_iter": 0,
            "last_plan_output": "",
        }

        final_state = self.graph.invoke(state, config={"recursion_limit": 200})
        msgs = final_state.get("messages", [])

        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                return msg.content

        return msgs[-1].content if msgs else ""
