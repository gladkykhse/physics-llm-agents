import logging as log
import re
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages

from src.agents.utils.llm import make_llm
from src.agents.utils.utils import scieval_split_problem_and_options
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/replanning_agent.yaml")

log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]  # only for final output
    problem: str
    analysis: str
    plan: List[str]
    step_results: List[str]  # no reducer — plain replacement semantics
    current_step: int
    plan_fix_iter: int
    last_plan_output: str
    review_feedback: str
    replan_count: int


class PhysicsReactAgent:
    """
    Plan-and-Execute agent with review and single-retry replanning.

    Context management:
      - All LLM calls build context from state fields (not from messages).
      - `messages` is only written to by finalize for output.
      - `step_results` is a plain list (no reducer) so it can be cleared on replan.
      - On replan: step_results=[], current_step=0, new plan — fully clean context.

    Workflow:
      1. Analyze  – free-form reasoning about the problem.
      2. Plan     – structured numbered plan from analysis.
      3. Fix Plan – retry if plan could not be parsed.
      4. Execute  – carry out each plan step sequentially.
      5. Review   – lightweight check: does the result match an answer choice?
      6. Replan   – if review fails, produce a fresh plan (max 1 replan).
      7. Finalize – synthesize a final answer.
    """

    def __init__(self) -> None:
        self.llm = make_llm(temperature=0.0, max_tokens=1024)
        self.review_llm = make_llm(temperature=0.0, max_tokens=256)

        graph = StateGraph(State)

        graph.add_node("analyze", self._analyze)
        graph.add_node("plan", self._plan)
        graph.add_node("fix_plan", self._fix_plan)
        graph.add_node("execute", self._execute)
        graph.add_node("review", self._review)
        graph.add_node("replan", self._replan)
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
            {"execute": "execute", "review": "review", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "review",
            self._route_review,
            {"finalize": "finalize", "replan": "replan"},
        )
        graph.add_edge("replan", "execute")
        graph.add_edge("finalize", END)

        self.graph = graph.compile()

    # ── context builder ──────────────────────────────────────────────────

    def _build_exec_context(self, state: State) -> list:
        """
        Build execution context from state fields.
        Produces: system + problem + plan + step_results[0..N].
        Used by execute, review, and finalize.
        """
        plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(state["plan"]))
        msgs = [
            SystemMessage(content=agent_cfg["main_system_prompt"]),
            HumanMessage(content=f"# Problem\n{state['problem']}"),
            AIMessage(content=plan_text),
        ]
        for result in state["step_results"]:
            msgs.append(AIMessage(content=result))
        return msgs

    # ── nodes ────────────────────────────────────────────────────────────

    def _analyze(self, state: State) -> State:
        msgs = [
            SystemMessage(content=agent_cfg["main_system_prompt"]),
            HumanMessage(content=f"# Problem\n{state['problem']}"),
            HumanMessage(content=agent_cfg["analyze_prompt"]),
        ]
        ai = self.llm.invoke(msgs)
        log.info(f"[ANALYZE] - {ai.content}")

        state["analysis"] = ai.content
        state["messages"] = []
        return state

    def _plan(self, state: State) -> State:
        prompt_content = agent_cfg["plan_prompt"].format(
            analysis=state["analysis"],
        )
        msgs = [
            SystemMessage(content=agent_cfg["main_system_prompt"]),
            HumanMessage(content=f"# Problem\n{state['problem']}"),
            HumanMessage(content=prompt_content),
        ]

        ai = self.llm.invoke(msgs)
        log.info(f"[PLAN] - {ai.content}")

        steps = self._parse_plan(ai.content)
        log.info(f"[PLAN] Parsed {len(steps)} steps: {steps}")

        state["plan"] = steps
        state["last_plan_output"] = ai.content
        state["plan_fix_iter"] = 0
        state["current_step"] = 0
        state["step_results"] = []
        state["messages"] = []
        return state

    def _fix_plan(self, state: State) -> State:
        prompt_content = agent_cfg["fix_plan_prompt"].format(
            analysis=state["analysis"],
            failed_output=state["last_plan_output"],
        )
        msgs = [
            SystemMessage(content=agent_cfg["main_system_prompt"]),
            HumanMessage(content=f"# Problem\n{state['problem']}"),
            HumanMessage(content=prompt_content),
        ]

        ai = self.llm.invoke(msgs)
        state["plan_fix_iter"] += 1
        log.info(f"[FIX_PLAN] Attempt {state['plan_fix_iter']} - {ai.content}")

        steps = self._parse_plan(ai.content)
        state["last_plan_output"] = ai.content

        if steps:
            state["plan"] = steps
            log.info(f"[FIX_PLAN] Parsed {len(steps)} steps: {steps}")
        elif state["plan_fix_iter"] >= agent_cfg["max_plan_fix_iters"]:
            state["plan"] = [ai.content.strip()]
            log.warning("[FIX_PLAN] Max fix attempts reached, falling back to single step.")

        state["messages"] = []
        return state

    def _execute(self, state: State) -> State:
        step_idx = state["current_step"]
        step_desc = state["plan"][step_idx]

        prompt_content = agent_cfg["execute_prompt"].format(
            step_number=step_idx + 1,
            total_steps=len(state["plan"]),
            step_description=step_desc,
        )
        msgs = self._build_exec_context(state) + [HumanMessage(content=prompt_content)]
        ai = self.llm.invoke(msgs)
        log.info(f"[EXECUTE step {step_idx + 1}] - {ai.content}")

        state["step_results"] = state["step_results"] + [ai.content]
        state["current_step"] = step_idx + 1
        state["messages"] = []
        return state

    def _review(self, state: State) -> State:
        msgs = self._build_exec_context(state) + [HumanMessage(content=agent_cfg["review_prompt"])]
        ai = self.review_llm.invoke(msgs)
        log.info(f"[REVIEW] - {ai.content}")

        state["review_feedback"] = ai.content
        state["messages"] = []
        return state

    def _replan(self, state: State) -> State:
        previous_plan = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(state["plan"]))
        prompt_content = agent_cfg["replan_prompt"].format(
            analysis=state["analysis"],
            previous_plan=previous_plan,
            review_feedback=state["review_feedback"],
        )
        msgs = [
            SystemMessage(content=agent_cfg["main_system_prompt"]),
            HumanMessage(content=f"# Problem\n{state['problem']}"),
            HumanMessage(content=prompt_content),
        ]

        ai = self.llm.invoke(msgs)
        log.info(f"[REPLAN] - {ai.content}")

        steps = self._parse_plan(ai.content)
        if not steps:
            steps = [ai.content.strip()]
            log.warning("[REPLAN] Could not parse plan, falling back to single step.")
        log.info(f"[REPLAN] New plan with {len(steps)} steps: {steps}")

        state["plan"] = steps
        state["step_results"] = []  # clear — fresh execution context
        state["current_step"] = 0
        state["replan_count"] = state["replan_count"] + 1
        state["messages"] = []
        return state

    def _finalize(self, state: State) -> State:
        prompt_content = agent_cfg["finalizer_prompt"].format(problem=state["problem"])
        msgs = self._build_exec_context(state) + [HumanMessage(content=prompt_content)]
        ai = self.llm.invoke(msgs)
        log.info(f"[FINALIZE] - {ai.content}")
        state["messages"] = [ai]  # only place we write to messages
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
            if state["replan_count"] == 0:
                log.info("[ROUTE_EXECUTE] All steps complete, going to review.")
                return "review"
            else:
                log.info("[ROUTE_EXECUTE] All steps complete (after replan), going to finalize.")
                return "finalize"

        if state["current_step"] >= agent_cfg["max_execute_steps"]:
            log.info("[ROUTE_EXECUTE] Max execution steps reached, going to finalize.")
            return "finalize"

        log.info(f"[ROUTE_EXECUTE] Continuing to step {state['current_step'] + 1}.")
        return "execute"

    def _route_review(self, state: State) -> str:
        feedback = (state.get("review_feedback") or "").upper()

        has_revise = "VERDICT: REVISE" in feedback
        has_pass = "VERDICT: PASS" in feedback

        if has_revise and not has_pass:
            if state["replan_count"] < agent_cfg["max_replans"]:
                log.info("[ROUTE_REVIEW] Review flagged revision, going to replan.")
                return "replan"
            else:
                log.info("[ROUTE_REVIEW] Review flagged revision but max replans reached, going to finalize.")
                return "finalize"

        log.info("[ROUTE_REVIEW] Review passed, going to finalize.")
        return "finalize"

    # ── helpers ───────────────────────────────────────────────────────────

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
            "step_results": [],
            "current_step": 0,
            "plan_fix_iter": 0,
            "last_plan_output": "",
            "review_feedback": "",
            "replan_count": 0,
        }

        final_state = self.graph.invoke(state, config={"recursion_limit": 200})
        msgs = final_state.get("messages", [])

        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                return msg.content

        return msgs[-1].content if msgs else ""
