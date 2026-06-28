import logging as log
import re
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode

from src.agents.utils.llm import make_llm
from src.agents.utils.tools import wikipedia_multi_search
from src.agents.utils.utils import scieval_split_problem_and_options
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/planning_retrieval_agent_gated.yaml")

log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MAX_BRIEF_CHARS = 600

BRIEF_HEADER = (
    "# Reference notes (UNVERIFIED external text — may be wrong or off-topic; "
    "trust standard physics over this if they conflict)\n"
)


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    knowledge_messages: Annotated[List[AnyMessage], add_messages]
    problem: str
    analysis: str
    needs_retrieval: bool
    knowledge_gap: str
    knowledge_brief: str
    plan: List[str]
    current_step: int
    plan_fix_iter: int
    last_plan_output: str


class PhysicsPlanningSelectiveRetrievalAgent:
    def __init__(self) -> None:
        knowledge_tools_list = [wikipedia_multi_search]

        self.llm = make_llm(temperature=0.0)
        self.knowledge_tools_llm = make_llm(temperature=0.0).bind_tools(knowledge_tools_list)

        graph = StateGraph(State)
        self.knowledge_tools = ToolNode(knowledge_tools_list, messages_key="knowledge_messages")

        graph.add_node("analyze", self._analyze)
        graph.add_node("retrieve_knowledge", self._retrieve_knowledge)
        graph.add_node("knowledge_tools", self.knowledge_tools)
        graph.add_node("filter_knowledge", self._filter_knowledge)
        graph.add_node("plan", self._plan)
        graph.add_node("fix_plan", self._fix_plan)
        graph.add_node("execute", self._execute)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("analyze")
        graph.add_conditional_edges(
            "analyze",
            self._route_gate,
            {"retrieve_knowledge": "retrieve_knowledge", "plan": "plan"},
        )
        graph.add_edge("retrieve_knowledge", "knowledge_tools")
        graph.add_edge("knowledge_tools", "filter_knowledge")
        graph.add_edge("filter_knowledge", "plan")
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

    def _analyze(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["analyze_prompt"])
        msgs = state["messages"] + [prompt]
        ai = self.llm.invoke(msgs)
        log.info(f"[ANALYZE] - {ai.content}")

        gap = self._parse_gap(ai.content)
        needs = self._needs_retrieval(gap)
        analysis = re.split(r"\n+#*\s*Knowledge gap", ai.content, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        analysis = analysis or ai.content.strip()

        log.info(f"[ANALYZE] needs_retrieval={needs} gap={gap!r}")

        state["analysis"] = analysis
        state["knowledge_gap"] = gap if needs else ""
        state["needs_retrieval"] = needs
        state["messages"] = []
        return state

    def _retrieve_knowledge(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["retrieve_knowledge_prompt"].format(gap=state["knowledge_gap"]))
        msgs = [SystemMessage(content=agent_cfg["main_system_prompt"]), prompt]
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

        if not knowledge_str.strip():
            log.info("[FILTER_KNOWLEDGE] - No retrieved content; skipping injection.")
            state["knowledge_brief"] = ""
            return state

        prompt = HumanMessage(
            content=agent_cfg["filter_knowledge_prompt"].format(gap=state["knowledge_gap"], knowledge=knowledge_str)
        )
        ai = self.llm.invoke([prompt])
        brief = ai.content.strip()

        if not brief or brief.lower().rstrip(".") == "none":
            log.info("[FILTER_KNOWLEDGE] - Brief empty/None; skipping injection.")
            state["knowledge_brief"] = ""
            return state

        if len(brief) > MAX_BRIEF_CHARS:
            brief = brief[:MAX_BRIEF_CHARS].rstrip() + " ..."

        brief = BRIEF_HEADER + brief
        log.info(f"[FILTER_KNOWLEDGE] - Output: {brief}")

        state["knowledge_brief"] = brief
        state["messages"] = [HumanMessage(content=brief)]
        return state

    def _plan(self, state: State) -> State:
        prompt_content = agent_cfg["plan_prompt"].format(analysis=state["analysis"])
        planning_msgs = [
            SystemMessage(content=agent_cfg["main_system_prompt"]),
            HumanMessage(content=f"# Problem\n{state['problem']}"),
        ]
        if state.get("knowledge_brief"):
            planning_msgs.append(HumanMessage(content=state["knowledge_brief"]))
        planning_msgs.append(HumanMessage(content=prompt_content))

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
        ]
        if state.get("knowledge_brief"):
            planning_msgs.append(HumanMessage(content=state["knowledge_brief"]))
        planning_msgs.append(HumanMessage(content=prompt_content))

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

    def _route_gate(self, state: State) -> str:
        if state.get("needs_retrieval"):
            log.info("[ROUTE_GATE] Knowledge gap flagged → retrieving.")
            return "retrieve_knowledge"
        log.info("[ROUTE_GATE] No gap → planning directly (pure-planning path).")
        return "plan"

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

    @staticmethod
    def _commit_plan(state: State, steps: list) -> None:
        plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        state["messages"] = [AIMessage(content=plan_text)]

    @staticmethod
    def _parse_plan(text: str) -> List[str]:
        lines = re.findall(r"^\s*\d+\.\s*(.+)$", text, re.MULTILINE)
        return [line.strip() for line in lines if line.strip()]

    @staticmethod
    def _parse_gap(text: str) -> str:
        matches = re.findall(r"Knowledge gap\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE)
        return matches[-1].strip() if matches else ""

    @staticmethod
    def _needs_retrieval(gap: str) -> bool:
        g = gap.strip().strip(".").strip().lower()
        if len(g) <= 3:
            return False
        if g in {"none", "no", "n/a", "na", "none needed", "no gap", "nothing"}:
            return False
        if re.match(r"^(none|no gap|nothing|not needed)\b", g):
            return False
        return True

    def solve(self, problem: str) -> str:
        question, options = scieval_split_problem_and_options(full_text=problem)
        log.info(f"[QUESTION] - {question}")
        log.info(f"[OPTIONS] - {options}")

        state: State = {
            "messages": [
                SystemMessage(content=agent_cfg["main_system_prompt"]),
                HumanMessage(content=f"# Problem\n{problem}"),
            ],
            "knowledge_messages": [],
            "problem": problem,
            "analysis": "",
            "needs_retrieval": False,
            "knowledge_gap": "",
            "knowledge_brief": "",
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
