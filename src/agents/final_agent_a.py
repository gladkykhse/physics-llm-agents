import logging as log
import re
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode

from src.agents.utils.llm import make_llm
from src.agents.utils.tools import sympy_eval, sympy_solve, vector_math, wikipedia_multi_search
from src.agents.utils.utils import scieval_split_problem_and_options
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/final_agent_a.yaml")

log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class State(TypedDict):
    knowledge_messages: Annotated[List[AnyMessage], add_messages]
    messages: Annotated[List[AnyMessage], add_messages]
    problem: str
    problem_type: str
    filtered_knowledge: str
    analysis: str
    plan: List[str]
    current_step: int
    plan_fix_iter: int
    last_plan_output: str


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

        graph.add_node("analyze", self._analyze)
        graph.add_node("plan", self._plan)
        graph.add_node("fix_plan", self._fix_plan)
        graph.add_node("execute_step", self._execute_step)
        graph.add_node("execute_tools", self.math_tools)
        graph.add_node("summarize_step", self._summarize_step)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("classify_problem")
        graph.add_edge("classify_problem", "cot_retrieve_knowledge")
        graph.add_edge("cot_retrieve_knowledge", "retrieve_knowledge")
        graph.add_edge("retrieve_knowledge", "knowledge_tools")
        graph.add_edge("knowledge_tools", "filter_knowledge")

        graph.add_conditional_edges(
            "filter_knowledge",
            self._route_problem_type,
            {"math": "analyze", "theory": "solve_problem"},
        )

        graph.add_edge("analyze", "plan")
        graph.add_conditional_edges(
            "plan",
            self._route_plan,
            {"execute": "execute_step", "fix_plan": "fix_plan", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "fix_plan",
            self._route_plan,
            {"execute": "execute_step", "fix_plan": "fix_plan", "finalize": "finalize"},
        )

        graph.add_conditional_edges(
            "execute_step",
            self._route_after_execute_step,
            {"tools": "execute_tools", "execute_step": "execute_step", "finalize": "finalize"},
        )
        graph.add_edge("execute_tools", "summarize_step")
        graph.add_conditional_edges(
            "summarize_step",
            self._route_after_summarize_step,
            {"execute_step": "execute_step", "finalize": "finalize"},
        )

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
        else:
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

        prompt = HumanMessage(
            content=agent_cfg["filter_knowledge_prompt"].format(problem=state["problem"], knowledge=knowledge_str)
        )
        ai = self.base_llm.invoke([prompt])

        filtered_knowledge = "# Wikipedia Search Results\n\n" + (ai.content or "")
        log.info(f"[FILTER_KNOWLEDGE] - Output: {filtered_knowledge}")

        state["filtered_knowledge"] = filtered_knowledge
        state["messages"] = [HumanMessage(content=filtered_knowledge)]
        return state

    def _solve_problem(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["solve_prompt"].format(problem=state["problem"]))
        msgs = state["messages"] + [prompt]
        ai = self.base_llm.invoke(msgs)

        log.info(f"[SOLVE_PROBLEM] - Output: {ai.content}")
        state["messages"] = [ai]
        return state

    def _analyze(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["analyze_prompt"])
        msgs = state["messages"] + [prompt]
        ai = self.base_llm.invoke(msgs)
        log.info(f"[ANALYZE] - {ai.content}")

        state["analysis"] = ai.content
        state["messages"] = []
        return state

    def _plan(self, state: State) -> State:
        planning_msgs = [
            SystemMessage(content=agent_cfg["math_system_prompt"]),
            HumanMessage(content=f"# Problem\n{state['problem']}"),
            HumanMessage(content=state["filtered_knowledge"]),
            HumanMessage(content=agent_cfg["plan_prompt"].format(analysis=state["analysis"])),
        ]
        ai = self.base_llm.invoke(planning_msgs)
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
        planning_msgs = [
            SystemMessage(content=agent_cfg["math_system_prompt"]),
            HumanMessage(content=f"# Problem\n{state['problem']}"),
            HumanMessage(content=state["filtered_knowledge"]),
            HumanMessage(
                content=agent_cfg["fix_plan_prompt"].format(
                    analysis=state["analysis"],
                    failed_output=state["last_plan_output"],
                )
            ),
        ]
        ai = self.base_llm.invoke(planning_msgs)
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

    def _execute_step(self, state: State) -> State:
        step_idx = state["current_step"]
        step_desc = state["plan"][step_idx]

        prompt = HumanMessage(
            content=agent_cfg["execute_step_prompt"].format(
                step_number=step_idx + 1,
                total_steps=len(state["plan"]),
                step_description=step_desc,
            )
        )
        msgs = state["messages"] + [prompt]
        ai = self.math_tools_llm.invoke(msgs)

        tool_calls = getattr(ai, "tool_calls", None)
        if tool_calls and len(tool_calls) > 1:
            log.warning(f"[EXECUTE_STEP] Model returned {len(tool_calls)} tool calls; truncating to 1.")
            ai.tool_calls = tool_calls[:1]
            tool_calls = ai.tool_calls

        log.info(f"[EXECUTE_STEP {step_idx + 1}/{len(state['plan'])}] - Content: {ai.content}")
        if tool_calls:
            log.info(f"[EXECUTE_STEP {step_idx + 1}/{len(state['plan'])}] - Tool: {tool_calls[0]}")

        state["messages"] = [ai]

        if not tool_calls:
            state["current_step"] += 1

        return state

    def _summarize_step(self, state: State) -> State:
        step_idx = state["current_step"]
        step_desc = state["plan"][step_idx]

        tool_name = "tool"
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                tool_name = msg.tool_calls[0]["name"]
                break

        prompt = HumanMessage(
            content=agent_cfg["summarize_step_prompt"].format(
                step_number=step_idx + 1,
                total_steps=len(state["plan"]),
                step_description=step_desc,
                tool_name=tool_name,
            )
        )
        msgs = state["messages"] + [prompt]
        ai = self.base_llm.invoke(msgs)

        log.info(f"[SUMMARIZE_STEP {step_idx + 1}/{len(state['plan'])}] - {ai.content}")

        state["messages"] = [ai]
        state["current_step"] += 1
        return state

    def _finalize(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["finalizer_prompt"].format(problem=state["problem"]))
        msgs = state["messages"] + [prompt]
        ai = self.base_llm.invoke(msgs)
        log.info(f"[FINALIZE] LLM Response: {ai.content}")
        state["messages"] = [ai]
        return state

    def _route_problem_type(self, state: State) -> str:
        return state["problem_type"]

    def _route_plan(self, state: State) -> str:
        if state["plan"]:
            log.info(f"[ROUTE_PLAN] Plan ready with {len(state['plan'])} steps.")
            return "execute"

        if state["plan_fix_iter"] >= agent_cfg["max_plan_fix_iters"]:
            log.warning("[ROUTE_PLAN] Max fixes exhausted with empty plan, skipping to finalize.")
            return "finalize"

        log.info(f"[ROUTE_PLAN] Plan parsing failed, routing to fix attempt {state['plan_fix_iter'] + 1}.")
        return "fix_plan"

    def _route_after_execute_step(self, state: State) -> str:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) if isinstance(last, AIMessage) else None

        if tool_calls:
            log.info("[ROUTE_AFTER_EXECUTE] Tool call emitted, going to execute_tools.")
            return "tools"

        return self._route_advance(state)

    def _route_after_summarize_step(self, state: State) -> str:
        return self._route_advance(state)

    def _route_advance(self, state: State) -> str:
        if state["current_step"] >= len(state["plan"]):
            log.info("[ROUTE_ADVANCE] All plan steps complete, going to finalize.")
            return "finalize"

        if state["current_step"] >= agent_cfg["max_execute_steps"]:
            log.info("[ROUTE_ADVANCE] Max execute steps reached, going to finalize.")
            return "finalize"

        log.info(f"[ROUTE_ADVANCE] Continuing to step {state['current_step'] + 1}.")
        return "execute_step"

    @staticmethod
    def _commit_plan(state: State, steps: list) -> None:
        plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        state["messages"] = [AIMessage(content=plan_text)]

    @staticmethod
    def _parse_plan(text: str) -> List[str]:
        lines = re.findall(r"^\s*\d+\.\s*(.+)$", text, re.MULTILINE)
        return [line.strip() for line in lines if line.strip()]

    def solve(self, problem: str) -> str:
        question, options = scieval_split_problem_and_options(full_text=problem)
        log.info(f"[QUESTION] - {question}")
        log.info(f"[OPTIONS] - {options}")

        state: State = {
            "knowledge_messages": [],
            "messages": [],
            "problem": problem,
            "problem_type": "",
            "filtered_knowledge": "",
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
