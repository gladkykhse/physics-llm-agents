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

agent_cfg = load_yaml("config/final_agent_b.yaml")

log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class State(TypedDict):
    knowledge_messages: Annotated[List[AnyMessage], add_messages]
    problem: str
    problem_type: str
    filtered_knowledge: str
    analysis: str
    plan: List[str]
    current_step: int
    step_summaries: List[str]
    plan_fix_iter: int
    last_plan_output: str
    step_context: List[AnyMessage]
    step_react_iter: int


class PhysicsReactAgent:
    def __init__(self) -> None:
        knowledge_tools_list = [wikipedia_multi_search]
        math_tools_list = [sympy_eval, sympy_solve, vector_math]

        self.base_llm = make_llm(temperature=0.0)
        self.knowledge_tools_llm = make_llm(temperature=0.0).bind_tools(knowledge_tools_list)
        self.math_tools_llm = make_llm(temperature=0.0).bind_tools(math_tools_list)

        self.tool_map = {t.name: t for t in math_tools_list}

        graph = StateGraph(State)

        self.knowledge_tools = ToolNode(knowledge_tools_list, messages_key="knowledge_messages")
        graph.add_node("classify", self._classify)
        graph.add_node("cot_retrieve_knowledge", self._cot_retrieve_knowledge)
        graph.add_node("retrieve_knowledge", self._retrieve_knowledge)
        graph.add_node("knowledge_tools", self.knowledge_tools)
        graph.add_node("filter_knowledge", self._filter_knowledge)

        graph.add_node("analyze", self._analyze)
        graph.add_node("plan", self._plan)
        graph.add_node("fix_plan", self._fix_plan)

        graph.add_node("prepare_step", self._prepare_step)
        graph.add_node("execute_theory_step", self._execute_theory_step)
        graph.add_node("step_thought", self._step_thought)
        graph.add_node("step_act", self._step_act)
        graph.add_node("summarize_step", self._summarize_step)

        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("classify")
        graph.add_edge("classify", "cot_retrieve_knowledge")
        graph.add_edge("cot_retrieve_knowledge", "retrieve_knowledge")
        graph.add_edge("retrieve_knowledge", "knowledge_tools")
        graph.add_edge("knowledge_tools", "filter_knowledge")

        graph.add_edge("filter_knowledge", "analyze")
        graph.add_edge("analyze", "plan")
        graph.add_conditional_edges(
            "plan",
            self._route_plan,
            {"execute": "prepare_step", "fix_plan": "fix_plan", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "fix_plan",
            self._route_plan,
            {"execute": "prepare_step", "fix_plan": "fix_plan", "finalize": "finalize"},
        )

        graph.add_conditional_edges(
            "prepare_step",
            self._route_step_type,
            {"theory": "execute_theory_step", "math": "step_thought"},
        )

        graph.add_edge("execute_theory_step", "summarize_step")

        graph.add_conditional_edges(
            "step_thought",
            self._route_step_thought,
            {"act": "step_act", "summarize": "summarize_step"},
        )
        graph.add_conditional_edges(
            "step_act",
            self._route_step_act,
            {"thought": "step_thought", "summarize": "summarize_step"},
        )

        graph.add_conditional_edges(
            "summarize_step",
            self._route_next_step,
            {"continue": "prepare_step", "finalize": "finalize"},
        )

        graph.add_edge("finalize", END)

        self.graph = graph.compile()

    def _classify(self, state: State) -> State:
        prompt = agent_cfg["problem_type_router_prompt"].format(problem=state["problem"])
        ai = self.base_llm.invoke([HumanMessage(content=prompt)])
        text = (ai.content or "").upper()

        if "ANSWER: THEORETICAL" in text:
            problem_type = "theory"
        elif "ANSWER: MATHEMATICAL" in text:
            problem_type = "math"
        else:
            problem_type = "math"

        log.info(f"[CLASSIFY] {problem_type} | raw: {ai.content}")
        state["problem_type"] = problem_type
        return state

    def _cot_retrieve_knowledge(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["cot_retrieve_knowledge_prompt"].format(problem=state["problem"]))
        msgs = state["knowledge_messages"] + [prompt]
        ai = self.base_llm.invoke(msgs)
        log.info(f"[COT_RETRIEVE] {ai.content}")
        state["knowledge_messages"] = [ai]
        return state

    def _retrieve_knowledge(self, state: State) -> State:
        prompt = HumanMessage(content=agent_cfg["retrieve_knowledge_prompt"])
        msgs = state["knowledge_messages"] + [prompt]
        ai = self.knowledge_tools_llm.invoke(msgs)
        log.info(f"[RETRIEVE] {ai.content}")
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

        filtered = "# Wikipedia Search Results\n\n" + ai.content
        log.info(f"[FILTER] {filtered[:300]}")
        state["filtered_knowledge"] = filtered
        return state

    def _analyze(self, state: State) -> State:
        msgs = [HumanMessage(content=f"# Problem\n{state['problem']}")]
        if state["filtered_knowledge"]:
            msgs.append(HumanMessage(content=state["filtered_knowledge"]))
        msgs.append(HumanMessage(content=agent_cfg["analyze_prompt"]))

        ai = self.base_llm.invoke(msgs)
        log.info(f"[ANALYZE] {ai.content}")
        state["analysis"] = ai.content
        return state

    def _plan(self, state: State) -> State:
        plan_key = "plan_math_prompt" if state["problem_type"] == "math" else "plan_theory_prompt"
        prompt_content = agent_cfg[plan_key].format(analysis=state["analysis"])

        sys_key = "math_system_prompt" if state["problem_type"] == "math" else "theory_system_prompt"
        planning_msgs = [
            SystemMessage(content=agent_cfg[sys_key]),
            HumanMessage(content=f"# Problem\n{state['problem']}"),
            HumanMessage(content=prompt_content),
        ]

        ai = self.base_llm.invoke(planning_msgs)
        log.info(f"[PLAN] {ai.content}")

        steps = self._parse_plan(ai.content)
        log.info(f"[PLAN] Parsed {len(steps)} steps: {steps}")

        state["plan"] = steps
        state["last_plan_output"] = ai.content
        state["plan_fix_iter"] = 0
        state["current_step"] = 0
        return state

    def _fix_plan(self, state: State) -> State:
        prompt_content = agent_cfg["fix_plan_prompt"].format(
            analysis=state["analysis"],
            failed_output=state["last_plan_output"],
        )
        sys_key = "math_system_prompt" if state["problem_type"] == "math" else "theory_system_prompt"
        planning_msgs = [
            SystemMessage(content=agent_cfg[sys_key]),
            HumanMessage(content=f"# Problem\n{state['problem']}"),
            HumanMessage(content=prompt_content),
        ]

        ai = self.base_llm.invoke(planning_msgs)
        state["plan_fix_iter"] += 1
        log.info(f"[FIX_PLAN] Attempt {state['plan_fix_iter']} - {ai.content}")

        steps = self._parse_plan(ai.content)
        state["last_plan_output"] = ai.content

        if steps:
            state["plan"] = steps
            log.info(f"[FIX_PLAN] Parsed {len(steps)} steps: {steps}")
        elif state["plan_fix_iter"] >= agent_cfg["max_plan_fix_iters"]:
            state["plan"] = [ai.content.strip()]
            log.warning("[FIX_PLAN] Max fix attempts; falling back to single step.")

        return state

    def _prepare_step(self, state: State) -> State:
        step_idx = state["current_step"]
        step_desc = state["plan"][step_idx]

        sys_key = "math_system_prompt" if state["problem_type"] == "math" else "theory_system_prompt"
        ctx: List[AnyMessage] = [
            SystemMessage(content=agent_cfg[sys_key]),
            HumanMessage(content=f"# Problem\n{state['problem']}"),
        ]

        if state["filtered_knowledge"]:
            ctx.append(HumanMessage(content=state["filtered_knowledge"]))

        plan_text = "\n".join(f"{'>> ' if i == step_idx else '   '}{i + 1}. {s}" for i, s in enumerate(state["plan"]))
        ctx.append(HumanMessage(content=f"# Plan\n{plan_text}"))

        if state["step_summaries"]:
            summaries = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(state["step_summaries"]))
            ctx.append(HumanMessage(content=f"# Completed Steps\n{summaries}"))

        state["step_context"] = ctx
        state["step_react_iter"] = 0
        log.info(f"[PREPARE_STEP] Step {step_idx + 1}/{len(state['plan'])}: {step_desc}")
        return state

    def _execute_theory_step(self, state: State) -> State:
        ctx = list(state["step_context"])
        step_idx = state["current_step"]

        prompt = HumanMessage(
            content=agent_cfg["execute_theory_step_prompt"].format(
                step_number=step_idx + 1,
                total_steps=len(state["plan"]),
                step_description=state["plan"][step_idx],
            )
        )
        ctx.append(prompt)
        ai = self.base_llm.invoke(ctx)
        log.info(f"[THEORY_STEP {step_idx + 1}] {ai.content}")

        ctx.append(ai)
        state["step_context"] = ctx
        return state

    def _step_thought(self, state: State) -> State:
        ctx = list(state["step_context"])
        step_idx = state["current_step"]

        prompt = HumanMessage(
            content=agent_cfg["step_thought_prompt"].format(
                step_number=step_idx + 1,
                total_steps=len(state["plan"]),
                step_description=state["plan"][step_idx],
            )
        )
        ctx.append(prompt)
        ai = self.base_llm.invoke(ctx)
        log.info(f"[THOUGHT step {step_idx + 1}, iter {state['step_react_iter']}] {ai.content}")

        ctx.append(ai)
        state["step_context"] = ctx
        return state

    def _step_act(self, state: State) -> State:
        ctx = list(state["step_context"])
        step_idx = state["current_step"]

        ctx.append(HumanMessage(content=agent_cfg["step_act_prompt"]))

        try:
            ai = self.math_tools_llm.invoke(ctx)
        except Exception as e:
            # Llama 8B can emit malformed tool calls that fail LangChain parsing
            log.warning(f"[ACT step {step_idx + 1}] LLM returned malformed output: {e}")
            ai = AIMessage(content="Tool call failed due to malformed output.")
        ctx.append(ai)

        tool_calls = getattr(ai, "tool_calls", None)

        if tool_calls:
            if len(tool_calls) > 1:
                log.warning(f"[ACT step {step_idx + 1}] {len(tool_calls)} calls; using first only.")
                ai.tool_calls = tool_calls[:1]

            tc = tool_calls[0]
            log.info(f"[ACT step {step_idx + 1}] Calling {tc['name']}({tc['args']})")

            try:
                result = self.tool_map[tc["name"]].invoke(tc["args"])
            except Exception as e:
                log.error(f"[ACT step {step_idx + 1}] Tool error: {e}")
                result = f"Error: {e}"

            ctx.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        else:
            log.info(f"[ACT step {step_idx + 1}] No tool call, model responded with text.")

        state["step_context"] = ctx
        state["step_react_iter"] += 1
        return state

    def _summarize_step(self, state: State) -> State:
        step_idx = state["current_step"]

        tool_results = []
        last_thought = ""
        for msg in state["step_context"]:
            if isinstance(msg, ToolMessage) and msg.content:
                tool_results.append(msg.content)
            elif isinstance(msg, AIMessage) and msg.content:
                last_thought = msg.content

        step_output = ""
        if tool_results:
            for i, tr in enumerate(tool_results, 1):
                step_output += f"Tool result {i}: {tr}\n"
        if last_thought:
            step_output += f"Final reasoning: {last_thought}"

        prompt = HumanMessage(
            content=agent_cfg["summarize_step_prompt"].format(
                step_number=step_idx + 1,
                step_description=state["plan"][step_idx],
                step_output=step_output,
            )
        )
        ai = self.base_llm.invoke([prompt])
        summary = ai.content.strip()
        log.info(f"[SUMMARY step {step_idx + 1}] {summary}")

        state["step_summaries"] = list(state["step_summaries"]) + [summary]
        state["current_step"] = step_idx + 1
        state["step_react_iter"] = 0
        state["step_context"] = []
        return state

    def _finalize(self, state: State) -> State:
        msgs: List[AnyMessage] = [
            HumanMessage(content=f"# Problem\n{state['problem']}"),
        ]

        if state["filtered_knowledge"]:
            msgs.append(HumanMessage(content=state["filtered_knowledge"]))

        if state["step_summaries"]:
            summaries = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(state["step_summaries"]))
            msgs.append(HumanMessage(content=f"# Step Results\n{summaries}"))

        msgs.append(HumanMessage(content=agent_cfg["finalizer_prompt"].format(problem=state["problem"])))
        ai = self.base_llm.invoke(msgs)
        log.info(f"[FINALIZE] {ai.content}")

        state["step_context"] = [ai]
        return state

    def _route_plan(self, state: State) -> str:
        if state["plan"]:
            log.info(f"[ROUTE_PLAN] Plan ready ({len(state['plan'])} steps).")
            return "execute"
        if state["plan_fix_iter"] >= agent_cfg["max_plan_fix_iters"]:
            log.warning("[ROUTE_PLAN] Max fixes exhausted; skipping to finalize.")
            return "finalize"
        log.info(f"[ROUTE_PLAN] Plan parse failed; fix attempt {state['plan_fix_iter'] + 1}.")
        return "fix_plan"

    def _route_step_type(self, state: State) -> str:
        return state["problem_type"]

    def _route_step_thought(self, state: State) -> str:
        last = state["step_context"][-1] if state["step_context"] else None
        content = (last.content or "").upper() if last else ""

        if "STEP COMPLETE" in content:
            # in math mode, forbid STEP COMPLETE before any tool has run
            if state["problem_type"] == "math" and state["step_react_iter"] == 0:
                log.info(
                    f"[ROUTE_THOUGHT] Step {state['current_step'] + 1} said STEP COMPLETE "
                    "but no tool was called yet → forcing act."
                )
                return "act"
            log.info(f"[ROUTE_THOUGHT] Step {state['current_step'] + 1} complete.")
            return "summarize"
        log.info(f"[ROUTE_THOUGHT] Step {state['current_step'] + 1} needs computation → act.")
        return "act"

    def _route_step_act(self, state: State) -> str:
        if state["step_react_iter"] >= agent_cfg["max_step_react_iters"]:
            log.info(f"[ROUTE_ACT] Max mini-react iters for step {state['current_step'] + 1}.")
            return "summarize"

        last = state["step_context"][-1] if state["step_context"] else None
        if isinstance(last, ToolMessage):
            log.info("[ROUTE_ACT] Tool result received → thought.")
            return "thought"

        log.info("[ROUTE_ACT] No tool call → summarize.")
        return "summarize"

    def _route_next_step(self, state: State) -> str:
        if state["current_step"] >= len(state["plan"]):
            log.info("[ROUTE_NEXT] All steps complete → finalize.")
            return "finalize"
        if state["current_step"] >= agent_cfg["max_plan_steps"]:
            log.info("[ROUTE_NEXT] Max plan steps reached → finalize.")
            return "finalize"
        log.info(f"[ROUTE_NEXT] Continuing to step {state['current_step'] + 1}.")
        return "continue"

    @staticmethod
    def _parse_plan(text: str) -> List[str]:
        lines = re.findall(r"^\s*\d+\.\s*(.+)$", text, re.MULTILINE)
        return [line.strip() for line in lines if line.strip()]

    def solve(self, problem: str) -> str:
        question, options = scieval_split_problem_and_options(full_text=problem)
        log.info(f"[QUESTION] {question}")
        log.info(f"[OPTIONS] {options}")

        state: State = {
            "knowledge_messages": [],
            "problem": problem,
            "problem_type": "",
            "filtered_knowledge": "",
            "analysis": "",
            "plan": [],
            "current_step": 0,
            "step_summaries": [],
            "plan_fix_iter": 0,
            "last_plan_output": "",
            "step_context": [],
            "step_react_iter": 0,
        }

        final_state = self.graph.invoke(state, config={"recursion_limit": 200})

        for msg in reversed(final_state.get("step_context", [])):
            if isinstance(msg, AIMessage):
                return msg.content

        if final_state.get("step_summaries"):
            return final_state["step_summaries"][-1]

        return ""
