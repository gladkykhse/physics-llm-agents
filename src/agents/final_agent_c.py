import logging as log
import re
from typing import Annotated, List, Optional, TypedDict

import sympy as sp
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages

from src.agents.utils.llm import make_llm
from src.agents.utils.tools import sympy_eval, sympy_solve, wikipedia_multi_search
from src.agents.utils.utils import scieval_split_problem_and_options
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/final_agent_c.yaml")

log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MAX_BRIEF_CHARS = int(agent_cfg["max_brief_chars"])
MAX_CALCS_PER_STEP = int(agent_cfg["max_calcs_per_step"])

# standalone number: optional sign, int/decimal, optional sci exponent
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
# markers indicating a vector/list rather than a single scalar
_VECTOR_MARKERS = ("<", ">", "[", "]", "{", "}")


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    problem: str
    allow_compute: bool
    options: str
    analysis: str
    queries: List[str]
    retrieval_raw: str
    knowledge_brief: str
    plan: List[str]
    current_step: int
    plan_fix_iter: int
    last_plan_output: str
    step_results: List[str]
    step_calcs: List[str]
    last_value: str
    computed_ok: bool
    final_solution: str


class PhysicsReactAgent:
    def __init__(self) -> None:
        self.llm = make_llm(temperature=0.0)

        graph = StateGraph(State)

        graph.add_node("analyze", self._analyze)
        graph.add_node("retrieve_knowledge", self._retrieve_knowledge)
        graph.add_node("filter_knowledge", self._filter_knowledge)
        graph.add_node("plan", self._plan)
        graph.add_node("fix_plan", self._fix_plan)
        graph.add_node("execute", self._execute)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("analyze")
        graph.add_edge("analyze", "retrieve_knowledge")
        graph.add_edge("retrieve_knowledge", "filter_knowledge")
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

    def _base_ctx(self, state: State) -> List[AnyMessage]:
        ctx: List[AnyMessage] = [
            SystemMessage(content=agent_cfg["main_system_prompt"]),
            HumanMessage(content=f"# Problem\n{state['problem']}"),
        ]
        if state.get("knowledge_brief"):
            ctx.append(HumanMessage(content=state["knowledge_brief"]))
        return ctx

    def _work_ctx(self, state: State) -> List[AnyMessage]:
        ctx = self._base_ctx(state)
        if state.get("plan"):
            plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(state["plan"]))
            ctx.append(AIMessage(content="# Plan\n" + plan_text))
        results = state.get("step_results", [])
        calcs = state.get("step_calcs", [])
        for i, reasoning in enumerate(results):
            ctx.append(AIMessage(content=reasoning))
            calc = calcs[i] if i < len(calcs) else ""
            if calc:
                ctx.append(HumanMessage(content="Calculator results (exact — use these):\n" + calc))
        return ctx

    def _analyze(self, state: State) -> State:
        question, options = scieval_split_problem_and_options(full_text=state["problem"])
        state["options"] = state.get("options") or options
        state["allow_compute"] = self._needs_compute(question)
        state.setdefault("step_results", [])
        state.setdefault("step_calcs", [])
        state.setdefault("last_value", "")
        state.setdefault("computed_ok", False)
        state.setdefault("knowledge_brief", "")
        log.info(f"[ROUTE_TYPE] allow_compute={state['allow_compute']}")

        ctx = [
            SystemMessage(content=agent_cfg["main_system_prompt"]),
            HumanMessage(content=f"# Problem\n{state['problem']}"),
            HumanMessage(content=agent_cfg["analyze_prompt"]),
        ]
        ai = self.llm.invoke(ctx)
        log.info(f"[ANALYZE] - {ai.content}")

        analysis = re.split(r"#+\s*Search Queries", ai.content, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        analysis = re.sub(r"^#+\s*Analysis\s*", "", analysis, flags=re.IGNORECASE).strip() or ai.content.strip()
        queries = self._parse_queries(ai.content)
        log.info(f"[ANALYZE] queries={queries}")

        state["analysis"] = analysis
        state["queries"] = queries
        return state

    def _retrieve_knowledge(self, state: State) -> State:
        queries = state.get("queries") or []
        if not queries:
            log.info("[RETRIEVE_KNOWLEDGE] - No queries parsed; skipping retrieval.")
            state["retrieval_raw"] = ""
            return state

        try:
            result = wikipedia_multi_search.invoke({"queries": queries})
        except Exception as e:
            log.error(f"[RETRIEVE_KNOWLEDGE] - Error: {e}")
            result = ""

        log.info(f"[RETRIEVE_KNOWLEDGE] - Retrieved {len(result)} chars for {len(queries)} queries.")
        state["retrieval_raw"] = result
        return state

    def _filter_knowledge(self, state: State) -> State:
        knowledge_str = state.get("retrieval_raw", "") or ""

        if not knowledge_str.strip() or knowledge_str.startswith("No relevant"):
            log.info("[FILTER_KNOWLEDGE] - No retrieved content; skipping injection.")
            state["knowledge_brief"] = ""
            return state

        prompt = HumanMessage(
            content=agent_cfg["filter_knowledge_prompt"].format(problem=state["problem"], knowledge=knowledge_str)
        )
        ai = self.llm.invoke([prompt])
        brief = ai.content.strip()

        if not brief or brief.lower().rstrip(".") == "none":
            log.info("[FILTER_KNOWLEDGE] - Brief empty/None; skipping injection.")
            state["knowledge_brief"] = ""
            return state

        if len(brief) > MAX_BRIEF_CHARS:
            brief = brief[:MAX_BRIEF_CHARS].rstrip() + " ..."

        brief = "# Reference (general theory; use only if relevant)\n" + brief
        log.info(f"[FILTER_KNOWLEDGE] - Output: {brief}")

        state["knowledge_brief"] = brief
        return state

    def _plan(self, state: State) -> State:
        prompt_content = agent_cfg["plan_prompt"].format(analysis=state["analysis"])
        ctx = self._base_ctx(state) + [HumanMessage(content=prompt_content)]

        ai = self.llm.invoke(ctx)
        log.info(f"[PLAN] - {ai.content}")

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
        ctx = self._base_ctx(state) + [HumanMessage(content=prompt_content)]

        ai = self.llm.invoke(ctx)
        state["plan_fix_iter"] = state.get("plan_fix_iter", 0) + 1
        log.info(f"[FIX_PLAN] Attempt {state['plan_fix_iter']} - {ai.content}")

        steps = self._parse_plan(ai.content)
        state["last_plan_output"] = ai.content

        if steps:
            state["plan"] = steps
            log.info(f"[FIX_PLAN] Parsed {len(steps)} steps: {steps}")
        elif state["plan_fix_iter"] >= agent_cfg["max_plan_fix_iters"]:
            state["plan"] = [ai.content.strip()]
            log.warning("[FIX_PLAN] Max fix attempts reached, falling back to single step.")

        return state

    def _execute(self, state: State) -> State:
        step_idx = state["current_step"]
        step_desc = state["plan"][step_idx]

        prompt_key = "execute_compute_prompt" if state.get("allow_compute", True) else "execute_reason_prompt"
        prompt_content = agent_cfg[prompt_key].format(
            step_number=step_idx + 1,
            total_steps=len(state["plan"]),
            step_description=step_desc,
        )
        ctx = self._work_ctx(state) + [HumanMessage(content=prompt_content)]
        ai = self.llm.invoke(ctx)
        result_text = ai.content
        log.info(f"[EXECUTE step {step_idx + 1}] - {result_text}")

        reasoning = self._strip_scaffolding(result_text)

        calc_block = ""
        if state.get("allow_compute", True):
            calc_block, last_value, computed_ok = self._run_computations(result_text)
            if calc_block:
                log.info(f"[EXECUTE step {step_idx + 1}] calculator -> {calc_block}")
            if computed_ok and last_value is not None:
                state["last_value"] = last_value
                state["computed_ok"] = True

        state["step_results"] = state.get("step_results", []) + [reasoning]
        state["step_calcs"] = state.get("step_calcs", []) + [calc_block]
        state["current_step"] = step_idx + 1
        return state

    def _finalize(self, state: State) -> State:
        prompt_content = agent_cfg["finalizer_prompt"].format(problem=state["problem"])
        ctx = self._work_ctx(state) + [HumanMessage(content=prompt_content)]
        ai = self.llm.invoke(ctx)
        content = ai.content
        log.info(f"[FINALIZE] - {content}")

        content = self._resolve_answer(content, state)
        log.info(f"[FINALIZE] resolved -> {content.splitlines()[-1] if content else ''}")

        state["final_solution"] = content
        state["messages"] = [AIMessage(content=content)]
        return state

    def _route_plan(self, state: State) -> str:
        if state.get("plan"):
            log.info(f"[ROUTE_PLAN] Plan ready with {len(state['plan'])} steps.")
            return "execute"

        if state.get("plan_fix_iter", 0) >= agent_cfg["max_plan_fix_iters"]:
            log.warning("[ROUTE_PLAN] Max fixes exhausted with empty plan, skipping to finalize.")
            return "finalize"

        log.info(f"[ROUTE_PLAN] Plan parsing failed, routing to fix attempt {state.get('plan_fix_iter', 0) + 1}.")
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
    def _parse_plan(text: str) -> List[str]:
        lines = re.findall(r"^\s*\d+\.\s*(.+)$", text, re.MULTILINE)
        return [line.strip() for line in lines if line.strip()]

    @staticmethod
    def _parse_queries(text: str) -> List[str]:
        block = re.split(r"#+\s*Search Queries", text, maxsplit=1, flags=re.IGNORECASE)
        query_block = block[1] if len(block) > 1 else text
        queries = re.findall(r"^\s*\d+\.\s*(.+)$", query_block, re.MULTILINE)
        return [q.strip(" \"'") for q in queries if q.strip()][:4]

    @classmethod
    def _run_computations(cls, text: str) -> tuple:
        calcs = re.findall(r"^\s*[-*>`\s]*CALC\s*:\s*(.+?)\s*$", text, re.MULTILINE | re.IGNORECASE)
        solves = re.findall(r"^\s*[-*>`\s]*SOLVE\s*:\s*(.+?)\s*$", text, re.MULTILINE | re.IGNORECASE)

        lines: List[str] = []
        last_value: Optional[str] = None
        computed_ok = False
        count = 0
        seen: set = set()

        for expr in calcs:
            if count >= MAX_CALCS_PER_STEP:
                break
            expr_clean = cls._clean_expr(expr)
            if not expr_clean or expr_clean in seen:
                continue
            seen.add(expr_clean)
            try:
                out = sympy_eval.invoke({"expression": expr_clean})
            except Exception as e:
                out = f"Error: {e}"
            lines.append(f"{expr_clean} = {out}")
            if cls._safe_float(out) is not None:
                last_value = str(out).strip()
                computed_ok = True
            count += 1

        for spec in solves:
            if count >= MAX_CALCS_PER_STEP:
                break
            equation, _, symbol = spec.partition("|")
            equation = cls._clean_expr(equation)
            symbol = symbol.strip() or None
            # skip degenerate 'SOLVE: x | x' (no equation)
            if "=" not in equation:
                continue
            if equation in seen:
                continue
            seen.add(equation)
            try:
                out = sympy_solve.invoke({"equation": equation, "symbol": symbol})
            except Exception as e:
                out = f"Error: {e}"
            lines.append(f"solve {equation}" + (f" for {symbol}" if symbol else "") + f" -> {out}")
            root = cls._pick_root(out)
            if root is not None:
                last_value = root
                computed_ok = True
            count += 1

        return ("\n".join(lines), last_value, computed_ok)

    @staticmethod
    def _strip_scaffolding(text: str) -> str:
        out = []
        for ln in text.splitlines():
            s = ln.strip()
            if re.match(r"(?i)^[-*>`\s]*(CALC|SOLVE)\s*:", s):
                continue
            if re.match(r"(?i)^[#>\s]*(verified computation|calculator results)", s):
                continue
            if re.match(r"(?i)^#+\s*(result of step|step\b)", s):
                continue
            out.append(ln)
        return "\n".join(out).strip()

    @staticmethod
    def _clean_expr(expr: str) -> str:
        expr = expr.strip().strip("`").strip("$").strip()
        expr = expr.replace("×", "*").replace("·", "*").replace("÷", "/")
        expr = re.sub(r"(?<![A-Za-z])π", "pi", expr)
        expr = re.split(r"\s{2,}|\s*#", expr)[0].strip()
        return expr

    @staticmethod
    def _safe_float(s) -> Optional[float]:
        if not isinstance(s, str):
            return None
        s = s.strip()
        if not s or any(m in s for m in ("Error", "Invalid", "Exception", "symbol", "list")):
            return None
        try:
            val = sp.sympify(s)
            if val.free_symbols:
                return None
            return float(val.evalf())
        except Exception:
            return None

    @classmethod
    def _pick_root(cls, solve_output: str) -> Optional[str]:
        nums = re.findall(_NUM, solve_output)
        vals = [v for v in (cls._safe_float(n) for n in nums) if v is not None]
        if not vals:
            return None
        positives = [v for v in vals if v > 0]
        chosen = min(positives) if positives else vals[0]
        return repr(chosen)

    def _resolve_answer(self, content: str, state: State) -> str:
        model_letter = self._extract_letter(content)

        options = self._parse_options(state.get("options", ""))
        numeric_opts = {k: v for k, v in options.items() if v is not None}

        candidate = self._safe_float(state.get("last_value") or "") if state.get("computed_ok") else None

        chosen = model_letter
        if candidate is not None and len(numeric_opts) >= 3:
            best, best_err = None, None
            for letter, val in numeric_opts.items():
                scale = max(abs(val), abs(candidate), 1e-9)
                err = abs(val - candidate) / scale
                if best_err is None or err < best_err:
                    best, best_err = letter, err
            if best is not None and best_err <= 0.20:
                if best != model_letter:
                    log.info(
                        f"[FINALIZE] Closest-option override {model_letter} -> {best} "
                        f"(verified value={candidate}, rel_err={best_err:.3f})"
                    )
                chosen = best

        if chosen is None:
            chosen = sorted(options)[0] if options else "A"

        return self._set_answer_letter(content, chosen)

    @staticmethod
    def _extract_letter(text: str) -> Optional[str]:
        m = re.findall(r"Answer\s*[:\-]?\s*\**\s*([A-D])\b", text, re.IGNORECASE)
        return m[-1].upper() if m else None

    @classmethod
    def _parse_options(cls, options_text: str) -> dict:
        result: dict = {}
        for letter, body in re.findall(r"(?m)^\s*([A-Da-d])[\.\)]\s*(.+?)\s*$", options_text or ""):
            result[letter.upper()] = cls._leading_value(body)
        return result

    @staticmethod
    def _leading_value(s) -> Optional[float]:
        if not isinstance(s, str):
            return None
        s = s.strip()
        if any(m in s for m in _VECTOR_MARKERS):
            return None
        parts = re.split(r"[;,]|\band\b", s)
        numeric_parts = [p for p in parts if re.search(_NUM, p)]
        if len(numeric_parts) >= 2:
            return None
        m = re.search(_NUM, s)
        if not m:
            return None
        try:
            value = float(m.group(0))
        except ValueError:
            return None
        rest = s[m.end() :]
        frac = re.match(r"\s*/\s*(" + _NUM + r")", rest)
        if frac:
            denom = float(frac.group(1))
            if denom != 0:
                value /= denom
            rest = rest[frac.end() :]
        sci = re.match(r"\s*(?:[x×*]\s*)?10\s*(?:\^|\*\*)\s*([-+]?\d+)", rest)
        if sci:
            value *= 10 ** int(sci.group(1))
        return value

    @staticmethod
    def _set_answer_letter(content: str, letter: str) -> str:
        if re.search(r"Answer\s*[:\-]?\s*\**\s*[A-D]\b", content, re.IGNORECASE):
            return re.sub(
                r"(Answer\s*[:\-]?\s*\**\s*)[A-D]\b",
                lambda m: m.group(1) + letter,
                content,
                flags=re.IGNORECASE,
            )
        return content.rstrip() + f"\nAnswer: {letter}"

    @staticmethod
    def _needs_compute(question: str) -> bool:
        return bool(re.search(r"\d", question or ""))

    def solve(self, problem: str) -> str:
        question, options = scieval_split_problem_and_options(full_text=problem)
        log.info(f"[QUESTION] - {question}")
        log.info(f"[OPTIONS] - {options}")

        state: State = {
            "messages": [],
            "problem": problem,
            "allow_compute": self._needs_compute(question),
            "options": options,
            "analysis": "",
            "queries": [],
            "retrieval_raw": "",
            "knowledge_brief": "",
            "plan": [],
            "current_step": 0,
            "plan_fix_iter": 0,
            "last_plan_output": "",
            "step_results": [],
            "step_calcs": [],
            "last_value": "",
            "computed_ok": False,
            "final_solution": "",
        }

        final_state = self.graph.invoke(state, config={"recursion_limit": 200})

        if final_state.get("final_solution"):
            return final_state["final_solution"]
        msgs = final_state.get("messages", [])
        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                return msg.content
        return msgs[-1].content if msgs else ""
