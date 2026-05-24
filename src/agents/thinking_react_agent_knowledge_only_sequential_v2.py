import logging as log
import re
from typing import Annotated, List, TypedDict

from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode

from src.agents.utils.llm import make_llm
from src.agents.utils.tools import wikipedia_multi_search
from src.agents.utils.utils import scieval_split_problem_and_options
from src.utils.helpers import load_yaml

agent_cfg = load_yaml("config/thinking_react_agent_knowledge_only_sequential_v2.yaml")

log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class State(TypedDict):
    # Fixed inputs
    problem: str
    # options: str

    # Loop control
    retrieval_iter: int
    previous_queries: List[str]

    # Per-iteration working fields (overwritten each round)
    retrieval_plan: str
    knowledge_messages: List[AnyMessage]  # managed manually, no add_messages reducer
    raw_retrieved: str
    filtered_this_round: str
    sufficiency_verdict: str

    # Accumulated across iterations
    accumulated_knowledge: str

    # Final solver context
    messages: Annotated[List[AnyMessage], add_messages]


class PhysicsReactAgent:
    def __init__(self) -> None:
        knowledge_tools_list = [wikipedia_multi_search]

        self.base_llm = make_llm(temperature=0.0)
        self.knowledge_tools_llm = make_llm(temperature=0.0).bind_tools(knowledge_tools_list)

        graph = StateGraph(State)
        self.knowledge_tools = ToolNode(knowledge_tools_list, messages_key="knowledge_messages")

        # ── Nodes ────────────────────────────────────────────
        graph.add_node("plan_retrieval", self._plan_retrieval)
        graph.add_node("generate_tool_call", self._generate_tool_call)
        graph.add_node("knowledge_tools", self.knowledge_tools)
        graph.add_node("extract_raw_results", self._extract_raw_results)
        graph.add_node("filter_knowledge", self._filter_knowledge)
        graph.add_node("check_sufficiency", self._check_sufficiency)
        graph.add_node("consolidate_knowledge", self._consolidate_knowledge)
        graph.add_node("solve_problem", self._solve_problem)

        # ── Edges ────────────────────────────────────────────
        graph.set_entry_point("plan_retrieval")
        graph.add_edge("plan_retrieval", "generate_tool_call")
        graph.add_conditional_edges(
            "generate_tool_call",
            self._has_tool_calls,
            {
                "has_tools": "knowledge_tools",
                "no_tools": "extract_raw_results",  # skip tool execution gracefully
            },
        )
        graph.add_edge("knowledge_tools", "extract_raw_results")
        graph.add_edge("extract_raw_results", "filter_knowledge")
        graph.add_edge("filter_knowledge", "check_sufficiency")
        graph.add_conditional_edges(
            "check_sufficiency",
            self._should_continue_retrieval,
            {
                "continue": "plan_retrieval",
                "finish": "consolidate_knowledge",
            },
        )
        graph.add_edge("consolidate_knowledge", "solve_problem")
        graph.add_edge("solve_problem", END)

        self.graph = graph.compile()

    # ── Routing Logic ────────────────────────────────────────

    def _has_tool_calls(self, state: State) -> str:
        """Routes based on whether the LLM produced valid tool calls."""
        msgs = state.get("knowledge_messages", [])
        if msgs:
            last_msg = msgs[-1]
            if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
                log.info("[ROUTING] - Tool calls found. Proceeding to tool execution.")
                return "has_tools"

        log.warning("[ROUTING] - No tool calls generated. Skipping tool execution.")
        return "no_tools"

    def _should_continue_retrieval(self, state: State) -> str:
        max_iters = agent_cfg.get("max_retrieval_iters", 3)
        verdict = state.get("sufficiency_verdict", "")
        current_iter = state.get("retrieval_iter", 1)

        if current_iter >= max_iters:
            log.info(f"[ROUTING] - Max retrieval iterations ({max_iters}) reached. Proceeding to consolidation.")
            return "finish"

        if "SUFFICIENT" in verdict.upper() and "INSUFFICIENT" not in verdict.upper():
            log.info("[ROUTING] - Knowledge is sufficient. Proceeding to consolidation.")
            return "finish"

        log.info(
            f"[ROUTING] - Knowledge insufficient (iter {current_iter}/{max_iters}). Looping back for more retrieval."
        )
        return "continue"

    # ── Node: Plan Retrieval ─────────────────────────────────

    def _plan_retrieval(self, state: State) -> State:
        current_iter = state.get("retrieval_iter", 0)

        if current_iter == 0:
            prompt_text = agent_cfg["initial_plan_prompt"].format(
                problem=state["problem"],
            )
            log.info("[PLAN_RETRIEVAL] - First iteration: cold-start analysis.")
        else:
            prompt_text = agent_cfg["refine_plan_prompt"].format(
                problem=state["problem"],
                accumulated_knowledge=state.get("accumulated_knowledge", ""),
                previous_queries="\n".join(f"- {q}" for q in state.get("previous_queries", [])),
                sufficiency_verdict=state.get("sufficiency_verdict", ""),
            )
            log.info(f"[PLAN_RETRIEVAL] - Iteration {current_iter + 1}: gap-targeted refinement.")

        ai = self.base_llm.invoke([HumanMessage(content=prompt_text)])
        log.info(f"[PLAN_RETRIEVAL] - Output: {ai.content}")

        state["retrieval_plan"] = ai.content

        # Parse queries from the plan and track them
        new_queries = self._parse_queries(ai.content)
        prev = state.get("previous_queries", [])
        state["previous_queries"] = prev + new_queries
        log.info(f"[PLAN_RETRIEVAL] - Parsed queries: {new_queries}")

        return state

    # ── Node: Generate Tool Call ─────────────────────────────

    def _generate_tool_call(self, state: State) -> State:
        prompt_text = agent_cfg["generate_tool_call_prompt"].format(
            retrieval_plan=state["retrieval_plan"],
        )

        ai = self.knowledge_tools_llm.invoke([HumanMessage(content=prompt_text)])
        log.info(f"[GENERATE_TOOL_CALL] - Output: {ai.content}")

        tool_calls = getattr(ai, "tool_calls", None)
        if tool_calls:
            log.info(f"[GENERATE_TOOL_CALL] - Tool Calls: {tool_calls}")
        else:
            log.warning("[GENERATE_TOOL_CALL] - No tool calls generated. Solver will work with existing knowledge.")

        # Overwrite knowledge_messages — this is a disposable interface for ToolNode
        state["knowledge_messages"] = [ai]
        return state

    # ── Node: Extract Raw Results ────────────────────────────

    def _extract_raw_results(self, state: State) -> State:
        """
        Extracts raw text from ToolMessages produced by ToolNode and stores it
        in raw_retrieved. Without add_messages reducer, knowledge_messages is
        fully overwritten each iteration — it contains only the current round's
        ToolMessages after ToolNode runs.
        """
        raw_parts = []
        for msg in state.get("knowledge_messages", []):
            if isinstance(msg, ToolMessage):
                raw_parts.append(msg.content)

        raw_text = "\n\n".join(raw_parts)

        if not raw_text.strip():
            raw_text = "No results returned from Wikipedia search."
            log.warning("[EXTRACT_RAW] - No tool messages found for current iteration.")
        else:
            log.info(f"[EXTRACT_RAW] - Extracted {len(raw_text)} chars of raw Wikipedia content.")

        state["raw_retrieved"] = raw_text
        return state

    # ── Node: Filter Knowledge ───────────────────────────────

    def _filter_knowledge(self, state: State) -> State:
        prompt_text = agent_cfg["filter_knowledge_prompt"].format(
            problem=state["problem"],
            raw_retrieved=state["raw_retrieved"],
        )

        ai = self.base_llm.invoke([HumanMessage(content=prompt_text)])
        log.info(f"[FILTER_KNOWLEDGE] - Output: {ai.content}")

        filtered = ai.content
        state["filtered_this_round"] = filtered

        # Append to accumulated knowledge with round label
        current_iter = state.get("retrieval_iter", 0) + 1
        accumulated = state.get("accumulated_knowledge", "")
        separator = (
            f"\n\n## Retrieval Round {current_iter}\n" if accumulated else f"## Retrieval Round {current_iter}\n"
        )
        state["accumulated_knowledge"] = accumulated + separator + filtered

        log.info(f"[FILTER_KNOWLEDGE] - Accumulated knowledge now {len(state['accumulated_knowledge'])} chars.")

        return state

    # ── Node: Check Sufficiency ──────────────────────────────

    def _check_sufficiency(self, state: State) -> State:
        prompt_text = agent_cfg["sufficiency_check_prompt"].format(
            problem=state["problem"],
            accumulated_knowledge=state["accumulated_knowledge"],
            previous_queries=state.get("previous_queries", []),
        )

        ai = self.base_llm.invoke([HumanMessage(content=prompt_text)])
        log.info(f"[CHECK_SUFFICIENCY] - Output: {ai.content}")

        state["sufficiency_verdict"] = ai.content
        state["retrieval_iter"] = state.get("retrieval_iter", 0) + 1

        return state

    # ── Node: Consolidate Knowledge ──────────────────────────

    def _consolidate_knowledge(self, state: State) -> State:
        accumulated = state.get("accumulated_knowledge", "")

        # If only one retrieval round happened, skip consolidation LLM call
        if state.get("retrieval_iter", 1) <= 1:
            log.info("[CONSOLIDATE] - Single retrieval round. Skipping consolidation LLM call.")
            consolidated = accumulated
        else:
            prompt_text = agent_cfg["consolidate_knowledge_prompt"].format(
                problem=state["problem"],
                accumulated_knowledge=accumulated,
            )

            ai = self.base_llm.invoke([HumanMessage(content=prompt_text)])
            consolidated = ai.content
            log.info(f"[CONSOLIDATE] - Output: {consolidated}")

        # Set up messages for the solver — clean context with only what it needs
        state["messages"] = [
            SystemMessage(content=agent_cfg["main_system_prompt"]),
            AIMessage(content=consolidated),
        ]

        return state

    # ── Node: Solve Problem ──────────────────────────────────

    def _solve_problem(self, state: State) -> State:
        prompt_text = agent_cfg["solve_prompt"].format(
            problem=state["problem"],
            # options=state["options"],
        )

        msgs = state["messages"] + [HumanMessage(content=prompt_text)]
        ai = self.base_llm.invoke(msgs)

        log.info(f"[SOLVE_PROBLEM] - Output: {ai.content}")
        state["messages"] = [ai]
        return state

    # ── Utility ──────────────────────────────────────────────

    @staticmethod
    def _parse_queries(plan_text: str) -> list[str]:
        """
        Extracts queries from a numbered list under '### Search Queries'.
        Expected format:
            ### Search Queries
            1. Projectile motion maximum height formula
            2. Standard acceleration due to gravity value
        Falls back to empty list if the format is not found.
        """
        # Find the "Search Queries" section
        section_match = re.search(r"###\s*Search Queries\s*\n(.*)", plan_text, re.DOTALL | re.IGNORECASE)
        if not section_match:
            log.warning("[PARSE_QUERIES] - Could not find '### Search Queries' section. Returning empty list.")
            return []

        section_text = section_match.group(1)

        # Extract numbered items (e.g., "1. query text", "2. query text")
        queries = re.findall(r"^\s*\d+\.\s*(.+)$", section_text, re.MULTILINE)
        queries = [q.strip().strip('"').strip("'") for q in queries if q.strip()]

        if not queries:
            log.warning("[PARSE_QUERIES] - Found section but no numbered queries inside. Returning empty list.")

        return queries

    # ── Public API ───────────────────────────────────────────

    def solve(self, problem: str) -> str:
        question, options = scieval_split_problem_and_options(full_text=problem)
        log.info(f"[QUESTION] - {question}")
        log.info(f"[OPTIONS] - {options}")

        state: State = {
            "knowledge_messages": [],
            "messages": [],
            "problem": problem,
            # "options": options,
            "retrieval_iter": 0,
            "previous_queries": [],
            "retrieval_plan": "",
            "raw_retrieved": "",
            "filtered_this_round": "",
            "sufficiency_verdict": "",
            "accumulated_knowledge": "",
        }

        final_state = self.graph.invoke(state, config={"recursion_limit": 200})
        msgs = final_state.get("messages", [])

        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                return msg.content

        return msgs[-1].content if msgs else ""
