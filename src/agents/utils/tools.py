import logging as log
import math
import os
import re
from typing import Optional

import httpx
import numpy as np
import sympy as sp
from dotenv import load_dotenv
from langchain_core.tools import tool

from src.knowledge_bases.vector_rag import PgVectorRetriever
from src.utils.helpers import load_yaml

load_dotenv()
log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_URL = os.getenv("RETRIEVER_URL", "https://wiki.sgcore.dev").rstrip("/") + "/retrieve"
_TOKEN = os.getenv("RETRIEVER_API_TOKEN", "")

_client = httpx.Client(
    headers={"Authorization": f"Bearer {_TOKEN}"},
    timeout=120.0,
    transport=httpx.HTTPTransport(retries=3),
)
vector_rag_cfg = load_yaml("config/vector_rag.yaml")
dsn = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)
retriever_backend = PgVectorRetriever(
    dsn=dsn,
    table=vector_rag_cfg["table"],
    model=vector_rag_cfg["embedding_model"],
    memory=True,
)


@tool
def retrieve_physics_theory(query: str) -> str:
    """
    Retrieves physics theory excerpts from a textbook index via semantic search.

    Args:
        query: A short physics concept or phrase.
            Example: "conservation of energy"
            Example: "moment of inertia formula"

    Returns:
        Relevant textbook excerpts with source citations.
    """
    top_k = int(vector_rag_cfg["retrieve_top_k"])
    results = retriever_backend(query=query, top_k=top_k)

    chunks_plain_text = ""
    for i, r in enumerate(results, 1):
        chunks_plain_text += f"{i}. Source: {r['source']} (score: {r['score']:.3f})\n"
        chunks_plain_text += f"   Content: {r['text']}\n\n"

    response = "Observation:\n"

    if len(results) == 0:
        response += (
            f"No new physics textbook excerpts were returned for the query `{query}`.\n"
            "The top matches for this query were already retrieved earlier and are present in the previous tool call outputs (observations).\n"
            "Next:\n"
            "- If the needed theory is already in your current context, use it and continue solving the problem step-by-step.\n"
            "- If it is NOT in the current context, proceed solving the problem step-by-step using your own knowledge as if you know the missing theory (do not get stuck on retrieval)."
        )
    elif len(results) < vector_rag_cfg["retrieve_top_k"]:
        missing = top_k - len(results)
        response += (
            f"Only {len(results)} of the requested top-{top_k} most similar excerpts are NEW for the query `{query}`.\n"
            f"The remaining {missing} highly similar excerpts were already retrieved earlier and are present in the previous tool call outputs (observations).\n"
            f"New chunks:\n{chunks_plain_text}\n\n"
            "Next:\n"
            "- If the needed theory is already in your current context, use it and continue solving the problem step-by-step further.\n"
            "- If it is NOT in the current context, proceed solving the problem step-by-step using your own knowledge as if you know the missing theory (do not get stuck on retrieval).\n"
            "- Optionally, you may retry retrieving physics theory ONLY ONCE again with a more general and suitable query.\n\n"
        )
    else:
        response += f"Extracted {vector_rag_cfg['retrieve_top_k']} new excerpts for the provided query: `{query}`\n\nNew chunks:\n{chunks_plain_text}"

    log.info(f"[RETRIEVER] - Tool output: {response[:200]}\n\n...\n\n{response[-200:]}")

    return response


def has_plain_text(expr: str) -> bool:
    # Words like "velocity", "distance", "find"
    matches = re.findall(r"\s[a-zA-Z]{3,}\s", expr)

    # Allow a small whitelist of math words
    allowed = {"sin", "cos", "tan", "log", "exp", "sqrt", "abs", "pi"}

    for m in matches:
        if m.lower() not in allowed:
            return True
    return False


def fix_integrate_syntax(expr: str) -> str:
    """
    Converts integrate(expr, var, lo, hi) to integrate(expr, (var, lo, hi))
    which is the correct SymPy syntax for definite integrals.
    """
    pattern = r"integrate\((.+?),\s*([a-zA-Z_]\w*)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\)"
    replacement = r"integrate(\1, (\2, \3, \4))"
    return re.sub(pattern, replacement, expr)


@tool
def sympy_eval(expression: str) -> str:
    """
    Evaluates purely numerical arithmetic and calculus expressions.

    Args:
        expression (str): A strictly mathematical expression containing numbers, operators, or basic calculus functions.

    Returns:
        str: The calculated numerical value.
    """
    if has_plain_text(expression):
        return (
            "Invalid expression: contains plain text. "
            "Provide only a mathematical numerical expression."
        )

    expression = (
        expression
        .replace("sympy.", "")
        .replace("sp.", "")
        .replace("math.", "")
        .replace("np.", "")
        .replace("numpy.", "")
        .strip()
    )
    expression = fix_integrate_syntax(expression)
    try:
        expr = sp.sympify(expression)
    except Exception as e:
        log.info(f"[SYMPY] - Parsing Exception: {e}")
        return (
            "Parsing Exception: Invalid expression. "
            "REMINDER: sympy_eval ONLY accepts pure math (numbers and operators). "
            "It CANNOT evaluate text, physics rules, or equations with '='. "
            "If you are answering a conceptual/theory question, STOP using tools and generate the final answer directly."
        )

    if getattr(expr, "free_symbols", None):
        return (
            f"Error: unresolved symbol(s) {sorted(str(s) for s in expr.free_symbols)}. "
            "sympy_eval is only for direct numerical evaluation. Substitute values first, or use sympy_solve if you need to solve for an unknown."
        )

    if isinstance(expr, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse, bool)):
        return str(bool(expr))

    if isinstance(expr, (list, tuple)) or type(expr).__name__ in ["Tuple", "tuple"]:
        return "Error: sympy_eval does not support lists or tuples. You must evaluate vector components individually."

    try:
        res = str(expr.evalf())
    except Exception as e:
        log.info(f"[SYMPY] - Evaluation Exception: {e}")
        return f"An error caught during expression evaluation: {e}"

    log.info(f"[SYMPY] - SymPy result: {res}")
    return res


@tool
def vector_math(operation: str, v1: str, v2: Optional[str] = None, scalar: Optional[str] = None) -> str:
    """
    Performs specific vector algebra operations on mathematical vectors.

    Args:
        operation (str): The name of the operation ('magnitude', 'distance', 'normalize', 'dot', 'cross', 'add', 'sub', 'scale', 'angle').
        v1 (str): First vector formatted as a string list, e.g., "[1.0, 2.5, 3.0]".
        v2 (str, optional): Second vector formatted as a string list.
        scalar (str, optional): A scalar numerical value as a string.

    Returns:
        str: The resulting vector or scalar value.
    """
    allowed_ops = {"magnitude", "distance", "normalize", "dot", "cross", "add", "sub", "scale", "angle"}
    same_dim_ops = {"distance", "dot", "add", "sub", "angle"}

    if v1:
        v1 = v1.replace("<", "[").replace(">", "]")

    if v2:
        v2 = v2.replace("<", "[").replace(">", "]")

    if has_plain_text(v1) or (v2 and has_plain_text(v2)) or (scalar and has_plain_text(scalar)):
        return (
            "Invalid vector input: contains plain text. "
            "Vectors must be numeric lists like [1, 2, 3]."
        )

    safe_dict = {
        "math": math,
        "np": np,
        "numpy": np,
        "sp": sp,
        "sympy": sp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "exp": math.exp,
        "log": math.log,
        "abs": abs,
    }

    try:
        # 1. Parse v1 (Mandatory)
        # Pass safe_dict as both globals and locals to ensure access
        vec1 = eval(v1, safe_dict, safe_dict)
        if not isinstance(vec1, (list, tuple)):
            return f"Error: v1 must be a list, got {type(vec1)}"
        a = np.array(vec1, dtype=float)

        # 2. Parse v2 (Optional)
        b = None
        if v2 and v2.strip().lower() != "none":
            vec2 = eval(v2, safe_dict, safe_dict)
            if not isinstance(vec2, (list, tuple)):
                return f"Error: v2 must be a list, got {type(vec2)}"
            b = np.array(vec2, dtype=float)

        # 3. Parse scalar (Optional)
        s_val = None
        if scalar and str(scalar).strip().lower() != "none":
            s_val = eval(scalar, safe_dict, safe_dict)
            if not isinstance(s_val, (int, float)):
                return f"Error: scalar must be a number, got {type(s_val)}"

        if a.ndim != 1:
            return "Error: v1 must be a 1D vector."

        if b is not None and b.ndim != 1:
            return "Error: v2 must be a 1D vector."

        if operation in same_dim_ops:
            if b is not None and len(a) != len(b):
                return f"Error: vectors must have the same dimension, got {len(a)} and {len(b)}."

        # 4. Perform Operations
        res = ""
        if operation == "magnitude":
            if b is not None:
                return "Error: 'magnitude' only takes v1. If you want the distance between two vectors, use operation='distance'."
            res = str(np.linalg.norm(a))
        elif operation == "distance":
            res = "Error: 'distance' needs v2" if b is None else str(np.linalg.norm(a - b))
        elif operation == "normalize":
            norm = np.linalg.norm(a)
            res = "Error: Zero vector" if norm == 0 else str((a / norm).tolist())
        elif operation == "dot":
            res = "Error: needs v2" if b is None else str(np.dot(a, b))
        elif operation == "cross":
            res = "Error: needs v2" if b is None else str(np.cross(a, b).tolist())
        elif operation == "add":
            res = "Error: needs v2" if b is None else str((a + b).tolist())
        elif operation == "sub":
            res = "Error: needs v2" if b is None else str((a - b).tolist())
        elif operation == "scale":
            res = "Error: needs scalar" if s_val is None else str((a * s_val).tolist())
        elif operation == "angle":
            if b is None:
                if len(a) >= 2:
                    res = str(np.degrees(np.arctan2(a[1], a[0])))
                else:
                    res = "Error: needs v2 for 1D vectors"
            else:
                cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                res = str(np.degrees(angle_rad))
        else:
            res = (
                f"Error: Unknown operation '{operation}'. "
                f"Allowed operations are: {allowed_ops}."
            )

        log.info(f"[VECTOR_MATH] - Result: {res}")
        return res

    except Exception as e:
        log.error(f"[VECTOR_MATH] - Error: {e}")
        return f"Error: {str(e)}"


@tool
def sympy_solve(equation: str, symbol: Optional[str] = None) -> str:
    """
    Solves an algebraic equation to isolate and find the value of a specific unknown variable.

    Args:
        equation (str): The algebraic equation containing numbers and ONE unknown variable.
        symbol (str, optional): The exact letter of the variable to solve for (e.g., "t", "v").

    Returns:
        str: The numerical solution(s) for the requested symbol.
    """
    if has_plain_text(equation):
        return (
            "Invalid expression: contains plain text. "
            "Provide only a mathematical equation (numbers, variables, operators)."
        )

    try:
        # 1. Preprocessing: Handle common syntax issues
        # equation = equation.replace("sympy.", "").replace("sp.", "").replace("math.", "")
        raw_eq = (
            equation
            .replace("^", "**")
            .replace("sympy.", "")
            .replace("sp.", "")
            .replace("math.", "")
            .replace("np.", "")
            .replace("numpy.", "")
            .replace("==", "=")
            .strip()
        )

        if symbol:
            symbol = symbol.strip()

        # 2. Parse the Equation (Handle "=")
        if "=" in raw_eq:
            lhs_str, rhs_str = raw_eq.split("=", 1)
            lhs = sp.sympify(lhs_str)
            rhs = sp.sympify(rhs_str)
            eq_obj = sp.Eq(lhs, rhs)
        else:
            # If no '=', assume expression equals zero (e.g. "x**2 - 4")
            eq_obj = sp.sympify(raw_eq)

        free_syms = eq_obj.free_symbols

        if symbol:
            sym_obj = sp.Symbol(symbol)

            if sym_obj not in free_syms:
                return (
                    f"Error: symbol '{symbol}' is not present in the equation. "
                    "If the expression is already numeric or the variable is already isolated, use sympy_eval instead."
                )

        else:
            if len(free_syms) == 0:
                return (
                    "Error: no unknown variable found. "
                    "If this is a direct numerical calculation, use sympy_eval instead."
                )

        # 3. Identify the Symbol
        if symbol:
            sym_obj = sp.Symbol(symbol)
        else:
            # Auto-detect symbol if only one free symbol exists
            free_syms = eq_obj.free_symbols
            if len(free_syms) == 1:
                sym_obj = list(free_syms)[0]
            else:
                return f"Error: The equation has multiple variables {free_syms}. Please specify which 'symbol' to solve for."

        # 4. Solve
        solutions = sp.solve(eq_obj, sym_obj)

        if len(solutions) == 0:
            return (
                f"Error: no solution found for {sym_obj}. "
                "Check whether the equation is written correctly and whether you used sympy_solve for the right unknown."
            )

        # 5. Format Output
        formatted_sols = []
        for sol in solutions:
            try:
                if sol.is_number:
                    formatted_sols.append(str(sol.evalf()))
                else:
                    formatted_sols.append(str(sol))
            except Exception as _:
                formatted_sols.append(str(sol))

        result = f"Solutions for {sym_obj}: {formatted_sols}"

        log.info(f"[SYMPY_SOLVE] - Result: {result}")

        return result

    except Exception as e:
        log.info(f"[SYMPY_SOLVE] - Error: {e}")
        return f"Error solving equation: {str(e)}"


def _retrieve(queries: list[str], k: int, page_limit: int, section_limit_per_page: int) -> list[list[dict]]:
    if not _TOKEN:
        raise RuntimeError("RETRIEVER_API_TOKEN env var is not set")
    resp = _client.post(
        _URL,
        json={"queries": queries, "k": k, "page_limit": page_limit, "section_limit_per_page": section_limit_per_page},
    )
    resp.raise_for_status()
    return resp.json()["results"]


@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for physics concepts, laws, constants, or named theorems.

    Args:
        query: Concise entity-based search term (2-6 words).

    Returns:
        Relevant text passages from matching Wikipedia sections.
    """
    try:
        hits = _retrieve([query], k=3, page_limit=7, section_limit_per_page=15)[0]
        output = [f"--- Article: {h['title']} | Section: {h['section_title']} ---\n{h['text']}" for h in hits]
        res = "\n\n".join(output)
        log.info("[WIKI_SEARCH] Query: %r | Found %d sections.", query, len(hits))
        return res
    except Exception as e:
        log.error("[WIKI_SEARCH] Error: %s", e)
        return f"Search failed due to an error: {e}"


@tool
def wikipedia_multi_search(queries: list[str]) -> str:
    """
    Performs multiple Wikipedia searches simultaneously.

    Args:
        queries: A list of concise entity-based search terms.

    Returns:
        A structured report with unique sections grouped by query.
    """
    seen: set[tuple[int, str]] = set()
    output = ["# Wikipedia Search Results\n"]
    try:
        results = _retrieve(queries, k=3, page_limit=5, section_limit_per_page=10)
        for query, hits in zip(queries, results):
            fresh = []
            for h in hits:
                key = (h["pageid"], h["section_title"])
                if key not in seen:
                    seen.add(key)
                    fresh.append(h)
            if not fresh:
                continue
            output.append(f"## Query: {query}")
            for i, h in enumerate(fresh, 1):
                output.append(f"### Section {i}: {h['title']} - {h['section_title']}")
                output.append(f"{h['text']}\n")
                output.append("---\n")
        if not seen:
            return "No relevant Wikipedia sections found for the provided queries."
        res = "\n".join(output)
        log.info("[MULTI_WIKI_SEARCH] Queries: %s | Found %d unique sections.", queries, len(seen))
        return res
    except Exception as e:
        log.error("[MULTI_WIKI_SEARCH] Error: %s", e)
        return f"Search failed due to an error: {e}"
