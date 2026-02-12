import logging as log
import os
from typing import Optional

import numpy as np
import sympy as sp
from dotenv import load_dotenv
from langchain_core.tools import tool

from src.knowledge_bases.vector_rag import PgVectorRetriever
from src.utils.helpers import load_yaml

load_dotenv()
log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
    Retrieves short physics theory excerpts from a textbook index via semantic similarity search.

    Use this tool to find definitions, laws, principles, and standard formulas.
    Do NOT use this for calculating values or finding worked examples.

    Args:
        query (str): A short, independent physics concept or phrase (e.g., "conservation of energy",
                     "moment of inertia formula"). Must NOT contain specific problem values,
                     numbers, or full sentences.

    Returns:
        str: A text block containing the most relevant textbook excerpts with source citations.
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


@tool
def sympy_eval(expression: str) -> str:
    """
    Evaluates a fully numeric mathematical expression using SymPy.

    Use this tool for all arithmetic calculations.
    The expression must be purely numeric; it cannot solve for variables.

    Args:
        expression (str): A valid string representation of a mathematical expression
                          (e.g., "9.81 * 10**2 / 2", "sin(30*pi/180)").
                          Must NOT contain symbols (x, y), equations (=), or text.

    Returns:
        str: The numeric result of the evaluation as a string.
    """
    expression = expression.strip().replace("math.", "").replace("sympy.", "")
    try:
        expr = sp.sympify(expression)
    except Exception as e:
        log.info(f"[SYMPY] - Parsing Exception: {e}")
        return (
            f"An error caught during expression parsing: {e}\n"
            "Please ensure that the expression is purely numerical and mathematically valid (no variables, no equations, etc.)"
        )

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
    Performs vector operations on lists of numbers. Use this for ALL vector tasks.

    Args:
        operation (str): 'magnitude', 'normalize', 'dot', 'cross', 'add', 'sub', 'scale', 'angle'.
        v1 (str): First vector as a string list, e.g., "[1, 2, 3]".
        v2 (str, optional): Second vector as a string list.
        scalar (str, optional): Scalar value as a string.
    """
    try:
        # 1. Parse v1 (Mandatory)
        vec1 = eval(v1)
        if not isinstance(vec1, list):
            return f"Error: v1 must be a list, got {type(vec1)}"
        a = np.array(vec1, dtype=float)

        # 2. Parse v2 (Optional)
        b = None
        if v2:
            vec2 = eval(v2)
            if not isinstance(vec2, list):
                return f"Error: v2 must be a list, got {type(vec2)}"
            b = np.array(vec2, dtype=float)

        # 3. Parse scalar (Optional)
        s_val = None
        if scalar:
            s_val = eval(scalar)
            if not isinstance(s_val, (int, float)):
                return f"Error: scalar must be a number, got {type(s_val)}"

        # 4. Perform Operations
        res = ""
        if operation == "magnitude":
            res = str(np.linalg.norm(a))
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
                res = "Error: needs v2"
            else:
                cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                res = str(np.degrees(angle_rad))
        else:
            res = f"Error: Unknown operation '{operation}'"

        log.info(f"[VECTOR_MATH] - Result: {res}")
        return res

    except Exception as e:
        log.error(f"[VECTOR_MATH] - Error: {e}")
        return f"Error: {str(e)}"


@tool
def sympy_solve(equation: str, symbol: Optional[str] = None) -> str:
    """
    Solves a mathematical equation for a specific variable.

    Args:
        equation (str): The equation to solve as a string.
                        e.g. "x**2 - 5*x + 6 = 0" or "v = u + a*t".
                        Supports standard operators and '='.
        symbol (str, optional): The variable to isolate/solve for (e.g., "x", "t").
                                If not provided, SymPy will attempt to guess.

    Returns:
        str: A list of solutions.
    """
    try:
        # 1. Preprocessing: Handle common syntax issues
        raw_eq = equation.replace("^", "**").strip()

        # 2. Parse the Equation (Handle "=")
        if "=" in raw_eq:
            lhs_str, rhs_str = raw_eq.split("=", 1)
            lhs = sp.sympify(lhs_str)
            rhs = sp.sympify(rhs_str)
            eq_obj = sp.Eq(lhs, rhs)
        else:
            # If no '=', assume expression equals zero (e.g. "x**2 - 4")
            eq_obj = sp.sympify(raw_eq)

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
