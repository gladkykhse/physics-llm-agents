import os

from dotenv import load_dotenv
from langchain_core.tools import tool

from src.knowledge_bases.vector_rag import PgVectorRetriever
from src.utils.helpers import load_yaml

load_dotenv()

vector_rag_cfg = load_yaml("config/vector_rag.yaml")
dsn = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)
retriever_backend = PgVectorRetriever(
    dsn=dsn,
    table=vector_rag_cfg["table"],
    model=vector_rag_cfg["embedding_model"],
    top_k=vector_rag_cfg["retrieve_top_k"],
)


@tool
def retriever(query: str) -> str:
    """
    This tool allows searching for authoritative physics textbook knowledge including
    - Fundamental concepts and definitions
    - Mathematical formulas and equations with explanations
    - Physical laws, theorems, and principles with applications
    - Constants, units, and measurement systems
    - Theoretical foundations and derivations

    CRITICAL CONSTRAINTS:
    - ONLY use for physics-related queries - this tool cannot answer questions about other subjects
    - Be specific and precise in your queries to get the most relevant results

    The tool returns the most relevant textbook passages with similarity scores.
    Use this when you need authoritative physics information to solve problems.

    Args:
        query: A specific, physics-related search query using appropriate terminology

    Returns:
        str: Formatted results containing relevant textbook passages with sources and similarity scores
    """
    results = retriever_backend(query=query)

    response = f"Found {len(results)} results for '{query}':\n\n"
    for i, r in enumerate(results, 1):
        response += f"{i}. Source: {r['source']} (score: {r['score']:.3f})\n"
        response += f"   Content: {r['text']}\n\n"

    return response


@tool
def finalize_solution() -> str:
    """
    This tool indicates that you have completed all reasoning steps and are ready to provide the final solution.

    Call this tool ONLY when:
    - You have executed all necessary steps from your plan
    - You have retrieved all required physics information using available tools
    - You have performed all calculations and reasoning
    - You are confident in your solution and ready to present the final answer

    DO NOT call this tool if:
    - You are still working through steps in your execution plan
    - You need to retrieve additional information or perform more calculations
    - You are uncertain about any part of the solution
    - You have not verified your answer matches the problem requirements

    This tool should be the final step in your problem-solving process, indicating
    that no further reasoning or tool usage is required, and you are prepared to
    synthesize everything into a complete, final answer.

    Returns:
        str: Confirmation that the system will proceed to final answer generation
    """
    return "Ready for final answer generation"
