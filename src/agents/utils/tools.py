import os
import logging as log

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
def retriever(query: str) -> str:
    """
    Retrieve short theory excerpts from a physics textbook using semantic similarity search.

    Physics textbook content includes definitions, laws, general equations, conceptual explanations, etc.
    The source textbook contains only theoretical knowledge. It does not include worked examples, numeric solutions, or problem-specific procedures.

    The query for this tool should specify the theoretical information you need for conceptual grounding—something you do not remember well or want to verify.
    Queries should relate to theoretical topics, not specific problems or scenarios.

    Examples of valid concept queries include:
    "Newton's second law", "Mass–energy equivalence", "Simple harmonic motion", "Electric potential gradient", "Angular momentum conservation".

    Problem-specific queries, especially those containing numbers or detailed scenarios, will produce low-quality and low-similarity results.

    If the retrieved results are low-quality or unhelpful, you may reformulate the query and try again.
    However, if several attempts still fail, assume the textbook does not contain the information and continue using your own knowledge to solve the subproblem.
    """
    top_k = int(vector_rag_cfg["retrieve_top_k"])
    results = retriever_backend(query=query, top_k=top_k)

    if len(results) == 0:
        return (
            f"No new chunks were returned for the query `{query}`.\n\n"
            "Most likely, the top matches for this phrasing were already retrieved earlier and are present in the current "
            "context, or the retriever could not find sufficiently relevant theory for this exact wording.\n\n"
            "What you can do next:\n"
            "- Reformulate the query more generally and with canonical terms (e.g., the law/theorem name).\n"
            "- Proceed using the theory already present in your context if it’s sufficient for grounding.\n"
        )
    if len(results) < vector_rag_cfg["retrieve_top_k"]:
        missing = top_k - len(results)
        response = (
            f"Only {len(results)} of the requested top-{top_k} most similar chunks are NEW for the query `{query}`.\n"
            f"The remaining {missing} highly similar chunk(s) were likely retrieved earlier and already exist in your "
            "conversation context.\n\n"
            "If you still need additional conceptual grounding, try reformulating the query more generally (prefer canonical "
            "terminology) and call this tool again; otherwise proceed with the context you already extracted.\n\n"
            "New chunks:\n\n"
        )
    else:
        response = (
            f"Extracted {vector_rag_cfg['retrieve_top_k']} new chunks "
            f"for the provided query: `{query}` \nNew chunks:\n\n"
        )

    for i, r in enumerate(results, 1):
        response += f"{i}. Source: {r['source']} (score: {r['score']:.3f})\n"
        response += f"   Content: {r['text']}\n\n"

    log.info(f"[RETRIEVER] - Tool output: {response[:200]}...")

    return response
