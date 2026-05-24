import logging

from langchain_core.tools import tool
from llama_index.tools.wikipedia import WikipediaToolSpec

log = logging.getLogger(__name__)

_wikipedia = WikipediaToolSpec()


@tool
def wikipedia_multi_search(queries: list[str]) -> str:
    """
    Performs multiple Wikipedia searches simultaneously.

    Args:
        queries: A list of concise entity-based search terms.

    Returns:
        A structured report with unique sections grouped by query.
    """
    output = ["# Wikipedia Search Results\n"]

    try:
        for query in queries:
            result = _wikipedia.search_data(query)

            output.append(f"## Query: {query}")
            output.append(str(result))
            output.append("---\n")

        res = "\n".join(output)
        log.info("[LLAMAINDEX_WIKI_SEARCH] Queries: %s", queries)
        return res

    except Exception as e:
        log.error("[LLAMAINDEX_WIKI_SEARCH] Error: %s", e)
        return f"Search failed due to an error: {e}"
