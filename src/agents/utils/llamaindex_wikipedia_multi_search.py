import json
import logging
import time
from datetime import timedelta

import wikipedia
from langchain_core.tools import tool
from llama_index.tools.wikipedia import WikipediaToolSpec

log = logging.getLogger(__name__)

wikipedia.set_user_agent("agent-wikipedia-baseline-eval/0.1 (gladkykh.sviatoslav@gmail.com)")
wikipedia.set_rate_limiting(True, min_wait=timedelta(seconds=2))

_wikipedia = WikipediaToolSpec()

_MAX_RETRIES = 5
_BACKOFF_BASE = 3.0
_MAX_RETRIEVAL_CHARS_PER_QUERY = 20_000


def _is_retryable_error(e: Exception) -> bool:
    text = repr(e).lower()
    return (
        isinstance(e, json.JSONDecodeError)
        or "expecting value" in text
        or "empty" in text
        or "timeout" in text
        or "temporarily" in text
        or "connection" in text
        or "429" in text
        or "502" in text
        or "503" in text
        or "504" in text
    )


def _clip_retrieval(text: str) -> str:
    text = str(text)

    if len(text) <= _MAX_RETRIEVAL_CHARS_PER_QUERY:
        return text

    return text[:_MAX_RETRIEVAL_CHARS_PER_QUERY].rstrip() + "\n...[retrieval truncated]"


def _run_single_query(query: str) -> str:
    last_error = None

    for attempt in range(_MAX_RETRIES):
        try:
            result = _wikipedia.search_data(query)

            if result and str(result).strip():
                return str(result)

            raise RuntimeError("Empty Wikipedia result")

        except Exception as e:
            last_error = e

            if not _is_retryable_error(e):
                raise

            if attempt < _MAX_RETRIES - 1:
                delay = _BACKOFF_BASE * (2**attempt)
                log.warning(
                    "[LLAMAINDEX_WIKI_SEARCH] Transient Wikipedia failure for '%s' "
                    "(attempt %d/%d, retrying in %.1fs): %s",
                    query,
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                    e,
                )
                time.sleep(delay)
            else:
                raise last_error


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
    succeeded = 0
    failed = 0

    seen = set()
    clean_queries = []
    for query in queries:
        q = str(query).strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            clean_queries.append(q)

    for query in clean_queries[:4]:
        try:
            result = _clip_retrieval(_run_single_query(query))
            output.append(f"## Query: {query}")
            output.append(result)
            output.append("---\n")
            succeeded += 1
        except Exception as e:
            log.error("[LLAMAINDEX_WIKI_SEARCH] Query '%s' failed: %s", query, e)
            failed += 1

    log.info(
        "[LLAMAINDEX_WIKI_SEARCH] Queries: %s | Succeeded: %d, Failed: %d",
        clean_queries[:4],
        succeeded,
        failed,
    )

    if succeeded == 0:
        return "[No retrieval results available]"

    return "\n".join(output)
