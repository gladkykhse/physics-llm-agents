import json
import logging
import time
from datetime import timedelta

import wikipedia
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

log = logging.getLogger(__name__)

wikipedia.set_user_agent(
    "agent-wikipedia-baseline-eval/0.1 "
    "(gladkykh.sviatoslav@gmail.com)"
)
wikipedia.set_rate_limiting(True, min_wait=timedelta(seconds=2))

_wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=1500,
    )
)

_MAX_RETRIES = 5
_BACKOFF_BASE = 3.0  # seconds


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


def _run_single_query(query: str) -> str:
    """Run a single Wikipedia query with retry for transient Wikipedia/API failures."""
    last_error = None

    for attempt in range(_MAX_RETRIES):
        try:
            result = _wikipedia.run(query)

            if result and str(result).strip():
                return result

            raise RuntimeError("Empty Wikipedia result")

        except Exception as e:
            last_error = e

            if not _is_retryable_error(e):
                raise

            if attempt < _MAX_RETRIES - 1:
                delay = _BACKOFF_BASE * (2 ** attempt)
                log.warning(
                    "[LANGCHAIN_WIKI_SEARCH] Transient Wikipedia failure for '%s' "
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

    # Minimal safety: dedupe and cap queries, but keep same interface.
    seen = set()
    clean_queries = []
    for query in queries:
        q = str(query).strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            clean_queries.append(q)

    for query in clean_queries[:4]:
        try:
            result = _run_single_query(query)
            output.append(f"## Query: {query}")
            output.append(str(result))
            output.append("---\n")
            succeeded += 1
        except Exception as e:
            log.error("[LANGCHAIN_WIKI_SEARCH] Query '%s' failed: %s", query, e)
            failed += 1

    log.info(
        "[LANGCHAIN_WIKI_SEARCH] Queries: %s | Succeeded: %d, Failed: %d",
        clean_queries[:4],
        succeeded,
        failed,
    )

    if succeeded == 0:
        return "[No retrieval results available]"

    return "\n".join(output)