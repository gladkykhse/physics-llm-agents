import asyncio
import os
import time

import aiohttp
from dotenv import load_dotenv

load_dotenv()

PARALLEL_WINDOW_SECONDS = float(os.environ["PARALLEL_WINDOW_SECONDS"])
MAX_WINDOW_REQUESTS = int(os.environ["MAX_WINDOW_REQUESTS"])


async def openai_completion_request(
    request: str, session: aiohttp.ClientSession, system_prompt: str = "", model: str = "gpt-4o-mini"
) -> dict:
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": ([{"role": "developer", "content": system_prompt}] if system_prompt else [])
        + [{"role": "user", "content": request}],
        "temperature": 0,
    }

    async with session.post(url=url, headers=headers, json=payload) as resp:
        resp.raise_for_status()
        response = await resp.json()
        response["initial_request"] = request
        return response


async def run_batched_completion(
    all_requests: list[str], system_prompt: str | list = "", model: str = "gpt-4o-mini"
) -> list[dict]:
    if isinstance(system_prompt, list) and len(all_requests) != len(system_prompt):
        raise ValueError(
            "`system_prompt` and `requests_batch` must be both lists of the same length if you want to use different system prompts for different requests."
        )

    timeout = aiohttp.ClientTimeout(total=1800)
    results: list[dict] = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        n = len(all_requests)
        for batch_n, start_idx in enumerate(range(0, n, MAX_WINDOW_REQUESTS), start=1):
            window_start = time.monotonic()
            end_idx = min(start_idx + MAX_WINDOW_REQUESTS, n)
            print(f"Processing batch {batch_n}: items {start_idx}..{end_idx - 1}")

            tasks = [
                asyncio.create_task(
                    openai_completion_request(
                        request=all_requests[i],
                        session=session,
                        system_prompt=(system_prompt[i] if isinstance(system_prompt, list) else system_prompt),
                        model=model,
                    )
                )
                for i in range(start_idx, end_idx)
            ]

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            elapsed = time.monotonic() - window_start
            print(f"Batch {batch_n} processed in {elapsed:.2f} seconds.")

            remaining = PARALLEL_WINDOW_SECONDS - elapsed
            if remaining > 0 and end_idx < n:
                await asyncio.sleep(remaining)

    return results
