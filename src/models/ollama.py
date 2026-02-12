import asyncio

import polars as pl
from ollama import AsyncClient


async def ollama_completion_request(
    client: AsyncClient,
    request: str,
    system_prompt: str,
    model: str = "llama3:8b",
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": request})

    resp = await client.chat(
        model=model,
        messages=messages,
        options={"temperature": 0.0},
    )
    return resp["message"]["content"]


async def run_completion(
    all_requests: list[str],
    system_prompt: str | list = "",
    model: str = "llama3:8b",
    batch_size: int = 8,
) -> pl.DataFrame:
    if isinstance(system_prompt, list):
        sys_prompts = system_prompt
    else:
        sys_prompts = [system_prompt] * len(all_requests)

    results: list[str] = []
    client = AsyncClient()  # single shared client; no close() needed currently
    for i in range(0, len(all_requests), batch_size):
        reqs_batch = all_requests[i : i + batch_size]
        prompts_batch = sys_prompts[i : i + batch_size]
        tasks = [ollama_completion_request(client, req, sp, model=model) for req, sp in zip(reqs_batch, prompts_batch)]
        answers = await asyncio.gather(*tasks)  # preserves order
        results.extend(answers)

    return pl.DataFrame({"question": all_requests, "answer_ai": results})
