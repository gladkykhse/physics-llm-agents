import asyncio
import os
from typing import List, Union

import polars as pl
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

load_dotenv()

async_client = AsyncOpenAI(
    base_url=f"{os.environ['VLLM_HOST']}:{os.environ['VLLM_PORT']}/v1",
    api_key=os.environ["VLLM_API_KEY"],
    timeout=60 * 5,
)


async def vllm_completion_request(
    client: AsyncOpenAI,
    request: str,
    system_prompt: str = "",
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    **kwargs,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": request})

    resp: ChatCompletion = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **({"max_tokens": max_tokens} if max_tokens is not None else {}),
        **kwargs,
    )
    return resp.choices[0].message.content or ""


async def run_completion(
    all_requests: List[str],
    system_prompt: Union[str, List[str]] = "",
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    batch_size: int = 8,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    **kwargs,
) -> pl.DataFrame:
    if isinstance(system_prompt, list):
        if len(system_prompt) != len(all_requests):
            raise ValueError("When passing a list, `system_prompt` length must match `all_requests` length.")
        sys_prompts = system_prompt
    else:
        sys_prompts = [system_prompt] * len(all_requests)

    results: List[str] = []
    client = async_client

    for i in range(0, len(all_requests), batch_size):
        reqs_batch = all_requests[i : i + batch_size]
        prompts_batch = sys_prompts[i : i + batch_size]

        tasks = [
            vllm_completion_request(
                client=client,
                request=req,
                system_prompt=sp,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            for req, sp in zip(reqs_batch, prompts_batch)
        ]

        answers_batch = await asyncio.gather(*tasks)
        for item in answers_batch:
            print(f"Answer length = {len(item)}")
        results.extend(answers_batch)

    return pl.DataFrame({"question": all_requests, "answer": results})
