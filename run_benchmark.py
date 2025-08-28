import asyncio
import os
import time

import aiohttp
import polars as pl
from dotenv import load_dotenv

load_dotenv()

PARALLEL_WINDOW_SECONDS = float(os.environ["PARALLEL_WINDOW_SECONDS"])
MAX_WINDOW_REQUESTS = int(os.environ["MAX_WINDOW_REQUESTS"])


def load_scieval(
    source: str = "benchmarks/SciEval/test/data-00000-of-00001.arrow",
) -> pl.DataFrame:
    df_scieval = pl.read_ipc_stream(source=source).filter(pl.col("category") == "physics")

    agg_cols = [c for c in df_scieval.columns if c != "question"]

    df_scieval = df_scieval.group_by("question", maintain_order=True).agg(
        [pl.col(c).drop_nulls().first().alias(c) for c in agg_cols]
    )

    df_scieval = df_scieval.with_columns(
        pl.col("question").str.strip_chars().str.strip_suffix("Answer:").str.strip_chars()
    )

    return df_scieval


def cot_system_prompt() -> str:
    return """You are an expert in physics, solving multiple-choice exam problems.
    Carefully analyze the question using relevant physics principles, formulas, and reasoning.
    Explain your thought process step by step to show how you arrive at the solution.

    After reasoning, provide your final choice in the format:
    Answer: A
    Answer: B
    Answer: C
    or
    Answer: D

    Do not include anything else after the final answer."""


def standard_system_prompt() -> str:
    return """Given a physics question and four options, please select the right answer.
    Your answer should be a single letter A, B, C or D.
    Please directly give the answer without any explanation."""


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


async def main() -> None:
    df_scieval = load_scieval()

    for prompt_fn in (standard_system_prompt, cot_system_prompt):
        questions = df_scieval["question"].to_list()
        responses = await run_batched_completion(all_requests=questions, system_prompt=prompt_fn())

        evaluation_df = pl.DataFrame(
            {
                "question": [response["initial_request"] for response in responses],
                f"answer_{prompt_fn.__name__}": [
                    response["choices"][0]["message"]["content"] for response in responses
                ],
            }
        )

        df_scieval = df_scieval.join(evaluation_df, on="question")

    os.makedirs("artifacts", exist_ok=True)
    df_scieval.write_parquet(file="artifacts/evaluation.parquet")


if __name__ == "__main__":
    asyncio.run(main())
