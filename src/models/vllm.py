import os

import polars as pl
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

load_dotenv()
client = OpenAI(
    base_url=f"http://{os.environ['VLLM_HOST']}:{os.environ['VLLM_PORT']}/v1", api_key=os.environ["VLLM_API_KEY"]
)


def vllm_completion_request(
    request: str,
    system_prompt: str = "",
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    temperature: float = 0.0,
) -> ChatCompletion:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": request})

    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )


def run_completion(
    all_requests: list[str],
    system_prompt: str | list[str] = "",
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
) -> pl.DataFrame:
    if isinstance(system_prompt, list) and len(all_requests) != len(system_prompt):
        raise ValueError(
            "`system_prompt` and `all_requests` must be lists of the same length if you want "
            "different system prompts per request."
        )

    results: list[str] = []
    for i, request in enumerate(all_requests):
        sp = system_prompt[i] if isinstance(system_prompt, list) else system_prompt
        answer = vllm_completion_request(request=request, system_prompt=sp, model=model)
        results.append(answer.choices[0].message.content)

    return pl.DataFrame({"question": all_requests, "answer": results})
