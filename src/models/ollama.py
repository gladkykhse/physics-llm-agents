import polars as pl
from ollama import ChatResponse, chat


def ollama_completion_request(request: str, system_prompt: str, model: str = "llama3:8b") -> ChatResponse:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": request})

    return chat(
        model=model,
        messages=messages,
        options={"temperature": 0.0},
    )


def run_completion(all_requests: list[str], system_prompt: str | list = "", model: str = "llama3:8b") -> pl.DataFrame:
    if isinstance(system_prompt, list) and len(all_requests) != len(system_prompt):
        raise ValueError(
            "`system_prompt` and `requests_batch` must be both lists of the same length if you want to use different system prompts for different requests."
        )

    results: list[dict] = []
    for request in all_requests:
        response = ollama_completion_request(request=request, system_prompt=system_prompt, model=model)
        results.append(response["message"]["content"])

    return pl.DataFrame(
        {
            "question": all_requests,
            "answer": results,
        }
    )
