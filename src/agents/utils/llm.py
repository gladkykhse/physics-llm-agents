import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def make_llm(model: str = "meta-llama/Llama-3.1-8B-Instruct", temperature: int = 0.2, max_tokens: int = 2048) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=f"{os.environ['VLLM_HOST']}:{os.environ['VLLM_PORT']}/v1",
        api_key=os.environ["VLLM_API_KEY"],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
