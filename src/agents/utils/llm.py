import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def make_llm(
    model: str = "/home/s_gladkykh/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=f"{os.environ['VLLM_HOST']}:{os.environ['VLLM_PORT']}/v1",
        api_key=os.environ["VLLM_API_KEY"],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
