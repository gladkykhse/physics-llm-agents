import sys
from typing import Union, List

import polars as pl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_tokenizer(model: str) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except AttributeError as e:
        if "vocab_size" not in str(e):
            raise
        # ChatGLM3 tokenizer bug: get_vocab() is called during __init__ before
        # sp_model is set (Python 3.13 + transformers >=4.44 incompatibility).
        # The class is now in sys.modules — patch get_vocab and retry.
        for name, mod in sys.modules.items():
            if "tokenization_chatglm" not in name:
                continue
            cls = getattr(mod, "ChatGLMTokenizer", None)
            if cls is not None and callable(getattr(cls, "get_vocab", None)):
                _orig = cls.get_vocab
                def _safe_get_vocab(self, _orig=_orig):
                    if not hasattr(self, "sp_model"):
                        return {}
                    return _orig(self)
                cls.get_vocab = _safe_get_vocab
                break
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def _build_prompt(tokenizer: AutoTokenizer, request: str, system_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": request})

    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback for models without a chat template
    parts = []
    for msg in messages:
        parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
    parts.append("Assistant:")
    return "\n".join(parts)


def run_completion(
    all_requests: List[str],
    system_prompt: Union[str, List[str]] = "",
    model: str = "zd21/SciGLM-6B",
    batch_size: int = 4,
    max_new_tokens: int = 2048,
    temperature: float = 0.01,
) -> pl.DataFrame:
    if isinstance(system_prompt, list):
        if len(system_prompt) != len(all_requests):
            raise ValueError("`system_prompt` list length must match `all_requests` length.")
        sys_prompts = system_prompt
    else:
        sys_prompts = [system_prompt] * len(all_requests)

    tokenizer = _load_tokenizer(model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    hf_model.eval()

    # Left-pad so all sequences in a batch end at the same position before generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        eos = tokenizer.eos_token_id
        tokenizer.pad_token_id = eos[0] if isinstance(eos, list) else eos

    results: list[str] = []
    for i in range(0, len(all_requests), batch_size):
        batch_requests = all_requests[i : i + batch_size]
        batch_prompts = sys_prompts[i : i + batch_size]

        prompts = [_build_prompt(tokenizer, req, sp) for req, sp in zip(batch_requests, batch_prompts)]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(hf_model.device)

        with torch.no_grad():
            output_ids = hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for seq in output_ids:
            answer = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            print(f"Answer length = {len(answer)}")
            results.append(answer)

    return pl.DataFrame({"question": all_requests, "answer_ai": results})