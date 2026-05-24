"""HuggingFace local model inference backend.

ChatGLM3-based models (e.g. SciGLM-6B) ship remote code that is incompatible
with transformers >= 4.44. Three known breakages are patched here at runtime
by catching the first failure, fixing the loaded class in sys.modules, and
retrying — model weights are already cached so the retry is cheap.
"""
import sys
from typing import List, Union

import polars as pl
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# ChatGLM3 compatibility patches (transformers >= 4.44 API changes)
# ---------------------------------------------------------------------------

def _patch_chatglm_tokenizer() -> None:
    """get_vocab() is called during __init__ before sp_model is set."""
    for name, mod in sys.modules.items():
        if "tokenization_chatglm" not in name:
            continue
        cls = getattr(mod, "ChatGLMTokenizer", None)
        if cls is None or not callable(getattr(cls, "get_vocab", None)):
            continue
        _orig = cls.get_vocab
        def _safe(self, _orig=_orig):
            return {} if not hasattr(self, "sp_model") else _orig(self)
        cls.get_vocab = _safe
        return


def _patch_chatglm_model() -> None:
    """all_tied_weights_keys renamed from _tied_weights_keys."""
    for name, mod in sys.modules.items():
        if "modeling_chatglm" not in name:
            continue
        for attr in dir(mod):
            cls = getattr(mod, attr, None)
            if isinstance(cls, type) and not hasattr(cls, "all_tied_weights_keys"):
                cls.all_tied_weights_keys = {}
        return


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_chatglm(model: str):
    """Load ChatGLM3-based models via AutoModel + .half().cuda(), exactly as
    the model documentation prescribes, to avoid device_map incompatibilities."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except AttributeError as e:
        if "vocab_size" not in str(e):
            raise
        _patch_chatglm_tokenizer()
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    if not hasattr(config, "max_length") and hasattr(config, "seq_length"):
        config.max_length = config.seq_length  # removed in transformers >= 4.44

    try:
        hf_model = AutoModel.from_pretrained(model, config=config, trust_remote_code=True)
    except AttributeError as e:
        if "all_tied_weights_keys" not in str(e):
            raise
        _patch_chatglm_model()
        hf_model = AutoModel.from_pretrained(model, config=config, trust_remote_code=True)

    return tokenizer, hf_model.half().cuda().eval()


def _load_standard(model: str):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=True, dtype=torch.float16, device_map="auto"
    )
    return tokenizer, hf_model.eval()


# ---------------------------------------------------------------------------
# Prompt builder (for non-ChatGLM models)
# ---------------------------------------------------------------------------

def _build_prompt(tokenizer: AutoTokenizer, request: str, system_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": request})

    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
    parts.append("Assistant:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

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

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    chatglm = "chatglm" in getattr(config, "model_type", "").lower()
    tokenizer, hf_model = _load_chatglm(model) if chatglm else _load_standard(model)

    results: list[str] = []

    if chatglm:
        for request, sp in zip(all_requests, sys_prompts):
            kwargs: dict = {"history": [], "max_length": max_new_tokens, "temperature": temperature}
            if sp:
                kwargs["system"] = sp
            with torch.no_grad():
                response, _ = hf_model.chat(tokenizer, request, **kwargs)
            print(f"Answer length = {len(response)}")
            results.append(response)
    else:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            eos = tokenizer.eos_token_id
            tokenizer.pad_token_id = eos[0] if isinstance(eos, list) else eos

        for i in range(0, len(all_requests), batch_size):
            batch_reqs = all_requests[i : i + batch_size]
            batch_sys = sys_prompts[i : i + batch_size]
            prompts = [_build_prompt(tokenizer, req, sp) for req, sp in zip(batch_reqs, batch_sys)]
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