import sys
from typing import Union, List

import polars as pl
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer


def _patch_chatglm_tokenizer() -> None:
    """ChatGLM3 tokenizer calls get_vocab() before sp_model is initialised
    (Python 3.13 + transformers >=4.44 incompatibility). The class is already
    in sys.modules after the failed import — patch it there and retry."""
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
            return


def _load_chatglm(model: str):
    """Load a ChatGLM3-based model the way its own docs prescribe:
    AutoModel + .half().cuda() — avoids every transformers >=4.44 breakage."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except AttributeError as e:
        if "vocab_size" not in str(e):
            raise
        _patch_chatglm_tokenizer()
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    # config.max_length was removed in transformers >=4.44; ChatGLM still uses it
    if not hasattr(config, "max_length") and hasattr(config, "seq_length"):
        config.max_length = config.seq_length
    hf_model = AutoModel.from_pretrained(model, config=config, trust_remote_code=True).half().cuda()
    return tokenizer, hf_model.eval()


def _load_standard(model: str):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=True, dtype=torch.float16, device_map="auto"
    )
    return tokenizer, hf_model.eval()


def _build_prompt(tokenizer: AutoTokenizer, request: str, system_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": request})

    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback for models without a chat template
    parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
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

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    chatglm = "chatglm" in getattr(config, "model_type", "").lower()
    tokenizer, hf_model = _load_chatglm(model) if chatglm else _load_standard(model)

    results: list[str] = []

    if chatglm:
        # ChatGLM's .chat() handles tokenisation and generation internally
        for request, sp in zip(all_requests, sys_prompts):
            kwargs: dict = {"history": [], "max_length": max_new_tokens,
                            "temperature": temperature}
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