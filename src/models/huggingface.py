"""HuggingFace local model inference backend.

ChatGLM3-based models (e.g. SciGLM-6B) ship remote code that is incompatible
with transformers >= 4.44. Known breakages are patched at runtime by catching
the first failure, fixing the loaded class in sys.modules, and retrying.
All models use the same tokenize → generate() → decode path; ChatGLM's
.chat() helper is intentionally avoided as it leaks kwargs into generate().
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
    for name, mod in sys.modules.items():
        if "tokenization_chatglm" not in name:
            continue
        cls = getattr(mod, "ChatGLMTokenizer", None)
        if cls is None:
            continue
        # get_vocab() called during __init__ before sp_model is set;
        # also self.vocab_size is a broken property in transformers 5.x MRO,
        # so bypass it entirely and read sp_model directly.
        if callable(getattr(cls, "get_vocab", None)):
            def _safe_get_vocab(self):
                if not hasattr(self, "sp_model"):
                    return {}
                n = self.sp_model.get_piece_size()
                vocab = {self._convert_id_to_token(i): i for i in range(n)}
                vocab.update(getattr(self, "added_tokens_encoder", {}))
                return vocab
            cls.get_vocab = _safe_get_vocab
        # _pad() doesn't accept padding_side kwarg added in newer transformers
        if callable(getattr(cls, "_pad", None)):
            _orig = cls._pad
            def _safe_pad(self, *args, padding_side=None, _o=_orig, **kwargs):
                return _o(self, *args, **kwargs)
            cls._pad = _safe_pad
        return


def _patch_chatglm_model() -> None:
    # all_tied_weights_keys renamed from _tied_weights_keys
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
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except AttributeError as e:
        if "vocab_size" not in str(e):
            raise
        _patch_chatglm_tokenizer()
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    _patch_chatglm_tokenizer()  # always apply _pad patch after load

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    # config.max_length removed in transformers >= 4.44; ChatGLM __init__ still reads it
    if not hasattr(config, "max_length") and hasattr(config, "seq_length"):
        config.max_length = config.seq_length

    try:
        hf_model = AutoModel.from_pretrained(model, config=config, trust_remote_code=True)
    except AttributeError as e:
        if "all_tied_weights_keys" not in str(e):
            raise
        _patch_chatglm_model()
        hf_model = AutoModel.from_pretrained(model, config=config, trust_remote_code=True)

    # Strip max_length from model.config so generate() doesn't raise about it
    hf_model.config.__dict__.pop("max_length", None)
    return tokenizer, hf_model.half().cuda().eval()


def _load_standard(model: str):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=True, dtype=torch.float16, device_map="auto"
    )
    return tokenizer, hf_model.eval()


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

    # Unified generate() path for all models — avoids ChatGLM's .chat() which
    # leaks extra kwargs (e.g. system=) into generate() and fails validation.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        eos = tokenizer.eos_token_id
        tokenizer.pad_token_id = eos[0] if isinstance(eos, list) else eos

    results: list[str] = []
    for i in range(0, len(all_requests), batch_size):
        batch_reqs = all_requests[i : i + batch_size]
        batch_sys = sys_prompts[i : i + batch_size]

        prompts = []
        for req, sp in zip(batch_reqs, batch_sys):
            messages = []
            if sp:
                messages.append({"role": "system", "content": sp})
            messages.append({"role": "user", "content": req})
            if getattr(tokenizer, "chat_template", None) is not None:
                prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ))
            else:
                parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
                parts.append("Assistant:")
                prompts.append("\n".join(parts))

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