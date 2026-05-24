"""HuggingFace local model inference backend.

SciGLM-6B ships ChatGLM3 remote code written for transformers~=4.30. Since
tokenizers<0.21 has no Python 3.13 wheels, we must run on transformers>=4.45
and patch the three known breakages at runtime after the remote code is loaded.
All models use standard generate(); ChatGLM's .chat() is not used because it
passes unknown kwargs into generate() and fails validation in newer transformers.
"""
import sys
from typing import List, Union

import polars as pl
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationMixin


def _patch_chatglm() -> None:
    """Apply all ChatGLM3 compatibility patches for transformers >= 4.42.

    Must be called after AutoTokenizer.from_pretrained so the remote-code
    modules are present in sys.modules.
    """
    for name, mod in list(sys.modules.items()):
        if "tokenization_chatglm" in name:
            cls = getattr(mod, "ChatGLMTokenizer", None)
            if cls is None:
                continue

            # (1) get_vocab() is called inside super().__init__() before
            #     sp_model is set. transformers 5.x also broke vocab_size as a
            #     property in this MRO, so bypass it and use sp_model directly.
            def _get_vocab(self):
                if not hasattr(self, "sp_model"):
                    return {}
                vocab = {self._convert_id_to_token(i): i
                         for i in range(self.sp_model.get_piece_size())}
                vocab.update(getattr(self, "added_tokens_encoder", {}))
                return vocab
            cls.get_vocab = _get_vocab

            # (2) transformers >= 4.44 passes padding_side to _pad(); ChatGLM's
            #     _pad() signature doesn't accept it.
            _orig = cls._pad
            def _pad(self, *args, padding_side=None, _o=_orig, **kwargs):
                return _o(self, *args, **kwargs)
            cls._pad = _pad

        elif "modeling_chatglm" in name:
            for attr in dir(mod):
                obj = getattr(mod, attr, None)
                if not isinstance(obj, type):
                    continue
                # (3) all_tied_weights_keys was renamed from _tied_weights_keys in
                #     transformers >= 4.44. Must be a dict (not list) as .keys() is called.
                if not hasattr(obj, "all_tied_weights_keys"):
                    obj.all_tied_weights_keys = {}
                # (4) transformers >= 5.0: PreTrainedModel no longer inherits GenerationMixin.
                #     ChatGLM's remote code never directly subclassed it, so generate() is gone.
                if hasattr(obj, "prepare_inputs_for_generation") and not issubclass(obj, GenerationMixin):
                    obj.__bases__ = obj.__bases__ + (GenerationMixin,)


def _load_chatglm(model_id: str):
    # Load tokenizer — this imports the remote code into sys.modules.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except (AttributeError, TypeError):
        _patch_chatglm()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Always patch after tokenizer load to cover _pad (even if __init__ succeeded).
    _patch_chatglm()

    # config.max_length was removed in transformers >= 4.44 but ChatGLM's
    # __init__ reads it. Restore from seq_length, then strip after model loads
    # so generate() doesn't complain about it.
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if not hasattr(config, "max_length") and hasattr(config, "seq_length"):
        config.max_length = config.seq_length

    try:
        model = AutoModel.from_pretrained(model_id, config=config, trust_remote_code=True)
    except AttributeError:
        _patch_chatglm()
        model = AutoModel.from_pretrained(model_id, config=config, trust_remote_code=True)

    model.config.__dict__.pop("max_length", None)
    return tokenizer, model.half().cuda().eval()


def _load_standard(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, model.eval()


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
            raise ValueError("system_prompt list length must match all_requests length")
        sys_prompts = system_prompt
    else:
        sys_prompts = [system_prompt] * len(all_requests)

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    is_chatglm = "chatglm" in getattr(config, "model_type", "").lower()
    tokenizer, hf_model = _load_chatglm(model) if is_chatglm else _load_standard(model)

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
            messages = ([{"role": "system", "content": sp}] if sp else [])
            messages.append({"role": "user", "content": req})
            if getattr(tokenizer, "chat_template", None) is not None:
                prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
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