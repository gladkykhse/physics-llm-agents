# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "transformers==4.41.2",
#   "torch>=2.1",
#   "sentencepiece",
#   "accelerate>=0.30.0",
#   "polars>=1.0",
#   "datasets>=2.18",
#   "huggingface_hub>=0.20",
# ]
# ///
"""Standalone HuggingFace benchmark runner for ChatGLM3-based and MoE models.

transformers==4.41.2 is the last version compatible with ChatGLM3 remote code.
It requires Python <3.13 (tokenizers 0.19 has no Python 3.13 wheels).
Run with:
    uv run --python 3.11 run_hf_benchmark.py -m zd21/SciGLM-6B -b scieval
    uv run --python 3.11 run_hf_benchmark.py -m OpenDFM/SciDFM-MoE-A5.6B-v1.0 -b scieval
"""
import argparse
import os
import sys
from datetime import datetime

import datasets as hf_datasets
import polars as pl
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

OUTPUTS_DIR = "artifacts"

# ---------------------------------------------------------------------------
# Benchmark data loading (inline — no src/ deps)
# ---------------------------------------------------------------------------

_NUM_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
MMLU_SUBSETS = ["college_physics", "conceptual_physics", "high_school_physics"]


def _load_scieval(save_dir: str, topics: list[str], abilities: list[str]) -> pl.DataFrame:
    data_path = os.path.join(save_dir, "SciEval")
    if not os.path.isdir(data_path):
        os.makedirs(data_path, exist_ok=True)
        hf_datasets.load_dataset("OpenDFM/SciEval").save_to_disk(data_path)

    arrow = os.path.join(data_path, "test", "data-00000-of-00001.arrow")
    df = pl.read_ipc_stream(source=arrow).filter(pl.col("category") == "physics")

    agg_cols = [c for c in df.columns if c != "question"]
    df = df.group_by("question", maintain_order=True).agg(
        [pl.col(c).drop_nulls().first().alias(c) for c in agg_cols]
    )
    df = df.with_columns(
        pl.col("question").str.strip_chars().str.strip_suffix("Answer:").str.strip_chars()
    )
    if topics:
        df = df.filter(pl.col("topic").is_in(topics))
    if abilities:
        df = df.filter(pl.col("ability").is_in(abilities))
    return df


def _load_mmlu(save_dir: str, subset: str) -> pl.DataFrame:
    data_path = os.path.join(save_dir, "MMLU", subset)
    if not os.path.isdir(data_path):
        os.makedirs(data_path, exist_ok=True)
        hf_datasets.load_dataset("cais/mmlu", subset).save_to_disk(data_path)

    arrow = os.path.join(data_path, "test", "data-00000-of-00001.arrow")
    df = pl.read_ipc_stream(source=arrow)
    return df.with_columns(
        pl.struct(["question", "choices"]).map_elements(
            lambda r: r["question"] + "\n\n" + "".join(
                f"{_NUM_TO_LETTER[i]}. {c}\n" for i, c in enumerate(r["choices"])
            ),
            return_dtype=pl.String,
        ).alias("question")
    )


def _cot_system_prompt() -> str:
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


def _standard_system_prompt() -> str:
    return """Given a physics question and four options, please select the right answer.
    Your answer should be a single letter A, B, C or D.
    Please directly give the answer without any explanation."""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _patch_chatglm_get_vocab() -> None:
    """Patch ChatGLMTokenizer.get_vocab() called before SentencePiece is assigned."""
    for name, mod in list(sys.modules.items()):
        if "tokenization_chatglm" not in name:
            continue
        cls = getattr(mod, "ChatGLMTokenizer", None)
        if cls is None:
            continue
        def _get_vocab(self):
            sp = getattr(self, "tokenizer", None) or getattr(self, "sp_model", None)
            if sp is None:
                return {}
            n = getattr(sp, "n_words", None) or getattr(sp, "get_piece_size", lambda: 0)()
            return {self._convert_id_to_token(i): i for i in range(n)}
        cls.get_vocab = _get_vocab
        return


def _is_chatglm(model_id: str) -> bool:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return "chatglm" in getattr(config, "model_type", "").lower()


def _load_chatglm(model_id: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except AttributeError:
        _patch_chatglm_get_vocab()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    return tokenizer, model


def _load_standard(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_completion(
    questions: list[str],
    system_prompt: str,
    model_id: str,
    batch_size: int = 4,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
) -> pl.DataFrame:
    chatglm = _is_chatglm(model_id)
    tokenizer, model = _load_chatglm(model_id) if chatglm else _load_standard(model_id)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        eos = tokenizer.eos_token_id
        tokenizer.pad_token_id = eos[0] if isinstance(eos, list) else eos

    results: list[str] = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        prompts = []
        for q in batch:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
            if getattr(tokenizer, "chat_template", None) is not None:
                prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                ))
            else:
                prompts.append(
                    f"System: {system_prompt}\nUser: {q}\nAssistant:"
                )

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for seq in output_ids:
            answer = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            print(f"[{i + len(results) + 1}/{len(questions)}] Answer length = {len(answer)}")
            results.append(answer)

    return pl.DataFrame({"question": questions, "answer_ai": results})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark HuggingFace models (ChatGLM3/MoE) on physics datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--model", required=True, help="HuggingFace model ID")
    parser.add_argument("-b", "--benchmark", default="scieval", choices=["scieval", "mmlu"])
    parser.add_argument("-p", "--prompt", default="cot", choices=["cot", "standard"])
    parser.add_argument("-s", "--subset", default="college_physics", choices=MMLU_SUBSETS,
                        help="MMLU subset (ignored for scieval)")
    parser.add_argument("-t", "--topics", nargs="*", default=[], help="SciEval topic filter")
    parser.add_argument("-a", "--abilities", nargs="*", default=[], help="SciEval ability filter")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--benchmarks-dir", default="benchmarks")
    args = parser.parse_args()

    prompt_fn = _cot_system_prompt if args.prompt == "cot" else _standard_system_prompt

    if args.benchmark == "scieval":
        df = _load_scieval(args.benchmarks_dir, args.topics, args.abilities)
        output_dir = os.path.join(OUTPUTS_DIR, "SciEval_evaluation")
    else:
        df = _load_mmlu(args.benchmarks_dir, args.subset)
        output_dir = os.path.join(OUTPUTS_DIR, "MMLU_evaluation", args.subset)

    questions = df["question"].to_list()
    print(f"Loaded {len(questions)} questions from {args.benchmark}")

    results_df = run_completion(
        questions=questions,
        system_prompt=prompt_fn(),
        model_id=args.model,
        batch_size=args.batch_size,
    )

    df = df.join(results_df, on="question")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{args.model.replace('/', '-')}_{prompt_fn.__name__[1:]}_{timestamp}.parquet")
    df.write_parquet(output_path)
    print(f"Results saved to {output_path}")
    print(f"Evaluate with: python run_evaluation.py --benchmark {args.benchmark} --filename {output_path}")