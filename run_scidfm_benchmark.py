# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "transformers==4.39.3",
#   "torch>=2.3.0",
#   "sentencepiece>=0.2.0",
#   "accelerate>=0.28.0",
#   "safetensors>=0.4.2",
#   "tokenizers>=0.15.0,<0.19",
#   "polars>=1.0",
#   "datasets>=2.18.0",
#   "huggingface_hub>=0.21.0",
#   "flash-attn>=2.5.0",
#   "packaging",
# ]
#
# [tool.uv.extra-build-dependencies]
# flash-attn = ["torch", "packaging", "setuptools", "wheel"]
# ///
"""Benchmark runner for OpenDFM/SciDFM-MoE-A5.6B-v1.0 on 3x A100-80GB GPUs (0-2).

Model: 19B total / 5.6B active params, BF16, ~38 GB.
Loaded once with device_map="auto" spread across GPUs 0-2 via CUDA_VISIBLE_DEVICES.
No multiprocessing — trust_remote_code + spawn is incompatible.

Run with:
    uv run --python 3.11 run_scidfm_benchmark.py -b scieval
    uv run --python 3.11 run_scidfm_benchmark.py -b mmlu -s college_physics
    uv run --python 3.11 run_scidfm_benchmark.py -b scieval -p standard
"""
import argparse
import os
from datetime import datetime

# Restrict to GPUs 0-2 before any torch/CUDA import
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2")

import datasets as hf_datasets
import polars as pl
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

MODEL_ID = "OpenDFM/SciDFM-MoE-A5.6B-v1.0"
OUTPUTS_DIR = "artifacts"
CHAT_TEMPLATE = "<|user|>:{instruction}<|assistant|>:"

# ---------------------------------------------------------------------------
# Benchmark data loading
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


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

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
# Inference
# ---------------------------------------------------------------------------

def run_completion(
    questions: list[str],
    system_prompt: str,
    tokenizer: LlamaTokenizer,
    model: AutoModelForCausalLM,
    batch_size: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
) -> pl.DataFrame:
    results: list[str] = []

    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        prompts = [CHAT_TEMPLATE.format(instruction=f"{system_prompt}\n{q}") for q in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_k=20 if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for seq in output_ids:
            answer = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            results.append(answer)
            print(f"[{len(results)}/{len(questions)}] len={len(answer)}", flush=True)

    return pl.DataFrame({"question": questions, "answer_ai": results})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Benchmark {MODEL_ID} on physics datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--benchmark", default="scieval", choices=["scieval", "mmlu"])
    parser.add_argument("-p", "--prompt", default="cot", choices=["cot", "standard"])
    parser.add_argument("-s", "--subset", default="college_physics", choices=MMLU_SUBSETS,
                        help="MMLU subset (ignored for scieval)")
    parser.add_argument("-t", "--topics", nargs="*", default=[], help="SciEval topic filter")
    parser.add_argument("-a", "--abilities", nargs="*", default=[], help="SciEval ability filter")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy decoding (recommended for MCQ)")
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
    print(f"Loaded {len(questions)} questions from {args.benchmark}", flush=True)

    n_gpus = torch.cuda.device_count()
    print(f"Loading {MODEL_ID} across {n_gpus} GPU(s) via device_map=auto ...", flush=True)

    tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        eos = tokenizer.eos_token_id
        tokenizer.pad_token_id = eos[0] if isinstance(eos, list) else eos

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    print(f"Model loaded. Running inference (batch_size={args.batch_size}) ...", flush=True)

    results_df = run_completion(
        questions=questions,
        system_prompt=prompt_fn(),
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    df = df.join(results_df, on="question")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = MODEL_ID.replace("/", "-")
    output_path = os.path.join(output_dir, f"{model_slug}_{prompt_fn.__name__[1:]}_{timestamp}.parquet")
    df.write_parquet(output_path)
    print(f"Results saved to {output_path}")
    print(f"Evaluate with: python run_evaluation.py --benchmark {args.benchmark} --filename {output_path}")