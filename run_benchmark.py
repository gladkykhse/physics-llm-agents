import argparse
import asyncio
import logging as log
import os
from datetime import datetime
from typing import Callable

import polars as pl

from src.benchmarks import mmlu, scieval
from src.models import langgraph, ollama, openai_api, vllm
from src.utils.helpers import load_yaml

BENCHMARK_CFG = load_yaml("config/benchmark.yaml")
log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


async def run_mmlu(
    model: str,
    subset: str,
    prompt_fn: Callable | None = None,
) -> None:
    dataset_path = mmlu.get_dataset(subset=subset, save_dir="benchmarks")
    file_path = os.path.join(dataset_path, "test", "data-00000-of-00001.arrow")
    df_mmlu = mmlu.preprocess_questions(mmlu.load_dataframe(source=file_path))

    questions = df_mmlu["question"].to_list()

    if model in BENCHMARK_CFG["agents"]:
        evaluation_df = langgraph.run_solving(
            all_requests=questions,
            agent=model,
        )
    else:
        if model in BENCHMARK_CFG["openai_models"]:
            evaluation_df = await openai_api.run_batched_completion(
                all_requests=questions,
                system_prompt=prompt_fn(),
                model=model,
            )
        elif model in BENCHMARK_CFG["ollama_models"]:
            evaluation_df = await ollama.run_completion(
                all_requests=questions,
                system_prompt=prompt_fn(),
                model=model,
                batch_size=4,
            )
        elif model in BENCHMARK_CFG["vllm_models"]:
            evaluation_df = await vllm.run_completion(
                all_requests=questions,
                system_prompt=prompt_fn(),
                model=model,
                batch_size=4,
            )
        else:
            raise ValueError(f"Model '{model}' not found in BENCHMARK_CFG.")

    df_mmlu = df_mmlu.join(evaluation_df, on="question")

    output_dir = os.path.join(
        BENCHMARK_CFG["outputs_dir"],
        "MMLU_evaluation",
        subset,
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{model.replace('/', '-')}_{prompt_fn.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
    )

    df_mmlu.write_parquet(file=output_path)
    log.info(f"Outputs saved to {output_path}")


async def run_scieval(
    model: str,
    topics: list[str] | None = None,
    prompt_fn: Callable | None = None,
) -> None:
    dataset_path = scieval.get_dataset(save_dir="benchmarks")
    file_path = os.path.join(dataset_path, "test", "data-00000-of-00001.arrow")
    df_scieval = scieval.load_dataframe(source=file_path)

    if topics:
        df_scieval = df_scieval.filter(pl.col("topic").is_in(topics))

    questions = df_scieval["question"].to_list()

    if model in BENCHMARK_CFG["agents"]:
        evaluation_df = langgraph.run_solving(
            all_requests=questions,
            agent=model,
        )
    else:
        if model in BENCHMARK_CFG["openai_models"]:
            evaluation_df = await openai_api.run_batched_completion(
                all_requests=questions,
                system_prompt=prompt_fn(),
                model=model,
            )
        elif model in BENCHMARK_CFG["ollama_models"]:
            evaluation_df = await ollama.run_completion(
                all_requests=questions,
                system_prompt=prompt_fn(),
                model=model,
                batch_size=4,
            )
        elif model in BENCHMARK_CFG["vllm_models"]:
            evaluation_df = await vllm.run_completion(
                all_requests=questions,
                system_prompt=prompt_fn(),
                model=model,
                batch_size=4,
            )
        else:
            raise ValueError(f"Model '{model}' not found in BENCHMARK_CFG.")

    df_scieval = df_scieval.join(evaluation_df, on="question")

    output_dir = os.path.join(BENCHMARK_CFG["outputs_dir"], "SciEval_evaluation")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{model.replace('/', '-')}_{prompt_fn.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
    )

    df_scieval.write_parquet(file=output_path)
    log.info(f"Outputs saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="llama3:8b",
        choices=BENCHMARK_CFG["openai_models"]
        + BENCHMARK_CFG["ollama_models"]
        + BENCHMARK_CFG["vllm_models"]
        + BENCHMARK_CFG["agents"],
        help="The name of the benchmark you want to run",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="cot",
        choices=["standard", "cot"],
        help="Type of prompt you wish to use (for benchmarking models only)",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        default="mmlu",
        choices=["mmlu", "scieval"],
        help="The name of the benchmark you want to run",
    )
    parser.add_argument(
        "-s",
        "--subset",
        type=str,
        default="college_physics",
        choices=mmlu.SUBSETS,
        help="Subset of the MMLU benchmark you want to run",
    )
    parser.add_argument(
        "-t",
        "--topics",
        nargs="*",
        type=str,
        default=[],
    )

    args = parser.parse_args()
    if args.benchmark == "mmlu":
        prompt_fn = mmlu.cot_system_prompt if args.prompt == "cot" else mmlu.standard_system_prompt
        asyncio.run(run_mmlu(model=args.model, subset=args.subset, prompt_fn=prompt_fn))
    elif args.benchmark == "scieval":
        prompt_fn = scieval.cot_system_prompt if args.prompt == "cot" else scieval.standard_system_prompt
        asyncio.run(run_scieval(model=args.model, topics=args.topics, prompt_fn=prompt_fn))
