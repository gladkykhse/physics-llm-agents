import argparse
import asyncio
import logging as log
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from typing import Callable

import polars as pl

from src.benchmarks import mmlu, scieval
from src.models import langgraph, ollama, openai_api, vllm
from src.utils.helpers import load_yaml, save_json

BENCHMARK_CFG = load_yaml("config/benchmark.yaml")
log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@contextmanager
def log_to_temp_file():
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".log", dir=os.getcwd())
    os.close(tmp_fd)
    handler = log.FileHandler(tmp_path)
    handler.setFormatter(log.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.getLogger().addHandler(handler)
    try:
        yield tmp_path
    finally:
        log.getLogger().removeHandler(handler)
        handler.close()


async def run_mmlu(
    model: str,
    subset: str,
    prompt_fn: Callable | None = None,
) -> str:
    dataset_path = mmlu.get_dataset(subset=subset, save_dir="benchmarks")
    file_path = os.path.join(dataset_path, "test", "data-00000-of-00001.arrow")
    df_mmlu = mmlu.preprocess_questions(mmlu.load_dataframe(source=file_path))

    questions = df_mmlu["question"].to_list()

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
        evaluation_df = langgraph.run_solving(
            all_requests=questions,
            agent=model,
        )

    df_mmlu = df_mmlu.join(evaluation_df, on="question")

    output_dir = os.path.join(
        BENCHMARK_CFG["outputs_dir"],
        "MMLU_evaluation",
        subset,
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{model.replace('/', '-')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
    )

    df_mmlu.write_parquet(file=output_path)
    log.info(f"Outputs saved to {output_path}")
    return output_path


async def run_scieval(
    model: str,
    topics: list[str] | None = None,
    abilities: list[str] | None = None,
    prompt_fn: Callable | None = None,
) -> str:
    dataset_path = scieval.get_dataset(save_dir="benchmarks")
    file_path = os.path.join(dataset_path, "test", "data-00000-of-00001.arrow")
    df_scieval = scieval.load_dataframe(source=file_path)

    if topics:
        df_scieval = df_scieval.filter(pl.col("topic").is_in(topics))
    if abilities:
        df_scieval = df_scieval.filter(pl.col("ability").is_in(abilities))

    questions = df_scieval["question"].to_list()

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
        evaluation_df = langgraph.run_solving(
            all_requests=questions,
            agent=model,
        )

    df_scieval = df_scieval.join(evaluation_df, on="question")

    output_dir = os.path.join(BENCHMARK_CFG["outputs_dir"], "SciEval_evaluation")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{model.replace('/', '-')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
    )

    df_scieval.write_parquet(file=output_path)
    log.info(f"Outputs saved to {output_path}")
    return output_path


def evaluate(benchmark: str, filename: str, prompt_fn: Callable | None = None) -> None:
    df = pl.read_parquet(source=filename)
    no_extension = os.path.splitext(filename)[0]
    single_letter = prompt_fn is not None and prompt_fn.__name__ == "standard_system_prompt"

    if benchmark == "scieval":
        results = scieval.parse_results_to_dict(
            df=df, model_answer_col="answer_ai", single_letter_ai_answer=single_letter
        )
        save_json(obj=results, filename=f"{no_extension}.json")
        scieval.plot_results(results_json=results, filename=f"{no_extension}.png")
        scieval.print_results_table(results_json=results)
    elif benchmark == "mmlu":
        results = mmlu.parse_results_to_dict(
            df=df, model_answer_col="answer_ai", single_letter_ai_answer=single_letter
        )
        save_json(obj=results, filename=f"{no_extension}.json")
        mmlu.plot_results(results_json=results, filename=f"{no_extension}.png")
        mmlu.print_results_table(results_json=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark and evaluate results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="llama3:8b",
        help="Model name or agent module name (e.g. thinking_react_agent_math_only_v2)",
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
    parser.add_argument(
        "-a",
        "--abilities",
        nargs="*",
        type=str,
        default=[],
        help="Filter SciEval questions by ability (e.g. knowledge calculation)",
    )

    args = parser.parse_args()
    if args.benchmark == "mmlu":
        prompt_fn = mmlu.cot_system_prompt if args.prompt == "cot" else mmlu.standard_system_prompt
        with log_to_temp_file() as tmp_log:
            output_path = asyncio.run(run_mmlu(model=args.model, subset=args.subset, prompt_fn=prompt_fn))
            evaluate(benchmark=args.benchmark, filename=output_path, prompt_fn=prompt_fn)
    elif args.benchmark == "scieval":
        prompt_fn = scieval.cot_system_prompt if args.prompt == "cot" else scieval.standard_system_prompt
        with log_to_temp_file() as tmp_log:
            output_path = asyncio.run(run_scieval(model=args.model, topics=args.topics, abilities=args.abilities, prompt_fn=prompt_fn))
            evaluate(benchmark=args.benchmark, filename=output_path, prompt_fn=prompt_fn)

    log_path = os.path.splitext(output_path)[0] + ".log"
    shutil.move(tmp_log, log_path)
    log.info(f"Logs saved to {log_path}")