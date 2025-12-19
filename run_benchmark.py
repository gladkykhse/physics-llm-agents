import argparse
import asyncio
import os
from datetime import datetime

import polars as pl

from src.benchmarks import mmlu, scieval
from src.models import langgraph, ollama, openai_api, vllm
from src.utils.helpers import load_yaml

BENCHMARK_CFG = load_yaml("config/benchmark.yaml")


async def run_mmlu(model: str, subset: str) -> None:
    dataset_path = mmlu.get_dataset(subset=subset, save_dir="benchmarks")
    file_path = os.path.join(dataset_path, "test", "data-00000-of-00001.arrow")
    df_mmlu = mmlu.preprocess_questions(mmlu.load_dataframe(source=file_path))

    questions = df_mmlu["question"].to_list()

    if model in BENCHMARK_CFG["agents"]:
        evaluation_df = langgraph.run_solving(all_requests=questions, agent=model)
        df_mmlu = df_mmlu.join(evaluation_df, on="question")
    else:
        for prompt_fn in (mmlu.standard_system_prompt, mmlu.cot_system_prompt):
            if model in BENCHMARK_CFG["openai_models"]:
                evaluation_df = await openai_api.run_batched_completion(
                    all_requests=questions, system_prompt=prompt_fn(), model=model
                )
            elif model in BENCHMARK_CFG["ollama_models"]:
                evaluation_df = await ollama.run_completion(
                    all_requests=questions, system_prompt=prompt_fn(), model=model, batch_size=4
                )
            elif model in BENCHMARK_CFG["vllm_models"]:
                evaluation_df = await vllm.run_completion(
                    all_requests=questions, system_prompt=prompt_fn(), model=model, batch_size=4
                )

            evaluation_df = evaluation_df.rename({"answer": f"answer_{prompt_fn.__name__}"})
            df_mmlu = df_mmlu.join(evaluation_df, on="question")

    output_dir = os.path.join(BENCHMARK_CFG["outputs_dir"], "MMLU_evaluation", subset)
    os.makedirs(output_dir, exist_ok=True)

    df_mmlu.write_parquet(
        file=os.path.join(
            output_dir,
            f"{model.replace('/', '-')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
        )
    )


async def run_scieval(model: str, topics: list[str]) -> None:
    dataset_path = scieval.get_dataset(save_dir="benchmarks")
    file_path = os.path.join(dataset_path, "test", "data-00000-of-00001.arrow")
    df_scieval = scieval.load_dataframe(source=file_path)

    if topics:
        df_scieval = df_scieval.filter(pl.col("topic").is_in(topics))

    questions = df_scieval["question"].to_list()

    if model in BENCHMARK_CFG["agents"]:
        evaluation_df = langgraph.run_solving(all_requests=questions, agent=model)
        df_scieval = df_scieval.join(evaluation_df, on="question")
    else:
        for prompt_fn in (scieval.standard_system_prompt, scieval.cot_system_prompt):
            questions = df_scieval["question"].to_list()

            if model in BENCHMARK_CFG["openai_models"]:
                evaluation_df = await openai_api.run_batched_completion(
                    all_requests=questions, system_prompt=prompt_fn(), model=model
                )
            elif model in BENCHMARK_CFG["ollama_models"]:
                evaluation_df = await ollama.run_completion(
                    all_requests=questions, system_prompt=prompt_fn(), model=model, batch_size=4
                )
            elif model in BENCHMARK_CFG["vllm_models"]:
                evaluation_df = await vllm.run_completion(
                    all_requests=questions, system_prompt=prompt_fn(), model=model, batch_size=4
                )

            evaluation_df = evaluation_df.rename({"answer": f"answer_{prompt_fn.__name__}"})
            df_scieval = df_scieval.join(evaluation_df, on="question")

    output_dir = os.path.join(BENCHMARK_CFG["outputs_dir"], "SciEval_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    df_scieval.write_parquet(
        file=os.path.join(
            output_dir,
            f"{model.replace('/', '-')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
        )
    )


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
        asyncio.run(run_mmlu(model=args.model, subset=args.subset))
    elif args.benchmark == "scieval":
        asyncio.run(
            run_scieval(
                model=args.model,
                topics=args.topics,
            )
        )
