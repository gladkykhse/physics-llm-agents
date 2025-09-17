import argparse
import asyncio
import os
from datetime import datetime

from src.benchmarks import mmlu, scieval
from src.models import ollama, openai_api, vllm
from src.utils.helpers import load_yaml

BENCHMARK_CFG = load_yaml("config/benchmark.yaml")


async def run_mmlu(model: str, subset: str) -> None:
    dataset_path = mmlu.get_dataset(subset=subset, save_dir="benchmarks")
    file_path = os.path.join(dataset_path, "test", "data-00000-of-00001.arrow")
    df_mmlu = mmlu.preprocess_questions(mmlu.load_dataframe(source=file_path))

    for prompt_fn in (mmlu.standard_system_prompt, mmlu.cot_system_prompt):
        questions = df_mmlu["question"].to_list()

        if model in BENCHMARK_CFG["openai_models"]:
            evaluation_df = await openai_api.run_batched_completion(
                all_requests=questions, system_prompt=prompt_fn(), model=model
            )
        elif model in BENCHMARK_CFG["ollama_models"]:
            evaluation_df = ollama.run_completion(all_requests=questions, system_prompt=prompt_fn(), model=model)
        elif model in BENCHMARK_CFG["vllm_models"]:
            evaluation_df = vllm.run_completion(all_requests=questions, system_prompt=prompt_fn(), model=model)

        evaluation_df = evaluation_df.rename({"answer": f"answer_{prompt_fn.__name__}"})
        df_mmlu = df_mmlu.join(evaluation_df, on="question")

    os.makedirs(f"artifacts/MMLU_evaluation/{subset}", exist_ok=True)
    df_mmlu.write_parquet(
        file=f"artifacts/MMLU_evaluation/{subset}/{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    )


async def run_scieval(model: str) -> None:
    dataset_path = scieval.get_dataset(save_dir="benchmarks")
    file_path = os.path.join(dataset_path, "test", "data-00000-of-00001.arrow")
    df_scieval = scieval.load_dataframe(source=file_path)

    for prompt_fn in (scieval.standard_system_prompt, scieval.cot_system_prompt):
        questions = df_scieval["question"].to_list()

        if model in BENCHMARK_CFG["openai_models"]:
            evaluation_df = await openai_api.run_batched_completion(
                all_requests=questions, system_prompt=prompt_fn(), model=model
            )
        elif model in BENCHMARK_CFG["ollama_models"]:
            evaluation_df = ollama.run_completion(all_requests=questions, system_prompt=prompt_fn(), model=model)
        elif model in BENCHMARK_CFG["vllm_models"]:
            evaluation_df = vllm.run_completion(all_requests=questions, system_prompt=prompt_fn(), model=model)

        evaluation_df = evaluation_df.rename({"answer": f"answer_{prompt_fn.__name__}"})
        df_scieval = df_scieval.join(evaluation_df, on="question")

    os.makedirs("artifacts/SciEval_evaluation", exist_ok=True)
    df_scieval.write_parquet(
        file=f"artifacts/SciEval_evaluation/{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
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
        choices=BENCHMARK_CFG["openai_models"] + BENCHMARK_CFG["ollama_models"] + BENCHMARK_CFG["vllm_models"],
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

    args = parser.parse_args()
    if args.benchmark == "mmlu":
        asyncio.run(run_mmlu(model=args.model, subset=args.subset))
    elif args.benchmark == "scieval":
        asyncio.run(
            run_scieval(
                model=args.model,
            )
        )
