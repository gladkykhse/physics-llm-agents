import argparse
import asyncio
import os
from datetime import datetime

import polars as pl

from src.utils import mmlu, scieval
from src.utils.openai_api import run_batched_completion


async def run_mmlu(subset: str = "MMLU_college_physics") -> None:
    path = os.path.join("benchmarks", subset, "test", "data-00000-of-00001.arrow")
    df_mmlu = mmlu.preprocess_questions(mmlu.load_dataframe(source=path))

    for prompt_fn in (mmlu.standard_system_prompt, mmlu.cot_system_prompt):
        questions = df_mmlu["question"].to_list()
        responses = await run_batched_completion(all_requests=questions, system_prompt=prompt_fn())

        evaluation_df = pl.DataFrame(
            {
                "question": [response["initial_request"] for response in responses],
                f"answer_{prompt_fn.__name__}": [
                    response["choices"][0]["message"]["content"] for response in responses
                ],
            }
        )

        df_mmlu = df_mmlu.join(evaluation_df, on="question")

    os.makedirs("artifacts", exist_ok=True)
    df_mmlu.write_parquet(
        file=f"artifacts/{subset}_evaluation/{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    )


async def run_scieval() -> None:
    df_scieval = scieval.load_dataframe()

    for prompt_fn in (scieval.standard_system_prompt, scieval.cot_system_prompt):
        questions = df_scieval["question"].to_list()
        responses = await run_batched_completion(all_requests=questions, system_prompt=prompt_fn())

        evaluation_df = pl.DataFrame(
            {
                "question": [response["initial_request"] for response in responses],
                f"answer_{prompt_fn.__name__}": [
                    response["choices"][0]["message"]["content"] for response in responses
                ],
            }
        )

        df_scieval = df_scieval.join(evaluation_df, on="question")

    os.makedirs("artifacts", exist_ok=True)
    df_scieval.write_parquet(file=f"artifacts/SciEval_evaluation/{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default="MMLU_college_physics",
        choices=mmlu.SUBSETS,
        help="Subset of the MMLU benchmark you want to run",
    )

    args = parser.parse_args()
    if args.benchmark == "mmlu":
        asyncio.run(run_mmlu(args.subset))
    elif args.benchmark == "scieval":
        asyncio.run(run_scieval())
