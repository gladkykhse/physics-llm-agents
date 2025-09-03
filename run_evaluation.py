import os
import argparse
import polars as pl

from src.utils import scieval, mmlu
from src.utils.helpers import save_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        default="scieval",
        choices=["mmlu", "scieval"],
        help="The name of the benchmark you want to run evaluation on",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        help="Filename of the benchmark output (parquet file)",
    )

    args = parser.parse_args()
    df = pl.read_parquet(source=args.filename)
    if args.benchmark == "scieval":
        results = scieval.parse_results_to_dict(df=df)
        no_extension = os.path.splitext(args.filename)[0]
        save_json(obj=results, filename=f"{no_extension}.json")
        scieval.plot_results(results_json=results, filename=f"{no_extension}.png")
    elif args.benchmark == "mmlu":
        results = mmlu.parse_results_to_dict(df=df)
        no_extension = os.path.splitext(args.filename)[0]
        save_json(obj=results, filename=f"{no_extension}.json")
        mmlu.plot_results(results_json=results, filename=f"{no_extension}.png")
