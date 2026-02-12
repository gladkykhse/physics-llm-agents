import os

import datasets
import matplotlib.pyplot as plt
import polars as pl

from rich.console import Console
from rich.table import Table
from rich import box


SUBSETS = [
    "college_physics",
    "conceptual_physics",
    "high_school_physics",
]

NUM_TO_LETTER = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}

LETTER_TO_NUM = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}


def get_dataset(subset: str = "college_physics", save_dir: str = "benchmarks") -> str:
    data_path = os.path.join(save_dir, "MMLU", subset)
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
        datasets.load_dataset(path="cais/mmlu", name=subset).save_to_disk(data_path)
    return data_path


def load_dataframe(source: str = "benchmarks/MMLU/college_physics/test/data-00000-of-00001.arrow") -> pl.DataFrame:
    return pl.read_ipc_stream(source=source)


def cot_system_prompt() -> str:
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


def standard_system_prompt() -> str:
    return """Given a physics question and four options, please select the right answer.
    Your answer should be a single letter A, B, C or D.
    Please directly give the answer without any explanation."""


def preprocess_questions(df: pl.DataFrame) -> pl.DataFrame:
    def combine_question_answers(question: str, choices: list[str]) -> str:
        res = f"{question}\n\n"
        for i, choice in enumerate(choices):
            res += f"{NUM_TO_LETTER[i]}. {choice}\n"
        return res

    return df.with_columns(
        pl.struct(["question", "choices"])
        .map_elements(
            lambda combined: combine_question_answers(combined["question"], combined["choices"]),
            return_dtype=pl.String,
        )
        .alias("question")
    )


def parse_results_to_dict(
    df: pl.DataFrame,
    model_answer_col: str = "answer_ai",
    single_letter_ai_answer: bool = False,
) -> dict[str, object]:
    if single_letter_ai_answer:
        df = df.with_columns(
            pl.col(model_answer_col).str.extract(r"([ABCD])").alias("pred_ans")
        )
    else:
        df = df.with_columns(
            pl.col(model_answer_col)
            .str.extract(r"(?i)answer.*?([ABCD])")
            .str.to_uppercase()
            .alias("pred_ans")
        )

    df = df.with_columns(
        pl.col("pred_ans").is_not_null().alias("answered"),
    )

    df = df.with_columns(
        answer_list=pl.when(pl.col("answer").is_not_null())
        .then(
            pl.col("answer").map_elements(
                lambda x: [NUM_TO_LETTER[x]],
                return_dtype=pl.List(pl.String),
            )
        )
        .otherwise(None)
    )

    df = df.with_columns(
        pl.when(pl.col("answered") & pl.col("answer_list").is_not_null())
        .then(pl.col("answer_list").list.contains(pl.col("pred_ans")).fill_null(False))
        .otherwise(False)
        .alias("correct")
    )

    df = df.with_columns(
        (pl.col("answered") & (~pl.col("correct"))).alias("incorrect"),
        (~pl.col("answered")).alias("not_answered"),
    )

    return {
        "correct": round(float(df["correct"].mean()), 4),
        "incorrect": round(float(df["incorrect"].mean()), 4),
        "not_answered": round(float(df["not_answered"].mean()), 4),
    }


def plot_results(results_json: dict[str, object], filename: str) -> None:
    correct = results_json["correct"]
    not_answered = results_json["not_answered"]
    incorrect = results_json["incorrect"]

    plt.figure(figsize=(6, 4))
    plt.bar(["Model"], [correct], label="Correct")
    plt.bar(["Model"], [not_answered], bottom=[correct], label="Not Answered")
    plt.bar(
        ["Model"],
        [incorrect],
        bottom=[correct + not_answered],
        label="Incorrect",
    )

    plt.title("Performance Breakdown")
    plt.ylabel("Percentage")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def print_results_table(results_json: dict[str, object], title: str = "MMLU Benchmark Results") -> None:
    console = Console()

    overall = Table(title=title, box=box.SIMPLE_HEAVY)
    overall.add_column("Metric", style="bold")
    overall.add_column("Value", justify="right")

    overall.add_row("Correct", f"{results_json['correct'] * 100:.2f}%")
    overall.add_row("Incorrect", f"{results_json['incorrect'] * 100:.2f}%")
    overall.add_row("Not Answered", f"{results_json['not_answered'] * 100:.2f}%")

    console.print(overall)
