import os

import datasets
import matplotlib.pyplot as plt
import polars as pl

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


def parse_results_to_dict(df: pl.DataFrame) -> dict[str, object]:
    df = df.with_columns(
        std_ans=pl.col("answer_standard_system_prompt").str.extract(r"([ABCD])"),
        cot_ans=pl.col("answer_cot_system_prompt").str.extract(r"(?i)answer.*?([ABCD])").str.to_uppercase(),
        answer_list=pl.when(pl.col("answer").is_not_null())
        .then(pl.col("answer").map_elements(lambda x: [NUM_TO_LETTER[x]], return_dtype=pl.List(pl.String)))
        .otherwise(None),
    )

    df = df.with_columns(
        answered_std=pl.col("std_ans").is_not_null(),
        answered_cot=pl.col("cot_ans").is_not_null(),
    )

    df = df.with_columns(
        correct_std=pl.when(pl.col("answered_std") & pl.col("answer_list").is_not_null())
        .then(pl.col("answer_list").list.contains(pl.col("std_ans")).fill_null(False))
        .otherwise(False),
        correct_cot=pl.when(pl.col("answered_cot") & pl.col("answer_list").is_not_null())
        .then(pl.col("answer_list").list.contains(pl.col("cot_ans")).fill_null(False))
        .otherwise(False),
    ).with_columns(
        # Incorrect = answered but wrong; Not answered = no parsed letter
        incorrect_std=pl.col("answered_std") & (~pl.col("correct_std")),
        incorrect_cot=pl.col("answered_cot") & (~pl.col("correct_cot")),
        not_answered_std=~pl.col("answered_std"),
        not_answered_cot=~pl.col("answered_cot"),
    )

    results_json = {
        "standard": {
            "correct": round(float(df["correct_std"].mean()), 4),
            "incorrect": round(float(df["incorrect_std"].mean()), 4),
            "not_answered": round(float(df["not_answered_std"].mean()), 4),
        },
        "cot": {
            "correct": round(float(df["correct_cot"].mean()), 4),
            "incorrect": round(float(df["incorrect_cot"].mean()), 4),
            "not_answered": round(float(df["not_answered_cot"].mean()), 4),
        },
    }

    return results_json


def plot_results(results_json: dict[str, object], filename: str) -> None:
    # Extract values
    standard = results_json["standard"]
    cot = results_json["cot"]

    labels = ["Standard", "CoT"]
    correct = [standard["correct"], cot["correct"]]
    not_answered = [standard["not_answered"], cot["not_answered"]]
    incorrect = [standard["incorrect"], cot["incorrect"]]

    # Plot stacked bars
    plt.figure(figsize=(6, 4))
    plt.bar(labels, correct, color="green", label="Correct")
    plt.bar(labels, not_answered, bottom=correct, color="gold", label="Not Answered")
    plt.bar(labels, incorrect, bottom=[c + n for c, n in zip(correct, not_answered)], color="red", label="Incorrect")

    plt.title("Performance Breakdown")
    plt.ylabel("Percentage")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
