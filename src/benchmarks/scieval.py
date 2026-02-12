import os

import datasets
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from rich.console import Console
from rich.table import Table
from rich import box


def get_dataset(save_dir: str = "benchmarks") -> str:
    data_path = os.path.join(save_dir, "SciEval")
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
        datasets.load_dataset("OpenDFM/SciEval").save_to_disk(data_path)
    return data_path


def load_dataframe(source: str = "benchmarks/SciEval/test/data-00000-of-00001.arrow") -> pl.DataFrame:
    df_scieval = pl.read_ipc_stream(source=source).filter(pl.col("category") == "physics")

    agg_cols = [c for c in df_scieval.columns if c != "question"]

    df_scieval = df_scieval.group_by("question", maintain_order=True).agg(
        [pl.col(c).drop_nulls().first().alias(c) for c in agg_cols]
    )

    df_scieval = df_scieval.with_columns(
        pl.col("question").str.strip_chars().str.strip_suffix("Answer:").str.strip_chars()
    )

    return df_scieval


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


def parse_results_to_dict(
    df: pl.DataFrame,
    model_answer_col: str = "answer_ai",
    single_letter_ai_answer: bool = False,
) -> dict[str, object]:
    if single_letter_ai_answer:
        extract_expr = (
            pl.col(model_answer_col)
            .str.extract(r"([ABCD])")
            .alias("pred_ans")
        )
    else:
        extract_expr = (
            pl.col(model_answer_col)
            .str.extract(r"(?i)answer.*?([ABCD])")
            .str.to_uppercase()
            .alias("pred_ans")
        )

    df = df.with_columns(extract_expr)

    df = df.with_columns(
        pl.col("pred_ans").is_not_null().alias("answered")
    )

    df = df.with_columns(
        pl.when(pl.col("answered") & pl.col("answer").is_not_null())
        .then(pl.col("answer").list.contains(pl.col("pred_ans")).fill_null(False))
        .otherwise(False)
        .alias("correct")
    )

    df = df.with_columns(
        (pl.col("answered") & (~pl.col("correct"))).alias("incorrect"),
        (~pl.col("answered")).alias("not_answered"),
    )

    pct_correct = float(df["correct"].mean())
    pct_incorrect = float(df["incorrect"].mean())
    pct_not_answered = float(df["not_answered"].mean())

    ability_accuracy = (
        df.group_by("ability")
        .agg(pl.col("correct").mean().alias("acc"))
        .sort("ability")
    )

    topic_accuracy = (
        df.group_by("topic")
        .agg(pl.col("correct").mean().alias("acc"))
        .sort("topic")
    )

    abilities = {
        row["ability"]: round(float(row["acc"]), 4)
        for row in ability_accuracy.to_dicts()
    }

    topics = {
        row["topic"]: round(float(row["acc"]), 4)
        for row in topic_accuracy.to_dicts()
    }

    return {
        "correct": round(pct_correct, 4),
        "incorrect": round(pct_incorrect, 4),
        "not_answered": round(pct_not_answered, 4),
        "abilities": abilities,
        "topics": topics,
    }


def plot_results(results_json: dict[str, object], filename: str) -> None:
    correct = results_json["correct"]
    not_answered = results_json["not_answered"]
    incorrect = results_json["incorrect"]

    abilities = sorted(results_json["abilities"].keys())
    topics = sorted(results_json["topics"].keys())

    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(
        ["Model"],
        [correct],
        label="Correct"
    )
    ax1.bar(
        ["Model"],
        [not_answered],
        bottom=[correct],
        label="Not Answered"
    )
    ax1.bar(
        ["Model"],
        [incorrect],
        bottom=[correct + not_answered],
        label="Incorrect"
    )
    ax1.set_ylim(0, 1)
    ax1.set_title("Overall Breakdown")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(abilities))
    values = [results_json["abilities"][a] for a in abilities]
    ax2.bar(x, values)
    ax2.set_xticks(x)
    ax2.set_xticklabels(abilities)
    ax2.set_ylim(0, 1)
    ax2.set_title("Ability Accuracy")

    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(topics))
    values = [results_json["topics"][t] for t in topics]
    ax3.bar(x, values)
    ax3.set_xticks(x)
    ax3.set_xticklabels(topics, rotation=45, ha="right")
    ax3.set_ylim(0, 1)
    ax3.set_title("Topic Accuracy")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def print_results_table(results_json: dict[str, object], title: str = "SciEval Benchmark Results") -> None:
    console = Console()

    overall = Table(title=title, box=box.SIMPLE_HEAVY)
    overall.add_column("Metric", style="bold")
    overall.add_column("Value", justify="right")

    overall.add_row("Correct", f"{results_json['correct'] * 100:.2f}%")
    overall.add_row("Incorrect", f"{results_json['incorrect'] * 100:.2f}%")
    overall.add_row("Not Answered", f"{results_json['not_answered'] * 100:.2f}%")

    console.print(overall)

    ability_table = Table(title="Ability Accuracy", box=box.MINIMAL_DOUBLE_HEAD)
    ability_table.add_column("Ability")
    ability_table.add_column("Accuracy", justify="right")

    for ability, value in sorted(results_json["abilities"].items()):
        ability_table.add_row(ability, f"{value * 100:.2f}%")

    console.print(ability_table)

    topic_table = Table(title="Topic Accuracy", box=box.MINIMAL_DOUBLE_HEAD)
    topic_table.add_column("Topic")
    topic_table.add_column("Accuracy", justify="right")

    for topic, value in sorted(results_json["topics"].items()):
        topic_table.add_row(topic, f"{value * 100:.2f}%")

    console.print(topic_table)