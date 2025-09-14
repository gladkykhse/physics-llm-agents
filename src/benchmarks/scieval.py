import os
import datasets
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


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


def parse_results_to_dict(df: pl.DataFrame) -> dict[str, object]:
    df = df.with_columns(
        std_ans=pl.col("answer_standard_system_prompt").str.extract(r"([ABCD])"),
        cot_ans=pl.col("answer_cot_system_prompt").str.extract(r"(?i)answer.*?([ABCD])").str.to_uppercase(),
    )

    df = df.with_columns(
        correct_std=pl.col("answer").list.contains(pl.col("std_ans")),
        correct_cot=pl.col("answer").list.contains(pl.col("cot_ans")),
    )

    overall_standard = float(df["correct_std"].mean())
    overall_cot = float(df["correct_cot"].mean())

    ability_accuracy = (
        df.group_by("ability")
        .agg(
            pl.col("correct_std").mean().alias("acc_std"),
            pl.col("correct_cot").mean().alias("acc_cot"),
        )
        .sort("ability")
    )

    topic_accuracy = (
        df.group_by("topic")
        .agg(
            pl.col("correct_std").mean().alias("acc_std"),
            pl.col("correct_cot").mean().alias("acc_cot"),
        )
        .sort("topic")
    )

    abilities_standard = {row["ability"]: round(float(row["acc_std"]), 4) for row in ability_accuracy.to_dicts()}
    abilities_cot = {row["ability"]: round(float(row["acc_cot"]), 4) for row in ability_accuracy.to_dicts()}
    topics_standard = {row["topic"]: round(float(row["acc_std"]), 4) for row in topic_accuracy.to_dicts()}
    topics_cot = {row["topic"]: round(float(row["acc_cot"]), 4) for row in topic_accuracy.to_dicts()}

    results_json = {
        "standard": {
            "overall": round(overall_standard, 4),
            "abilities": abilities_standard,
            "topics": topics_standard,
        },
        "cot": {
            "overall": round(overall_cot, 4),
            "abilities": abilities_cot,
            "topics": topics_cot,
        },
    }

    return results_json


def plot_results(results_json: dict[str, object], filename: str) -> None:
    overall_standard = results_json["standard"]["overall"]
    overall_cot = results_json["cot"]["overall"]

    abilities = sorted(
        set(results_json["standard"]["abilities"].keys()) | set(results_json["cot"]["abilities"].keys())
    )
    topics = sorted(set(results_json["standard"]["topics"].keys()) | set(results_json["cot"]["topics"].keys()))

    abilities_std = [results_json["standard"]["abilities"].get(a, np.nan) for a in abilities]
    abilities_cot = [results_json["cot"]["abilities"].get(a, np.nan) for a in abilities]

    topics_std = [results_json["standard"]["topics"].get(t, np.nan) for t in topics]
    topics_cot = [results_json["cot"]["topics"].get(t, np.nan) for t in topics]

    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(["Standard", "CoT"], [overall_standard, overall_cot], color=["tab:blue", "tab:orange"])
    ax1.set_title("Overall Accuracy")
    ax1.set_ylabel("Accuracy")

    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(abilities))
    width = 0.35
    ax2.bar(x - width / 2, abilities_std, width, label="Standard", color="tab:blue")
    ax2.bar(x + width / 2, abilities_cot, width, label="CoT", color="tab:orange")
    ax2.set_xticks(x)
    ax2.set_xticklabels(abilities, rotation=0)
    ax2.set_title("Ability Accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(topics))
    ax3.bar(x - width / 2, topics_std, width, label="Standard", color="tab:blue")
    ax3.bar(x + width / 2, topics_cot, width, label="CoT", color="tab:orange")
    ax3.set_xticks(x)
    ax3.set_xticklabels(topics, rotation=45, ha="right")
    ax3.set_title("Topic Accuracy")
    ax3.set_ylabel("Accuracy")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(filename)
