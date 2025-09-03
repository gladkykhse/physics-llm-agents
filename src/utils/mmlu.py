import polars as pl
import matplotlib.pyplot as plt

SUBSETS = [
    "MMLU_college_physics",
    "MMLU_conceptual_physics",
    "MMLU_high_school_physics",
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


def load_dataframe(source: str = "benchmarks/MMLU_college_physics/test/data-00000-of-00001.arrow") -> pl.DataFrame:
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
        pl.col("answer_standard_system_prompt").str.extract(r"([ABCD])").alias("std_ans"),
        pl.col("answer_cot_system_prompt").str.extract(r"(?i)answer.*?([ABCD])").str.to_uppercase().alias("cot_ans"),
        pl.col("answer").map_elements(lambda x: [NUM_TO_LETTER[x]], return_dtype=pl.List(pl.String)),
    )

    df = df.with_columns(
        correct_std=pl.col("answer").list.contains(pl.col("std_ans")),
        correct_cot=pl.col("answer").list.contains(pl.col("cot_ans")),
    )

    overall_standard = float(df["correct_std"].mean())
    overall_cot = float(df["correct_cot"].mean())

    results_json = {
        "standard": {
            "overall": round(overall_standard, 4),
        },
        "cot": {
            "overall": round(overall_cot, 4),
        },
    }

    return results_json

def plot_results(results_json: dict[str, object], filename: str) -> None:
    overall_standard = results_json["standard"]["overall"]
    overall_cot = results_json["cot"]["overall"]

    plt.bar(["Standard", "CoT"], [overall_standard, overall_cot], color=["tab:blue", "tab:orange"])
    plt.title("Overall Accuracy")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(filename)