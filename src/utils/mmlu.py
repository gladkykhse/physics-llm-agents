import polars as pl

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
