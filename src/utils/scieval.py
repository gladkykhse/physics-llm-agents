import polars as pl


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
