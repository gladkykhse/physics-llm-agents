import re


def scieval_split_problem_and_options(full_text: str):
    pattern = r"(?:^|\n)\s*A\.\s+"
    match = re.search(pattern, full_text)

    if match:
        question_text = full_text[: match.start()].strip()
        options_text = full_text[match.start() :].strip()
        return question_text, options_text

    return full_text, ""
