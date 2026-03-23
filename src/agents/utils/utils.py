import re


def scieval_split_problem_and_options(full_text: str):
    # Regex to find the start of the options block (starting with "A.")
    # It looks for "A." preceded by a newline or start of string
    pattern = r"(?:^|\n)\s*A\.\s+"
    match = re.search(pattern, full_text)

    if match:
        question_text = full_text[: match.start()].strip()
        # Keep the "A." and everything after it
        options_text = full_text[match.start() :].strip()
        return question_text, options_text

    # Fallback if no options found (e.g. pure numeric question)
    return full_text, ""
