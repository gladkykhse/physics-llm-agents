import json


def save_json(obj: dict[str, object], filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(obj, f)
