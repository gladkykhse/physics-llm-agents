import json

import yaml


def load_yaml(file: str) -> dict:
    with open(file, "r") as f:
        return yaml.safe_load(f)


def save_json(obj: dict[str, object], filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(obj, f)
