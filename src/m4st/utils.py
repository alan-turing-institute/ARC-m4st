import json
import os


def load_json(json_path: os.PathLike | str) -> list:

    with open(json_path) as input_file:
        return json.load(input_file)
