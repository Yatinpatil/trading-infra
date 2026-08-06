from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent


def load_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)
