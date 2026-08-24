"""Strategy config loader.

The YAML files in this directory stay the human-editable, version-controlled
source; load_config() seeds each one into the configs table the first time
it's asked for, then serves it from there. `file_mtime` is what keeps this
from going stale silently -- if the YAML file on disk is newer than what's
in the database, it's re-read and the row is refreshed, so editing a config
still takes effect without needing a manual re-seed.
"""
from pathlib import Path

import yaml

from db.connection import connect

CONFIG_DIR = Path(__file__).parent


def load_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    file_mtime = path.stat().st_mtime if path.exists() else None

    with connect() as conn:
        row = conn.execute("SELECT yaml, file_mtime FROM configs WHERE name = ?", (name,)).fetchone()

        if row is not None and (file_mtime is None or file_mtime <= row["file_mtime"]):
            return yaml.safe_load(row["yaml"])

        text = path.read_text()
        conn.execute(
            "INSERT INTO configs (name, yaml, file_mtime) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET yaml = excluded.yaml, file_mtime = excluded.file_mtime",
            (name, text, file_mtime),
        )
        return yaml.safe_load(text)
