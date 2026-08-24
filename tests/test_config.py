import os
import time

import pytest

import configs
import db.connection as db_connection
from configs import load_config
from db.connection import connect


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_connection, "DB_PATH", tmp_path / "test.db")


def test_load_mean_reversion_config():
    config = load_config("mean_reversion")
    assert config["strategy"] == "mean_reversion"
    assert "params" in config
    assert "costs" in config
    assert "risk" in config


def test_first_load_seeds_the_db_from_the_yaml_file():
    load_config("mean_reversion")
    with connect() as conn:
        row = conn.execute("SELECT yaml FROM configs WHERE name = 'mean_reversion'").fetchone()
    assert row is not None
    assert "mean_reversion" in row["yaml"]


def test_a_config_that_only_exists_in_the_db_loads_without_touching_disk():
    # e.g. a config seeded once, whose YAML source has since been removed --
    # load_config must not require the file to exist to serve it from the DB.
    with connect() as conn:
        conn.execute(
            "INSERT INTO configs (name, yaml, file_mtime) VALUES (?, ?, ?)",
            ("db_only_config", "strategy: buy_and_hold\nparams: {}\n", None),
        )

    config = load_config("db_only_config")

    assert config["strategy"] == "buy_and_hold"


def test_editing_the_yaml_file_after_seeding_is_picked_up_on_next_load(tmp_path, monkeypatch):
    monkeypatch.setattr(configs, "CONFIG_DIR", tmp_path)
    config_path = tmp_path / "fake_strategy.yaml"
    config_path.write_text("strategy: mean_reversion\nparams: {}\n")

    load_config("fake_strategy")  # seeds the DB

    # bump the file's mtime forward so it reads as newer than the DB row
    future = time.time() + 10
    config_path.write_text("strategy: mean_reversion\nparams: {}\nextra_marker: true\n")
    os.utime(config_path, (future, future))

    refreshed = load_config("fake_strategy")
    assert refreshed.get("extra_marker") is True
