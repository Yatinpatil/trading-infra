import pandas as pd
import pytest

from universe import constituents


@pytest.fixture
def history_path(tmp_path, monkeypatch):
    path = tmp_path / "constituents_history.csv"
    monkeypatch.setattr(constituents, "HISTORY_PATH", path)
    return path


def _write_history(path, rows):
    pd.DataFrame(rows, columns=constituents.HISTORY_COLUMNS).to_csv(path, index=False)


def test_get_universe_returns_active_members_on_date(history_path):
    _write_history(
        history_path,
        [
            {"index_name": "NIFTY 50", "symbol": "AAA", "start_date": "2020-01-01", "end_date": ""},
            {"index_name": "NIFTY 50", "symbol": "BBB", "start_date": "2020-01-01", "end_date": "2022-06-01"},
            {"index_name": "NIFTY 50", "symbol": "CCC", "start_date": "2022-06-01", "end_date": ""},
        ],
    )

    assert constituents.get_universe("NIFTY 50", "2021-01-01") == ["AAA", "BBB"]
    assert constituents.get_universe("NIFTY 50", "2023-01-01") == ["AAA", "CCC"]


def test_get_universe_before_history_warns_and_falls_back(history_path):
    _write_history(
        history_path,
        [{"index_name": "NIFTY 50", "symbol": "AAA", "start_date": "2022-01-01", "end_date": ""}],
    )

    with pytest.warns(UserWarning, match="predates earliest recorded snapshot"):
        result = constituents.get_universe("NIFTY 50", "2015-01-01")

    assert result == ["AAA"]


def test_get_universe_no_history_warns_and_uses_current(history_path, monkeypatch):
    monkeypatch.setattr(constituents, "get_current_constituents", lambda index_name: ["ZZZ"])

    with pytest.warns(UserWarning, match="No point-in-time history"):
        result = constituents.get_universe("NIFTY 50", "2021-01-01")

    assert result == ["ZZZ"]


def test_snapshot_constituents_closes_dropped_and_adds_new(history_path, monkeypatch):
    _write_history(
        history_path,
        [
            {"index_name": "NIFTY 50", "symbol": "AAA", "start_date": "2020-01-01", "end_date": ""},
            {"index_name": "NIFTY 50", "symbol": "BBB", "start_date": "2020-01-01", "end_date": ""},
        ],
    )
    monkeypatch.setattr(constituents, "get_current_constituents", lambda index_name: ["AAA", "CCC"])

    constituents.snapshot_constituents("NIFTY 50", as_of="2024-01-01")

    history = pd.read_csv(history_path, parse_dates=["start_date", "end_date"])
    bbb = history[history["symbol"] == "BBB"].iloc[0]
    assert bbb["end_date"] == pd.Timestamp("2024-01-01")

    ccc = history[history["symbol"] == "CCC"].iloc[0]
    assert ccc["start_date"] == pd.Timestamp("2024-01-01")
    assert pd.isna(ccc["end_date"])

    aaa = history[history["symbol"] == "AAA"].iloc[0]
    assert pd.isna(aaa["end_date"])
