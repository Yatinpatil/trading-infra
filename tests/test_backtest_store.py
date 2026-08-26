import pytest

import db.connection as db_connection
from analytics.backtest_store import load_backtest_results, save_backtest_result


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_connection, "DB_PATH", tmp_path / "test.db")


METRICS = {
    "cagr": 0.05, "sharpe": 0.6, "sortino": 0.7, "max_drawdown": -0.12,
    "win_rate": 0.55, "profit_factor": 1.3, "num_trades": 100, "total_return": 0.3,
}


def test_load_backtest_results_is_empty_before_anything_saved():
    assert load_backtest_results() == {}


def test_save_and_load_round_trips():
    save_backtest_result("mean_reversion", "2021-01-01", "2026-08-24", "2026-08-25T11:00:00", METRICS, 1_300_000.0)

    results = load_backtest_results()
    assert set(results) == {"mean_reversion"}
    row = results["mean_reversion"]
    assert row["cagr"] == 0.05
    assert row["num_trades"] == 100
    assert row["final_equity"] == 1_300_000.0
    assert row["start_date"] == "2021-01-01"


def test_saving_again_replaces_rather_than_duplicates():
    save_backtest_result("momentum", "2021-01-01", "2026-08-24", "2026-08-25T11:00:00", METRICS, 1_300_000.0)
    updated = {**METRICS, "cagr": 0.09}
    save_backtest_result("momentum", "2021-01-01", "2026-08-24", "2026-08-25T12:00:00", updated, 1_400_000.0)

    results = load_backtest_results()
    assert len(results) == 1
    assert results["momentum"]["cagr"] == 0.09
    assert results["momentum"]["final_equity"] == 1_400_000.0
