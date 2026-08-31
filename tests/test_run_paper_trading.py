import os
import time

import numpy as np
import pandas as pd
import pytest

import db.connection as db_connection
import run_paper_trading as rpt
from db.connection import connect


@pytest.fixture(autouse=True)
def _isolated_logging_and_db(tmp_path, monkeypatch):
    """Keep each test's log file and DB inside its own tmp_path, and stop
    handlers from accumulating on the module-level logger across tests."""
    monkeypatch.setattr(rpt, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(db_connection, "DB_PATH", tmp_path / "test.db")
    rpt.logger.handlers.clear()
    yield
    rpt.logger.handlers.clear()


def _fake_ohlcv(symbol, start, end, adjust=True, use_cache=True, allow_yahoo_fallback=True):
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    dates = pd.bdate_range(start, end)
    closes = 100 + np.abs(rng.normal(0, 1, len(dates)).cumsum())
    return pd.DataFrame(
        {
            "OPEN": closes,
            "HIGH": closes * 1.01,
            "LOW": closes * 0.99,
            "CLOSE": closes,
            "VOLUME": rng.integers(10_000, 50_000, len(dates)),
        },
        index=dates,
    )


def _account_exists(name: str) -> bool:
    with connect() as conn:
        return conn.execute("SELECT 1 FROM accounts WHERE name = ?", (name,)).fetchone() is not None


CONFIG = {
    "strategy": "mean_reversion",
    "params": {"lookback": 10, "entry_zscore": -2.0, "exit_zscore": 0.0},
    "universe": {"index": "NIFTY500", "min_avg_daily_value": 0},
    "costs": {"brokerage_pct": 0.0003},
    "risk": {"max_concurrent_positions": 5, "position_size_pct": 0.1},
}


def _patch_common(monkeypatch, tmp_path):
    monkeypatch.setattr(rpt, "STATE_DIR", tmp_path)  # lock files only -- account state lives in the DB
    monkeypatch.setattr(rpt, "load_config", lambda name: CONFIG)
    monkeypatch.setattr("execution.paper_trading.get_ohlcv", _fake_ohlcv)
    monkeypatch.setattr(
        "execution.paper_trading.get_corporate_actions",
        lambda symbol, use_cache=True: pd.DataFrame(columns=["ex_date", "action_type", "ratio"]),
    )


def test_daily_step_writes_state_for_the_account(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)

    summary = rpt.main(
        ["--config", "mean_reversion", "--symbol", "RELIANCE", "--as-of", "2023-06-15"]
    )

    assert summary["skipped"] is False
    assert _account_exists("mean_reversion_RELIANCE")


def test_report_flag_reads_ledger_and_writes_tearsheet(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)

    rpt.main(["--config", "mean_reversion", "--symbol", "RELIANCE", "--as-of", "2023-06-15"])
    rpt.main(
        [
            "--config", "mean_reversion",
            "--symbol", "RELIANCE",
            "--report",
            "--output-dir", str(tmp_path / "reports"),
        ]
    )

    assert (tmp_path / "reports" / "mean_reversion_RELIANCE.html").exists()


def test_a_fresh_lock_blocks_a_concurrent_run(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)
    lock_path = tmp_path / "mean_reversion_RELIANCE.lock"
    lock_path.write_text("12345")  # simulates another invocation currently holding the lock

    summary = rpt.main(["--config", "mean_reversion", "--symbol", "RELIANCE", "--as-of", "2023-06-15"])

    assert summary["skipped"] is True
    assert "in progress" in summary["reason"]
    assert not _account_exists("mean_reversion_RELIANCE")  # no step was run


def test_a_stale_lock_is_overridden(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)
    lock_path = tmp_path / "mean_reversion_RELIANCE.lock"
    lock_path.write_text("12345")
    old_time = time.time() - rpt.STALE_LOCK_SECONDS - 60
    os.utime(lock_path, (old_time, old_time))

    summary = rpt.main(["--config", "mean_reversion", "--symbol", "RELIANCE", "--as-of", "2023-06-15"])

    assert summary["skipped"] is False
    assert _account_exists("mean_reversion_RELIANCE")


def test_lock_is_released_after_a_successful_run(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)

    rpt.main(["--config", "mean_reversion", "--symbol", "RELIANCE", "--as-of", "2023-06-15"])

    assert not (tmp_path / "mean_reversion_RELIANCE.lock").exists()


def test_lock_is_released_even_if_the_daily_step_raises(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)

    def boom(self, as_of=None):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(rpt.PaperTradingEngine, "run_daily_step", boom)

    with pytest.raises(RuntimeError, match="simulated crash"):
        rpt.main(["--config", "mean_reversion", "--symbol", "RELIANCE", "--as-of", "2023-06-15"])

    assert not (tmp_path / "mean_reversion_RELIANCE.lock").exists()


CONFIG_ML = {
    "strategy": "ml_strategy",
    "params": {
        "horizon": 5, "return_threshold": 0.0, "entry_threshold": 0.6, "exit_threshold": 0.45,
        "max_iter": 30, "max_depth": 2, "learning_rate": 0.1, "min_samples_leaf": 5, "l2_regularization": 0.1,
    },
    "universe": {"index": "NIFTY500", "min_avg_daily_value": 0},
    "costs": {"brokerage_pct": 0.0003},
    "risk": {"max_concurrent_positions": 5, "position_size_pct": 0.1},
}


def _patch_common_ml(monkeypatch, tmp_path):
    monkeypatch.setattr(rpt, "STATE_DIR", tmp_path)
    monkeypatch.setattr(rpt, "load_config", lambda name: CONFIG_ML)
    monkeypatch.setattr(rpt, "get_ohlcv", _fake_ohlcv)
    monkeypatch.setattr("execution.paper_trading.get_ohlcv", _fake_ohlcv)
    monkeypatch.setattr(
        "execution.paper_trading.get_corporate_actions",
        lambda symbol, use_cache=True: pd.DataFrame(columns=["ex_date", "action_type", "ratio"]),
    )


def _ml_model_row(account: str):
    with connect() as conn:
        return conn.execute(
            "SELECT fitted_at, model_blob FROM ml_models WHERE account = ?", (account,)
        ).fetchone()


def test_ml_strategy_fits_fresh_and_persists_a_model(monkeypatch, tmp_path):
    _patch_common_ml(monkeypatch, tmp_path)

    summary = rpt.main(
        ["--config", "ml_strategy_nifty50", "--symbol", "RELIANCE", "--as-of", "2023-06-15", "--train-days", "150"]
    )

    assert summary["skipped"] is False
    row = _ml_model_row("ml_strategy_nifty50_RELIANCE")
    assert row is not None
    assert row["fitted_at"] == "2023-06-15"


def test_ml_strategy_reuses_a_fresh_model_without_refitting(monkeypatch, tmp_path):
    _patch_common_ml(monkeypatch, tmp_path)

    rpt.main(["--config", "ml_strategy_nifty50", "--symbol", "RELIANCE", "--as-of", "2023-06-15", "--train-days", "150"])
    first_blob = _ml_model_row("ml_strategy_nifty50_RELIANCE")["model_blob"]

    rpt.main(
        [
            "--config", "ml_strategy_nifty50", "--symbol", "RELIANCE",
            "--as-of", "2023-06-20", "--train-days", "150", "--refit-days", "126",
        ]
    )

    row = _ml_model_row("ml_strategy_nifty50_RELIANCE")
    assert row["fitted_at"] == "2023-06-15"  # unchanged
    assert row["model_blob"] == first_blob  # not rewritten


def test_ml_strategy_refits_once_the_model_is_stale(monkeypatch, tmp_path):
    _patch_common_ml(monkeypatch, tmp_path)

    rpt.main(["--config", "ml_strategy_nifty50", "--symbol", "RELIANCE", "--as-of", "2023-06-15", "--train-days", "150"])
    first_blob = _ml_model_row("ml_strategy_nifty50_RELIANCE")["model_blob"]

    rpt.main(
        [
            "--config", "ml_strategy_nifty50", "--symbol", "RELIANCE",
            "--as-of", "2024-01-15", "--train-days", "150", "--refit-days", "126",
        ]
    )

    row = _ml_model_row("ml_strategy_nifty50_RELIANCE")
    assert row["fitted_at"] == "2024-01-15"
    assert row["model_blob"] != first_blob
