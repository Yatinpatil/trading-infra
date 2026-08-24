import pytest

import db.connection as db_connection
from db.connection import connect
from execution.accounts import ACCOUNTS, account_by_key, load_account_state


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_connection, "DB_PATH", tmp_path / "test.db")


def test_account_by_key_finds_each_declared_account():
    for meta in ACCOUNTS:
        assert account_by_key(meta["key"]) is meta


def test_account_by_key_raises_for_an_unknown_key():
    with pytest.raises(ValueError, match="Unknown account key"):
        account_by_key("does_not_exist")


def test_load_account_state_for_a_never_started_account():
    meta = account_by_key("mean_reversion")
    state = load_account_state(meta)

    assert state["started"] is False
    assert state["status"] == "not_started"
    assert state["equity"] is None
    assert state["positions"] == []
    assert state["trades"] == []
    assert state["metrics"] is None


def test_load_account_state_reflects_a_saved_broker():
    from execution.broker import PaperBroker

    meta = account_by_key("mean_reversion")
    broker = PaperBroker(meta["account"], initial_capital=100_000)
    broker.open_position("AAA", "2024-01-02", price=100.0, quantity=10, entry_cost=0.0, stop_price=95.0)
    broker.record_equity("2024-01-02", 99_000)
    broker.last_run_date = "2024-01-02"
    broker.save()

    state = load_account_state(meta)

    assert state["started"] is True
    assert state["cash"] == broker.cash
    assert state["num_open_positions"] == 1
    assert state["positions"][0]["symbol"] == "AAA"
    assert state["last_run_date"] == "2024-01-02"


def test_load_account_state_trade_limit_caps_returned_trades():
    from execution.broker import PaperBroker

    meta = account_by_key("mean_reversion")
    broker = PaperBroker(meta["account"], initial_capital=100_000)
    for i in range(3):
        broker.open_position(f"SYM{i}", "2024-01-01", price=100.0, quantity=1, entry_cost=0.0, stop_price=None)
        broker.close_position(f"SYM{i}", "2024-01-02", price=110.0, exit_cost=0.0, reason="signal")

    state = load_account_state(meta, trade_limit=2)
    assert len(state["trades"]) == 2
    assert state["num_trades"] == 3  # the count still reflects everything, only the list is capped

    state_all = load_account_state(meta, trade_limit=None)
    assert len(state_all["trades"]) == 3
