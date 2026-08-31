import pandas as pd
import pytest

import db.connection as db_connection
from execution.broker import PaperBroker


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_connection, "DB_PATH", tmp_path / "test.db")


def test_new_broker_starts_with_initial_capital_and_no_positions():
    broker = PaperBroker("acct", initial_capital=100_000)
    assert broker.cash == 100_000
    assert broker.positions == {}
    assert broker.last_run_date is None


def test_open_position_debits_cash_including_cost():
    broker = PaperBroker("acct", initial_capital=100_000)
    broker.open_position("AAA", "2024-01-02", price=100.0, quantity=10, entry_cost=5.0, stop_price=95.0)

    assert broker.cash == 100_000 - 10 * 100.0 - 5.0
    position = broker.positions["AAA"]
    assert position.quantity == 10
    assert position.entry_price == 100.0
    assert position.stop_price == 95.0


def test_close_position_credits_cash_and_removes_position():
    broker = PaperBroker("acct", initial_capital=100_000)
    broker.open_position("AAA", "2024-01-02", price=100.0, quantity=10, entry_cost=5.0, stop_price=95.0)

    row = broker.close_position("AAA", "2024-01-03", price=110.0, exit_cost=6.0, reason="signal")

    assert "AAA" not in broker.positions
    assert broker.cash == 100_000 - 1000 - 5.0 + 1100 - 6.0
    assert row["pnl"] == (110.0 - 100.0) * 10 - 5.0 - 6.0
    assert row["exit_reason"] == "signal"


def test_trades_and_equity_round_trip_through_new_instance():
    broker = PaperBroker("acct", initial_capital=100_000)
    broker.open_position("AAA", "2024-01-02", price=100.0, quantity=10, entry_cost=0.0, stop_price=None)
    broker.close_position("AAA", "2024-01-03", price=110.0, exit_cost=0.0, reason="signal")
    broker.record_equity("2024-01-02", 100_000)
    broker.record_equity("2024-01-03", 100_100)
    broker.last_run_date = "2024-01-03"
    broker.save()

    reloaded = PaperBroker("acct")
    assert reloaded.cash == broker.cash
    assert reloaded.last_run_date == "2024-01-03"
    assert reloaded.positions == {}

    trades = reloaded.trades()
    assert len(trades) == 1
    assert trades.iloc[0]["symbol"] == "AAA"

    equity = reloaded.equity_curve()
    assert list(equity.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert list(equity.values) == [100_000, 100_100]


def test_trades_includes_holding_days():
    """analytics.metrics.avg_trade_duration() reads trades["holding_days"]
    -- the same column engine/portfolio.py's backtest trades always
    included, but PaperBroker's live trades didn't, which only surfaced
    once a real paper-trading account closed its first trade (until then
    compute_metrics never actually touched an empty trades frame's
    columns) and crashed the dashboard API with a KeyError.
    """
    broker = PaperBroker("acct", initial_capital=100_000)
    broker.open_position("AAA", "2024-01-02", price=100.0, quantity=10, entry_cost=0.0, stop_price=None)
    broker.close_position("AAA", "2024-01-05", price=110.0, exit_cost=0.0, reason="signal")

    trades = broker.trades()
    assert trades.iloc[0]["holding_days"] == 3

    from analytics.metrics import avg_trade_duration

    assert avg_trade_duration(trades) == 3.0


def test_record_equity_dedupes_same_day_on_read():
    broker = PaperBroker("acct", initial_capital=100_000)
    broker.record_equity("2024-01-02", 100_000)
    broker.record_equity("2024-01-02", 101_000)  # e.g. a re-run of the same day

    equity = broker.equity_curve()
    assert len(equity) == 1
    assert equity.iloc[0] == 101_000  # last write for that date wins


def test_two_accounts_in_the_same_db_are_independent():
    a = PaperBroker("acct-a", initial_capital=100_000)
    b = PaperBroker("acct-b", initial_capital=200_000)
    a.open_position("AAA", "2024-01-02", price=100.0, quantity=10, entry_cost=0.0, stop_price=None)
    a.last_run_date = "2024-01-02"
    a.save()
    b.save()

    reloaded_b = PaperBroker("acct-b")
    assert reloaded_b.cash == 200_000
    assert reloaded_b.positions == {}
    assert reloaded_b.last_run_date is None
