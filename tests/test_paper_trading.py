import pandas as pd
import pytest

import db.connection as db_connection
from execution.broker import PaperBroker
from execution.paper_trading import PaperTradingEngine
from strategies.base import Strategy, empty_signals


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_connection, "DB_PATH", tmp_path / "test.db")


@pytest.fixture(autouse=True)
def _no_corporate_actions(monkeypatch):
    """Every test here uses synthetic symbols with no real corporate-action
    cache — stub the lookup so run_daily_step never touches the network."""
    empty = pd.DataFrame(columns=["ex_date", "action_type", "ratio"])
    monkeypatch.setattr("execution.paper_trading.get_corporate_actions", lambda symbol, use_cache=True: empty)


class FixedSignalStrategy(Strategy):
    """Same helper pattern as tests/test_portfolio_engine.py — fixed
    entry/exit dates, independent of real indicator behavior."""

    def __init__(self, entry_dates=(), exit_dates=()):
        super().__init__({})
        self.entry_dates = {pd.Timestamp(d) for d in entry_dates}
        self.exit_dates = {pd.Timestamp(d) for d in exit_dates}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = empty_signals(df.index)
        signals.loc[signals.index.isin(self.entry_dates), "entry_long"] = True
        signals.loc[signals.index.isin(self.exit_dates), "exit_long"] = True
        return signals


def _ohlcv(rows):
    dates = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {
            "OPEN": [r[1] for r in rows],
            "HIGH": [r[2] for r in rows],
            "LOW": [r[3] for r in rows],
            "CLOSE": [r[4] for r in rows],
            "VOLUME": [1000] * len(rows),
        },
        index=dates,
    )


NO_LIMITS_CONFIG = {"costs": {}, "risk": {"position_size_pct": 0.1}, "params": {}}


def _fake_get_ohlcv(data_by_symbol):
    def fake(symbol, start, end, adjust=True, use_cache=True):
        df = data_by_symbol[symbol]
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    return fake


def test_entry_queues_then_fills_next_open_and_stop_loss_closes_it(tmp_path, monkeypatch):
    aaa = _ohlcv(
        [
            ("2024-01-01", 100, 101, 99, 100),  # entry signal fires at today's close
            ("2024-01-02", 100, 101, 99, 100),  # queued entry fills at today's open; stop = 95
            ("2024-01-03", 99, 100, 90, 96),    # low breaches stop -> exit @ 95
            ("2024-01-04", 96, 97, 95, 96),
        ]
    )
    monkeypatch.setattr("execution.paper_trading.get_ohlcv", _fake_get_ohlcv({"AAA": aaa}))

    strategy = FixedSignalStrategy(entry_dates=["2024-01-01"])
    config = {"costs": {}, "risk": {"position_size_pct": 0.1}, "params": {"stop_loss_pct": 0.05}}
    broker = PaperBroker("acct", initial_capital=100_000)
    engine = PaperTradingEngine(strategy, config, broker, ["AAA"], history_days=10)

    step1 = engine.run_daily_step("2024-01-01")
    assert step1["open_positions"] == []
    assert broker.pending_entries == {"AAA"}

    step2 = engine.run_daily_step("2024-01-02")
    assert step2["open_positions"] == ["AAA"]
    assert step2["trades_today"] == []
    assert broker.positions["AAA"].entry_price == 100.0
    assert broker.positions["AAA"].stop_price == 95.0

    step3 = engine.run_daily_step("2024-01-03")
    assert step3["open_positions"] == []
    assert len(step3["trades_today"]) == 1
    trade = step3["trades_today"][0]
    assert trade["exit_reason"] == "stop_loss"
    assert trade["exit_price"] == 95.0

    assert len(broker.trades()) == 1
    assert len(broker.equity_curve()) == 3


def test_running_the_same_day_twice_is_a_no_op(tmp_path, monkeypatch):
    aaa = _ohlcv([("2024-01-01", 100, 101, 99, 100), ("2024-01-02", 100, 101, 99, 100)])
    monkeypatch.setattr("execution.paper_trading.get_ohlcv", _fake_get_ohlcv({"AAA": aaa}))

    strategy = FixedSignalStrategy(entry_dates=["2024-01-01"])
    broker = PaperBroker("acct", initial_capital=100_000)
    engine = PaperTradingEngine(strategy, NO_LIMITS_CONFIG, broker, ["AAA"], history_days=10)

    engine.run_daily_step("2024-01-01")
    repeat = engine.run_daily_step("2024-01-01")

    assert repeat == {"skipped": True, "reason": "already ran for 2024-01-01", "as_of": "2024-01-01"}
    assert len(broker.equity_curve()) == 1


def test_state_persists_across_new_broker_and_engine_instances(tmp_path, monkeypatch):
    aaa = _ohlcv(
        [
            ("2024-01-01", 100, 101, 99, 100),
            ("2024-01-02", 100, 101, 99, 100),  # entry fills here
            ("2024-01-03", 110, 111, 109, 110),  # exit signal fires at close, fills next step
            ("2024-01-04", 112, 113, 111, 112),
        ]
    )
    monkeypatch.setattr("execution.paper_trading.get_ohlcv", _fake_get_ohlcv({"AAA": aaa}))

    strategy = FixedSignalStrategy(entry_dates=["2024-01-01"], exit_dates=["2024-01-03"])

    broker1 = PaperBroker("acct", initial_capital=100_000)
    engine1 = PaperTradingEngine(strategy, NO_LIMITS_CONFIG, broker1, ["AAA"], history_days=10)
    engine1.run_daily_step("2024-01-01")
    engine1.run_daily_step("2024-01-02")
    assert "AAA" in broker1.positions

    # simulate a fresh process picking the persisted account back up
    broker2 = PaperBroker("acct", initial_capital=999_999)
    assert broker2.cash == broker1.cash
    assert "AAA" in broker2.positions

    engine2 = PaperTradingEngine(strategy, NO_LIMITS_CONFIG, broker2, ["AAA"], history_days=10)
    engine2.run_daily_step("2024-01-03")
    step4 = engine2.run_daily_step("2024-01-04")

    assert step4["open_positions"] == []
    assert len(broker2.trades()) == 1
    assert broker2.trades().iloc[0]["exit_reason"] == "signal"


def test_execution_is_immune_to_a_pending_future_corporate_action(tmp_path, monkeypatch):
    """A split/bonus can be announced (ex-date known) before it takes effect.
    Backward-adjustment rescales every date strictly before the ex-date —
    including "today", if today is still before that future ex-date — which
    would wrongly discount a fill/mark-to-market price for an event that
    hasn't happened yet. Execution must use the raw series, not the adjusted
    one, so a pending announcement can't distort real money math.
    """
    aaa_raw = _ohlcv(
        [
            ("2024-01-01", 100, 101, 99, 100),
            ("2024-01-02", 100, 101, 99, 100),  # entry fills here
            ("2024-01-03", 100, 101, 99, 100),
        ]
    )

    def fake_get_ohlcv(symbol, start, end, adjust=True, use_cache=True):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        window = aaa_raw.loc[(aaa_raw.index >= start_ts) & (aaa_raw.index <= end_ts)].copy()
        if not adjust:
            return window
        # A bonus announced with a FUTURE ex-date (beyond `end`): naive
        # backward adjustment discounts every date in the window, including
        # "today", even though the event hasn't happened yet.
        for col in ["OPEN", "HIGH", "LOW", "CLOSE"]:
            window[col] = window[col] / 2.0
        return window

    monkeypatch.setattr("execution.paper_trading.get_ohlcv", fake_get_ohlcv)

    strategy = FixedSignalStrategy(entry_dates=["2024-01-01"])
    config = {"costs": {}, "risk": {"position_size_pct": 0.1}, "params": {}}
    broker = PaperBroker("acct", initial_capital=100_000)
    engine = PaperTradingEngine(strategy, config, broker, ["AAA"], history_days=10)

    engine.run_daily_step("2024-01-01")
    engine.run_daily_step("2024-01-02")

    # fill must happen at the real traded price (100), not the artificially
    # halved "adjusted" price (50) a pending-action recompute would produce
    assert broker.positions["AAA"].entry_price == 100.0
    assert broker.cash == 100_000 - broker.positions["AAA"].quantity * 100.0


def test_max_concurrent_positions_limits_simultaneous_paper_entries(tmp_path, monkeypatch):
    aaa = _ohlcv([("2024-01-01", 100, 101, 99, 100), ("2024-01-02", 100, 101, 99, 100), ("2024-01-03", 100, 101, 99, 100)])
    bbb = _ohlcv([("2024-01-01", 50, 51, 49, 50), ("2024-01-02", 50, 51, 49, 50), ("2024-01-03", 50, 51, 49, 50)])
    monkeypatch.setattr("execution.paper_trading.get_ohlcv", _fake_get_ohlcv({"AAA": aaa, "BBB": bbb}))

    strategy = FixedSignalStrategy(entry_dates=["2024-01-01"])
    config = {"costs": {}, "risk": {"position_size_pct": 0.1, "max_concurrent_positions": 1}, "params": {}}
    broker = PaperBroker("acct", initial_capital=100_000)
    engine = PaperTradingEngine(strategy, config, broker, ["AAA", "BBB"], history_days=10)

    engine.run_daily_step("2024-01-01")
    engine.run_daily_step("2024-01-02")

    assert set(broker.positions) == {"AAA"}  # alphabetically-first symbol wins the single slot


def test_non_trading_day_is_skipped_without_mutating_state(tmp_path, monkeypatch):
    aaa = _ohlcv([("2024-01-01", 100, 101, 99, 100), ("2024-01-02", 100, 101, 99, 100)])
    monkeypatch.setattr("execution.paper_trading.get_ohlcv", _fake_get_ohlcv({"AAA": aaa}))

    strategy = FixedSignalStrategy()
    broker = PaperBroker("acct", initial_capital=100_000)
    engine = PaperTradingEngine(strategy, NO_LIMITS_CONFIG, broker, ["AAA"], history_days=10)

    # 2024-01-03 (a Wednesday, so not a weekend) has no data at all -- e.g. an
    # NSE holiday -- and must be a clean no-op, not an equity-curve entry
    # built from stale marks.
    result = engine.run_daily_step("2024-01-03")

    assert result["skipped"] is True
    assert broker.last_run_date is None
    assert len(broker.equity_curve()) == 0


def test_one_symbols_fetch_failure_does_not_block_the_rest_of_the_universe(tmp_path, monkeypatch):
    aaa = _ohlcv([("2024-01-01", 100, 101, 99, 100), ("2024-01-02", 100, 101, 99, 100)])
    bbb = _ohlcv([("2024-01-01", 50, 51, 49, 50), ("2024-01-02", 50, 51, 49, 50)])
    good = _fake_get_ohlcv({"AAA": aaa, "BBB": bbb})

    def flaky(symbol, start, end, adjust=True, use_cache=True):
        if symbol == "BBB":
            raise ConnectionError("NSE is down")
        return good(symbol, start, end, adjust=adjust, use_cache=use_cache)

    monkeypatch.setattr("execution.paper_trading.get_ohlcv", flaky)

    strategy = FixedSignalStrategy(entry_dates=["2024-01-01"])
    broker = PaperBroker("acct", initial_capital=100_000)
    engine = PaperTradingEngine(strategy, NO_LIMITS_CONFIG, broker, ["AAA", "BBB"], history_days=10)

    step1 = engine.run_daily_step("2024-01-01")
    step2 = engine.run_daily_step("2024-01-02")

    assert step1["skipped"] is False  # AAA alone was enough to make progress
    assert step2["open_positions"] == ["AAA"]
    assert "BBB" not in broker.positions


def test_warns_when_a_corporate_action_falls_inside_an_open_holding(tmp_path, monkeypatch, caplog):
    aaa = _ohlcv([("2024-01-01", 100, 101, 99, 100), ("2024-01-02", 100, 101, 99, 100), ("2024-01-03", 100, 101, 99, 100)])
    monkeypatch.setattr("execution.paper_trading.get_ohlcv", _fake_get_ohlcv({"AAA": aaa}))
    actions = pd.DataFrame(
        {"ex_date": [pd.Timestamp("2024-01-03")], "action_type": ["bonus"], "ratio": [2.0]}
    )
    monkeypatch.setattr("execution.paper_trading.get_corporate_actions", lambda symbol, use_cache=True: actions)

    strategy = FixedSignalStrategy(entry_dates=["2024-01-01"])
    broker = PaperBroker("acct", initial_capital=100_000)
    engine = PaperTradingEngine(strategy, NO_LIMITS_CONFIG, broker, ["AAA"], history_days=10)

    engine.run_daily_step("2024-01-01")
    engine.run_daily_step("2024-01-02")  # entry fills; holding opens before the ex-date

    with caplog.at_level("WARNING", logger="execution.paper_trading"):
        engine.run_daily_step("2024-01-03")  # ex-date falls inside the holding period

    assert any("bonus" in record.message for record in caplog.records)
