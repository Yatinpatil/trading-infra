import pandas as pd
import pytest

from engine.single_stock import run_single_backtest
from strategies.base import Strategy, empty_signals


class FixedSignalStrategy(Strategy):
    """Test double: entry/exit fire on caller-specified dates, bypassing any
    indicator logic so engine timing can be asserted precisely.
    """

    def __init__(self, entry_dates=(), exit_dates=()):
        super().__init__({})
        self.entry_dates = set(entry_dates)
        self.exit_dates = set(exit_dates)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = empty_signals(df.index)
        signals.loc[signals.index.isin(self.entry_dates), "entry_long"] = True
        signals.loc[signals.index.isin(self.exit_dates), "exit_long"] = True
        return signals


def _ohlcv(rows):
    """rows: list of (date_str, open, high, low, close)."""
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


NO_COST_CONFIG = {"costs": {}, "risk": {"position_size_pct": 1.0}, "params": {}}


def test_entry_fills_at_next_bar_open_not_signal_bar_close():
    df = _ohlcv(
        [
            ("2024-01-01", 100, 101, 99, 100),
            ("2024-01-02", 105, 106, 104, 105),  # signal fires here (close=105)
            ("2024-01-03", 110, 111, 109, 110),  # fill should happen at this OPEN=110
            ("2024-01-04", 112, 113, 111, 112),
        ]
    )
    strategy = FixedSignalStrategy(entry_dates=[pd.Timestamp("2024-01-02")])

    result = run_single_backtest(strategy, df, NO_COST_CONFIG, initial_capital=10_000)

    assert len(result.trades) == 1  # position closed at end_of_data
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == 110.0
    assert trade["entry_date"] == pd.Timestamp("2024-01-03")


def test_exit_fills_at_next_bar_open_not_signal_bar_close():
    df = _ohlcv(
        [
            ("2024-01-01", 100, 101, 99, 100),  # entry signal
            ("2024-01-02", 100, 101, 99, 100),  # fill entry at open=100
            ("2024-01-03", 100, 101, 99, 100),  # exit signal fires (close=100)
            ("2024-01-04", 120, 121, 119, 120),  # fill exit at this OPEN=120
        ]
    )
    strategy = FixedSignalStrategy(
        entry_dates=[pd.Timestamp("2024-01-01")], exit_dates=[pd.Timestamp("2024-01-03")]
    )

    result = run_single_backtest(strategy, df, NO_COST_CONFIG, initial_capital=10_000)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_price"] == 120.0
    assert trade["exit_date"] == pd.Timestamp("2024-01-04")
    assert trade["exit_reason"] == "signal"


def test_open_position_closed_at_end_of_data():
    df = _ohlcv(
        [
            ("2024-01-01", 100, 101, 99, 100),
            ("2024-01-02", 100, 101, 99, 100),
            ("2024-01-03", 105, 106, 104, 103),
        ]
    )
    strategy = FixedSignalStrategy(entry_dates=[pd.Timestamp("2024-01-01")])

    result = run_single_backtest(strategy, df, NO_COST_CONFIG, initial_capital=10_000)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "end_of_data"
    assert trade["exit_price"] == 103.0  # last CLOSE
    assert trade["exit_date"] == pd.Timestamp("2024-01-03")


def test_stop_loss_triggers_intrabar_on_low():
    df = _ohlcv(
        [
            ("2024-01-01", 100, 101, 99, 100),   # entry signal
            ("2024-01-02", 100, 101, 99, 100),   # fill entry at open=100; stop = 100*(1-0.05)=95
            ("2024-01-03", 99, 100, 90, 96),      # low=90 breaches stop -> exit at 95, same bar
            ("2024-01-04", 96, 97, 95, 96),
        ]
    )
    strategy = FixedSignalStrategy(entry_dates=[pd.Timestamp("2024-01-01")])
    config = {"costs": {}, "risk": {"position_size_pct": 1.0}, "params": {"stop_loss_pct": 0.05}}

    result = run_single_backtest(strategy, df, config, initial_capital=10_000)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert trade["exit_price"] == pytest.approx(95.0)
    assert trade["exit_date"] == pd.Timestamp("2024-01-03")


def test_costs_reduce_pnl():
    df = _ohlcv(
        [
            ("2024-01-01", 100, 101, 99, 100),
            ("2024-01-02", 100, 101, 99, 100),
            ("2024-01-03", 100, 101, 99, 100),
            ("2024-01-04", 110, 111, 109, 110),
        ]
    )
    strategy = FixedSignalStrategy(
        entry_dates=[pd.Timestamp("2024-01-01")], exit_dates=[pd.Timestamp("2024-01-03")]
    )
    no_cost = run_single_backtest(strategy, df, NO_COST_CONFIG, initial_capital=10_000)

    costly_config = {"costs": {"brokerage_pct": 0.01}, "risk": {"position_size_pct": 1.0}, "params": {}}
    with_cost = run_single_backtest(strategy, df, costly_config, initial_capital=10_000)

    assert with_cost.trades.iloc[0]["pnl"] < no_cost.trades.iloc[0]["pnl"]
    assert with_cost.final_equity < no_cost.final_equity


def test_equity_curve_covers_full_date_range():
    df = _ohlcv([(f"2024-01-{d:02d}", 100, 101, 99, 100) for d in range(1, 6)])
    strategy = FixedSignalStrategy()  # no signals at all -> flat throughout

    result = run_single_backtest(strategy, df, NO_COST_CONFIG, initial_capital=10_000)

    assert len(result.equity_curve) == 5
    assert (result.equity_curve == 10_000).all()
    assert result.trades.empty


def test_no_reentry_while_position_open():
    df = _ohlcv(
        [
            ("2024-01-01", 100, 101, 99, 100),  # entry signal
            ("2024-01-02", 100, 101, 99, 100),  # filled here
            ("2024-01-03", 100, 101, 99, 100),  # duplicate entry signal while in position -> ignored
            ("2024-01-04", 100, 101, 99, 100),
        ]
    )
    strategy = FixedSignalStrategy(
        entry_dates=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")]
    )

    result = run_single_backtest(strategy, df, NO_COST_CONFIG, initial_capital=10_000)

    assert len(result.trades) == 1  # only one position ever opened
