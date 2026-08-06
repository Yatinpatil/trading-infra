import pandas as pd

from engine.portfolio import run_portfolio_backtest
from strategies.base import Strategy, empty_signals


class FixedSignalStrategy(Strategy):
    """Same entry/exit dates applied to every symbol it's run against —
    enough to test the engine's allocation/risk logic without depending on
    real indicator behavior.
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


def test_combined_equity_curve_reflects_both_symbols():
    aaa = _ohlcv(
        [
            ("2024-01-01", 100, 101, 99, 100),
            ("2024-01-02", 100, 101, 99, 100),
            ("2024-01-03", 120, 121, 119, 120),
        ]
    )
    bbb = _ohlcv(
        [
            ("2024-01-01", 50, 51, 49, 50),
            ("2024-01-02", 50, 51, 49, 50),
            ("2024-01-03", 40, 41, 39, 40),
        ]
    )
    strategy = FixedSignalStrategy(entry_dates=[pd.Timestamp("2024-01-01")])

    result = run_portfolio_backtest(
        strategy, {"AAA": aaa, "BBB": bbb}, NO_LIMITS_CONFIG, initial_capital=100_000
    )

    assert len(result.trades) == 2  # both closed at end_of_data
    assert set(result.trades["symbol"]) == {"AAA", "BBB"}
    aaa_trade = result.trades[result.trades["symbol"] == "AAA"].iloc[0]
    bbb_trade = result.trades[result.trades["symbol"] == "BBB"].iloc[0]
    assert aaa_trade["pnl"] > 0   # AAA rallied
    assert bbb_trade["pnl"] < 0   # BBB dropped
    # combined equity = cash back from both trades, more than just one leg
    assert result.final_equity == 100_000 + aaa_trade["pnl"] + bbb_trade["pnl"]


def test_max_concurrent_positions_limits_simultaneous_entries():
    aaa = _ohlcv([("2024-01-01", 100, 101, 99, 100), ("2024-01-02", 100, 101, 99, 100), ("2024-01-03", 100, 101, 99, 100)])
    bbb = _ohlcv([("2024-01-01", 50, 51, 49, 50), ("2024-01-02", 50, 51, 49, 50), ("2024-01-03", 50, 51, 49, 50)])
    strategy = FixedSignalStrategy(entry_dates=[pd.Timestamp("2024-01-01")])
    config = {"costs": {}, "risk": {"position_size_pct": 0.1, "max_concurrent_positions": 1}, "params": {}}

    result = run_portfolio_backtest(strategy, {"AAA": aaa, "BBB": bbb}, config, initial_capital=100_000)

    # only AAA (alphabetically first) should ever have been opened
    assert set(result.trades["symbol"]) == {"AAA"}


def test_stop_loss_is_independent_per_symbol():
    aaa = _ohlcv(
        [
            ("2024-01-01", 100, 101, 99, 100),
            ("2024-01-02", 100, 101, 99, 100),   # entry fill; stop = 100*0.95=95
            ("2024-01-03", 99, 100, 90, 96),      # low breaches stop -> exit @ 95
            ("2024-01-04", 96, 97, 95, 96),
        ]
    )
    bbb = _ohlcv(
        [
            ("2024-01-01", 50, 51, 49, 50),
            ("2024-01-02", 50, 51, 49, 50),   # entry fill; stop = 47.5
            ("2024-01-03", 50, 51, 49, 50),   # no breach
            ("2024-01-04", 55, 56, 54, 55),
        ]
    )
    strategy = FixedSignalStrategy(entry_dates=[pd.Timestamp("2024-01-01")])
    config = {"costs": {}, "risk": {"position_size_pct": 0.1}, "params": {"stop_loss_pct": 0.05}}

    result = run_portfolio_backtest(strategy, {"AAA": aaa, "BBB": bbb}, config, initial_capital=100_000)

    aaa_trade = result.trades[result.trades["symbol"] == "AAA"].iloc[0]
    bbb_trade = result.trades[result.trades["symbol"] == "BBB"].iloc[0]
    assert aaa_trade["exit_reason"] == "stop_loss"
    assert aaa_trade["exit_price"] == 95.0
    assert bbb_trade["exit_reason"] == "end_of_data"


def test_sector_exposure_limit_blocks_second_entry_in_same_sector():
    aaa = _ohlcv([("2024-01-01", 100, 101, 99, 100), ("2024-01-02", 100, 101, 99, 100), ("2024-01-03", 100, 101, 99, 100)])
    bbb = _ohlcv([("2024-01-01", 100, 101, 99, 100), ("2024-01-02", 100, 101, 99, 100), ("2024-01-03", 100, 101, 99, 100)])
    strategy = FixedSignalStrategy(entry_dates=[pd.Timestamp("2024-01-01")])
    config = {
        "costs": {},
        "risk": {"position_size_pct": 0.1, "max_exposure_per_sector_pct": 0.15},
        "params": {},
    }

    result = run_portfolio_backtest(
        strategy, {"AAA": aaa, "BBB": bbb}, config, initial_capital=100_000,
        sector_map={"AAA": "TECH", "BBB": "TECH"},
    )

    # AAA takes 10% of equity; adding BBB would push sector exposure to ~20% > 15% cap
    assert set(result.trades["symbol"]) == {"AAA"}


def test_correlation_limit_blocks_highly_correlated_second_entry():
    # AAA and BBB move in lockstep for the whole warmup period
    shared_prices = [100, 102, 101, 103, 102, 104, 103, 105]
    dates = pd.date_range("2024-01-01", periods=len(shared_prices) + 2, freq="D")

    def build(prices):
        rows = [(str(d.date()), p, p + 1, p - 1, p) for d, p in zip(dates, prices)]
        return _ohlcv(rows)

    aaa = build(shared_prices + [106, 107])
    bbb = build(shared_prices + [106, 107])

    entry_date = dates[len(shared_prices)]  # both signal entry on the same, later day
    strategy = FixedSignalStrategy(entry_dates=[entry_date])
    config = {
        "costs": {},
        "risk": {"position_size_pct": 0.1, "max_correlation": 0.8},
        "params": {},
    }

    result = run_portfolio_backtest(strategy, {"AAA": aaa, "BBB": bbb}, config, initial_capital=100_000)

    assert set(result.trades["symbol"]) == {"AAA"}  # BBB rejected as too correlated with AAA
