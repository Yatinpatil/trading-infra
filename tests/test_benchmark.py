import numpy as np
import pandas as pd

from strategies.base import Strategy, empty_signals
from strategies.buy_and_hold import BuyAndHoldStrategy
from validation.benchmark import compare_to_buy_and_hold

NO_COST_CONFIG = {"costs": {}, "risk": {"position_size_pct": 1.0}, "params": {}}


class DelayedEntryStrategy(Strategy):
    """Enters on a fixed day (avoiding an early crash buy-and-hold has to
    sit through), never exits. Used to construct a case where the strategy
    should clearly beat buy-and-hold.
    """

    def __init__(self, entry_bar_index: int):
        super().__init__({})
        self.entry_bar_index = entry_bar_index

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = empty_signals(df.index)
        signals.iloc[self.entry_bar_index, signals.columns.get_loc("entry_long")] = True
        return signals


def _crash_then_recover_ohlcv():
    # days 0-9: crash from 100 -> 50; days 10-59: steady climb 50 -> 150
    crash = np.linspace(100, 50, 10)
    recover = np.linspace(50, 150, 50)
    close = np.concatenate([crash, recover])
    dates = pd.date_range("2024-01-01", periods=len(close), freq="D")
    close = pd.Series(close, index=dates)
    return pd.DataFrame(
        {"OPEN": close, "HIGH": close + 1, "LOW": close - 1, "CLOSE": close, "VOLUME": 1000}
    )


def test_strategy_that_avoids_crash_beats_buy_and_hold():
    df = _crash_then_recover_ohlcv()
    strategy = DelayedEntryStrategy(entry_bar_index=10)

    comparison = compare_to_buy_and_hold(strategy, df, NO_COST_CONFIG, initial_capital=100_000)

    assert comparison["beats_benchmark_cagr"]
    assert comparison["edge_cagr"] > 0
    assert comparison["strategy"]["total_return"] > comparison["benchmark"]["total_return"]


def test_buy_and_hold_against_itself_has_zero_edge():
    df = _crash_then_recover_ohlcv()

    comparison = compare_to_buy_and_hold(BuyAndHoldStrategy(), df, NO_COST_CONFIG, initial_capital=100_000)

    assert comparison["edge_cagr"] == 0
    assert comparison["edge_sharpe"] == 0
    assert not comparison["beats_benchmark_cagr"]  # equal, not strictly greater
