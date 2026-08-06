import pandas as pd

from indicators.library import rolling_high, rolling_low
from strategies.base import Strategy, empty_signals


class BreakoutStrategy(Strategy):
    """Donchian-channel breakout: buy when close exceeds the highest close of
    the trailing `entry_lookback` bars, exit when it drops below the lowest
    close of the trailing `exit_lookback` bars.
    """

    default_params = {"entry_lookback": 20, "exit_lookback": 10}

    def __init__(self, params: dict | None = None):
        super().__init__({**self.default_params, **(params or {})})

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = empty_signals(df.index)
        close = df["CLOSE"]

        entry_level = rolling_high(close, self.params["entry_lookback"])
        exit_level = rolling_low(close, self.params["exit_lookback"])

        signals["entry_long"] = close > entry_level
        signals["exit_long"] = close < exit_level
        return signals
