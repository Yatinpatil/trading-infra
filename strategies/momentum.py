import pandas as pd

from indicators.library import rate_of_change
from strategies.base import Strategy, empty_signals


class MomentumStrategy(Strategy):
    """Time-series momentum: buy when trailing `lookback`-bar return exceeds
    `entry_threshold`, exit once it falls back to `exit_threshold`.
    """

    default_params = {"lookback": 60, "entry_threshold": 0.10, "exit_threshold": 0.0}

    def __init__(self, params: dict | None = None):
        super().__init__({**self.default_params, **(params or {})})

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = empty_signals(df.index)
        roc = rate_of_change(df["CLOSE"], self.params["lookback"])

        signals["entry_long"] = roc >= self.params["entry_threshold"]
        signals["exit_long"] = roc <= self.params["exit_threshold"]
        return signals
