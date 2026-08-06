import pandas as pd

from indicators.library import zscore
from strategies.base import Strategy, empty_signals


class MeanReversionStrategy(Strategy):
    """Z-score mean reversion: buy when price is `entry_zscore` std-devs below
    its rolling mean, exit once it reverts back to `exit_zscore`.
    """

    default_params = {"lookback": 20, "entry_zscore": -2.0, "exit_zscore": 0.0}

    def __init__(self, params: dict | None = None):
        super().__init__({**self.default_params, **(params or {})})

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = empty_signals(df.index)
        z = zscore(df["CLOSE"], self.params["lookback"])

        signals["entry_long"] = z <= self.params["entry_zscore"]
        signals["exit_long"] = z >= self.params["exit_zscore"]
        return signals
