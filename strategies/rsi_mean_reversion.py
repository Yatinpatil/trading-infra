import pandas as pd

from indicators.library import rsi
from strategies.base import Strategy, empty_signals


class RSIMeanReversionStrategy(Strategy):
    """RSI mean reversion: buy when RSI drops to/below `entry_rsi` (oversold),
    exit once it recovers to `exit_rsi`. A different mean-reversion signal
    than MeanReversionStrategy's rolling z-score -- RSI is bounded [0, 100]
    and reacts to the size and persistence of recent moves rather than
    distance from a rolling mean.
    """

    default_params = {"lookback": 14, "entry_rsi": 30.0, "exit_rsi": 50.0}

    def __init__(self, params: dict | None = None):
        super().__init__({**self.default_params, **(params or {})})

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = empty_signals(df.index)
        rsi_value = rsi(df["CLOSE"], self.params["lookback"])

        signals["entry_long"] = rsi_value <= self.params["entry_rsi"]
        signals["exit_long"] = rsi_value >= self.params["exit_rsi"]
        return signals
