import pandas as pd

from indicators.library import macd
from strategies.base import Strategy, empty_signals


class MACDCrossoverStrategy(Strategy):
    """MACD trend following: hold long while the MACD line sits above its
    signal line (momentum accelerating upward), exit once it drops back
    below. A different momentum read than MomentumStrategy's raw N-day
    rate of change -- MACD reacts to the trend's rate of change itself.
    """

    default_params = {"fast": 12, "slow": 26, "signal": 9}

    def __init__(self, params: dict | None = None):
        super().__init__({**self.default_params, **(params or {})})

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = empty_signals(df.index)
        m = macd(df["CLOSE"], self.params["fast"], self.params["slow"], self.params["signal"])

        signals["entry_long"] = m["macd"] > m["signal"]
        signals["exit_long"] = m["macd"] < m["signal"]
        return signals
