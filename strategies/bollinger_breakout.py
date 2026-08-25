import pandas as pd

from indicators.library import bollinger_bands
from strategies.base import Strategy, empty_signals


class BollingerBreakoutStrategy(Strategy):
    """Bollinger Band breakout: buy when close breaks above the upper band,
    exit once it falls back below the middle band. A volatility-adaptive
    breakout level (the bands widen and narrow with recent volatility),
    unlike BreakoutStrategy's fixed Donchian lookback high.
    """

    default_params = {"lookback": 20, "num_std": 2.0}

    def __init__(self, params: dict | None = None):
        super().__init__({**self.default_params, **(params or {})})

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = empty_signals(df.index)
        bands = bollinger_bands(df["CLOSE"], self.params["lookback"], self.params["num_std"])

        signals["entry_long"] = df["CLOSE"] > bands["upper"]
        signals["exit_long"] = df["CLOSE"] < bands["mid"]
        return signals
