import pandas as pd

from indicators.library import atr, ema
from strategies.base import Strategy, empty_signals


class ATRChannelBreakoutStrategy(Strategy):
    """ATR/Keltner-style channel breakout: buy when close breaks above an
    EMA + `num_atr` * ATR band, exit once it falls back below the EMA
    centerline. A volatility-adaptive breakout built from true range
    (reacts to gaps) rather than BollingerBreakoutStrategy's close-to-close
    standard deviation.
    """

    default_params = {"ema_lookback": 20, "atr_lookback": 14, "num_atr": 2.0}

    def __init__(self, params: dict | None = None):
        super().__init__({**self.default_params, **(params or {})})

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = empty_signals(df.index)
        centerline = ema(df["CLOSE"], self.params["ema_lookback"])
        band = atr(df["HIGH"], df["LOW"], df["CLOSE"], self.params["atr_lookback"])
        upper = centerline + self.params["num_atr"] * band

        signals["entry_long"] = df["CLOSE"] > upper
        signals["exit_long"] = df["CLOSE"] < centerline
        return signals
