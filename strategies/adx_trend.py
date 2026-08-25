import pandas as pd

from indicators.library import adx
from strategies.base import Strategy, empty_signals


class ADXTrendStrategy(Strategy):
    """ADX/DMI trend following: buy once ADX confirms a trending market
    (>= `entry_adx`) with +DI leading -DI, exit the moment -DI crosses back
    above +DI (a trend reversal) regardless of ADX level -- ADX can dip
    during a pullback inside an otherwise-intact trend, so exiting on ADX
    alone would cut winners short.
    """

    default_params = {"lookback": 14, "entry_adx": 25.0}

    def __init__(self, params: dict | None = None):
        super().__init__({**self.default_params, **(params or {})})

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = empty_signals(df.index)
        d = adx(df["HIGH"], df["LOW"], df["CLOSE"], self.params["lookback"])

        signals["entry_long"] = (d["adx"] >= self.params["entry_adx"]) & (d["plus_di"] > d["minus_di"])
        signals["exit_long"] = d["minus_di"] > d["plus_di"]
        return signals
