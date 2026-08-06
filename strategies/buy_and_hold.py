import pandas as pd

from strategies.base import Strategy, empty_signals


class BuyAndHoldStrategy(Strategy):
    """Enter on the first bar, never signal an exit — the engine's
    end-of-data close-out does the rest. Used as the benchmark every other
    strategy has to beat, after the same costs.
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = empty_signals(df.index)
        if len(df) > 0:
            signals.loc[df.index[0], "entry_long"] = True
        return signals
