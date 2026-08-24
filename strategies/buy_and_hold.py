import pandas as pd

from strategies.base import Strategy, empty_signals


class BuyAndHoldStrategy(Strategy):
    """Enter as soon as not already holding, never signal an exit — the
    engine's end-of-data close-out does the rest. Used as the benchmark
    every other strategy has to beat, after the same costs.

    Signaling entry_long on every bar (not just the first) looks redundant
    for a one-shot backtest — the engine only acts on it while
    `symbol not in positions`, so it still enters exactly once, on day one,
    same as before. It matters for paper trading: PaperTradingEngine feeds
    a rolling window ending "today", so "the first bar of df" is ~history_days
    in the past, never today, and a signal fired only there would never
    actually be seen by an account that wasn't already holding — it would
    sit in cash forever. Signaling every bar means today's row always
    carries the same entry_long=True the first bar would have, however far
    back that first bar has scrolled.
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = empty_signals(df.index)
        signals["entry_long"] = True
        return signals
