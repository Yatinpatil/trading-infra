"""Abstract strategy interface.

A Strategy turns an OHLCV DataFrame into signals — nothing more. It never
decides position size, never sees costs, never knows what fills. That
separation (Section 6 of the project plan) is what lets the same strategy
code run in a single-stock backtest, a portfolio backtest, and eventually
paper/live trading without being rewritten.
"""
from abc import ABC, abstractmethod

import pandas as pd

SIGNAL_COLUMNS = ["entry_long", "exit_long", "entry_short", "exit_short"]


class Strategy(ABC):
    def __init__(self, params: dict | None = None):
        self.params = params or {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame aligned to df.index with boolean columns from
        SIGNAL_COLUMNS (any missing column is treated as all-False).

        The value at row t may only depend on df.loc[:t] — never on future
        rows. This method decides *whether* a signal fires on day t's
        close-of-data; the execution engine decides *when* it fills (never
        the same bar's close).
        """
        raise NotImplementedError

    def with_params(self, **overrides) -> "Strategy":
        return type(self)({**self.params, **overrides})


def empty_signals(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(False, index=index, columns=SIGNAL_COLUMNS)
