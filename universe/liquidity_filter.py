"""Liquidity filtering — drop symbols too thinly traded to fill realistically."""
import pandas as pd


def average_daily_value(ohlcv: pd.DataFrame, lookback_days: int = 20) -> float:
    """Mean traded value (CLOSE * VOLUME) over the trailing `lookback_days`."""
    if ohlcv.empty:
        return 0.0
    window = ohlcv.tail(lookback_days)
    return float((window["CLOSE"] * window["VOLUME"]).mean())


def filter_by_liquidity(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    min_avg_daily_value: float,
    lookback_days: int = 20,
) -> list[str]:
    """Symbols whose trailing average daily traded value (INR) meets `min_avg_daily_value`.

    `ohlcv_by_symbol` values should already be sliced to end at the as-of date
    — this function only looks at the tail of what it's given, it doesn't
    know "today".
    """
    return sorted(
        symbol
        for symbol, df in ohlcv_by_symbol.items()
        if average_daily_value(df, lookback_days) >= min_avg_daily_value
    )
