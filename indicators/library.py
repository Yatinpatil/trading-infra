"""Shared technical indicators.

Every function here is causal: the value at row t is computed only from data
at or before t (rolling/exponential windows looking backward). That's what
makes them safe to feed into strategies without lookahead bias — but it's
the *strategy's* job to act on the value at t only for a decision that fills
at t+1 or later; these functions don't enforce that shift themselves.
"""
import numpy as np
import pandas as pd


def zscore(series: pd.Series, lookback: int) -> pd.Series:
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std()
    return (series - mean) / std


def ema(series: pd.Series, lookback: int) -> pd.Series:
    return series.ewm(span=lookback, adjust=False).mean()


def bollinger_bands(series: pd.Series, lookback: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = series.rolling(lookback).mean()
    std = series.rolling(lookback).std()
    return pd.DataFrame(
        {
            "mid": mid,
            "upper": mid + num_std * std,
            "lower": mid - num_std * std,
        }
    )


def rsi(series: pd.Series, lookback: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / lookback, min_periods=lookback, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / lookback, min_periods=lookback, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift()
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 14) -> pd.Series:
    """Wilder's ATR."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / lookback, min_periods=lookback, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": macd_line - signal_line}
    )


def adx(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 14) -> pd.DataFrame:
    """Wilder's ADX, plus the +DI/-DI it's derived from."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr = true_range(high, low, close)
    atr_smoothed = tr.ewm(alpha=1 / lookback, min_periods=lookback, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / lookback, min_periods=lookback, adjust=False).mean() / atr_smoothed
    minus_di = 100 * minus_dm.ewm(alpha=1 / lookback, min_periods=lookback, adjust=False).mean() / atr_smoothed

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_line = dx.ewm(alpha=1 / lookback, min_periods=lookback, adjust=False).mean()

    return pd.DataFrame({"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di})


def rolling_correlation(series_a: pd.Series, series_b: pd.Series, lookback: int = 20) -> pd.Series:
    return series_a.rolling(lookback).corr(series_b)


def rate_of_change(series: pd.Series, lookback: int = 10) -> pd.Series:
    return series.pct_change(lookback)


def rolling_high(series: pd.Series, lookback: int) -> pd.Series:
    """Highest value over the trailing `lookback` bars, excluding the current bar."""
    return series.shift(1).rolling(lookback).max()


def rolling_low(series: pd.Series, lookback: int) -> pd.Series:
    """Lowest value over the trailing `lookback` bars, excluding the current bar."""
    return series.shift(1).rolling(lookback).min()
