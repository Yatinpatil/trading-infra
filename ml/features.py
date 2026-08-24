"""Feature engineering and labeling for ML strategies.

Features are built entirely from indicators/library.py, so they inherit its
causal guarantee (row t uses only data through t) — the same property every
rule-based strategy in this repo relies on. Every feature is also
scale-invariant (a ratio, a z-score, a 0-100 oscillator) rather than a raw
price level, because MLStrategy pools rows across many differently-priced
symbols into one training set; a raw MACD or ATR value from a Rs 50 stock
and a Rs 5,000 stock aren't comparable, but their z-scores and percentages
are.

Labels are the one place this module is deliberately NOT causal — a
supervised model needs the future outcome to learn from. That forward
window is exactly why train and test data must never sit adjacent in time
without a gap: see `embargo_train_test_split` and its docstring for why.
"""
import numpy as np
import pandas as pd

from indicators.library import adx, atr, bollinger_bands, macd, rate_of_change, rolling_high, rolling_low, rsi, zscore

FEATURE_COLUMNS = [
    "zscore_20",
    "rsi_14",
    "macd_hist_pct",
    "adx_14",
    "di_diff",
    "atr_pct",
    "roc_10",
    "bb_percent_b",
    "donchian_position",
    "volume_zscore",
]

_EPSILON = 1e-9


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal, scale-invariant features aligned to df.index."""
    close, high, low, volume = df["CLOSE"], df["HIGH"], df["LOW"], df["VOLUME"]

    macd_frame = macd(close)
    adx_frame = adx(high, low, close)
    bb = bollinger_bands(close)
    high_20, low_20 = rolling_high(close, 20), rolling_low(close, 20)

    features = pd.DataFrame(index=df.index)
    features["zscore_20"] = zscore(close, 20)
    features["rsi_14"] = rsi(close, 14) / 100.0
    features["macd_hist_pct"] = macd_frame["hist"] / close
    features["adx_14"] = adx_frame["adx"] / 100.0
    features["di_diff"] = (adx_frame["plus_di"] - adx_frame["minus_di"]) / 100.0
    features["atr_pct"] = atr(high, low, close, 14) / close
    features["roc_10"] = rate_of_change(close, 10)
    features["bb_percent_b"] = (close - bb["lower"]) / (bb["upper"] - bb["lower"] + _EPSILON)
    features["donchian_position"] = (close - low_20) / (high_20 - low_20 + _EPSILON)
    features["volume_zscore"] = zscore(volume, 20)

    return features


def build_labels(df: pd.DataFrame, horizon: int, return_threshold: float = 0.0) -> pd.Series:
    """1 if the forward `horizon`-bar return exceeds `return_threshold`, else
    0. NaN for the last `horizon` rows, where no future bar exists yet.
    """
    forward_return = df["CLOSE"].shift(-horizon) / df["CLOSE"] - 1.0
    label = (forward_return > return_threshold).astype(float)
    label[forward_return.isna()] = np.nan
    return label


def make_training_frame(df: pd.DataFrame, horizon: int, return_threshold: float = 0.0) -> pd.DataFrame:
    """Features + label, aligned, with unlabeled rows (the last `horizon`,
    where the forward return isn't known yet) dropped. Feature warmup NaNs
    are left in — the gradient-boosted model handles missing values
    natively rather than needing them imputed or dropped.
    """
    features = build_features(df)
    label = build_labels(df, horizon, return_threshold)
    frame = features.copy()
    frame["label"] = label
    return frame.dropna(subset=["label"])


def embargo_train_test_split(
    df: pd.DataFrame, train_frac: float = 0.7, horizon: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Like validation.walk_forward.train_test_split, but purges the
    `horizon` rows immediately before the split point from the training
    side.

    A label at row t is built from the close `horizon` bars *later* — so
    without the purge, the last few rows of "train" would carry labels
    computed from data that falls inside the test window, leaking test-period
    information into training through the label rather than the features.
    """
    split_idx = int(len(df) * train_frac)
    train_end = max(0, split_idx - horizon)
    return df.iloc[:train_end], df.iloc[split_idx:]
