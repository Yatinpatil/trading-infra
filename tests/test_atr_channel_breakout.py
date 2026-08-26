import numpy as np
import pandas as pd

from indicators.library import atr, ema
from strategies.atr_channel_breakout import ATRChannelBreakoutStrategy


def _ohlcv(closes, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    closes = pd.Series(closes, index=dates, dtype="float64")
    return pd.DataFrame(
        {"OPEN": closes, "HIGH": closes, "LOW": closes, "CLOSE": closes, "VOLUME": 1000}
    )


def test_signals_match_ema_atr_band_thresholds():
    rng = np.random.default_rng(13)
    closes = 100 + rng.normal(0, 1, 80).cumsum()
    # force a sharp spike (above the upper band) and a sharp drop (below the EMA)
    closes[40] += 25
    closes[45] -= 40
    df = _ohlcv(closes)

    params = {"ema_lookback": 20, "atr_lookback": 14, "num_atr": 2.0}
    strategy = ATRChannelBreakoutStrategy(params)
    signals = strategy.generate_signals(df)

    centerline = ema(df["CLOSE"], 20)
    band = atr(df["HIGH"], df["LOW"], df["CLOSE"], 14)
    upper = centerline + 2.0 * band
    expected_entry = (df["CLOSE"] > upper).fillna(False)
    expected_exit = (df["CLOSE"] < centerline).fillna(False)

    pd.testing.assert_series_equal(signals["entry_long"], expected_entry, check_names=False)
    pd.testing.assert_series_equal(signals["exit_long"], expected_exit, check_names=False)
    assert signals["entry_long"].any()
    assert signals["exit_long"].any()


def test_no_signal_columns_for_short():
    df = _ohlcv([100, 101, 99, 98, 102])
    strategy = ATRChannelBreakoutStrategy()
    signals = strategy.generate_signals(df)
    assert not signals["entry_short"].any()
    assert not signals["exit_short"].any()


def test_default_params():
    strategy = ATRChannelBreakoutStrategy()
    assert strategy.params == {"ema_lookback": 20, "atr_lookback": 14, "num_atr": 2.0}
