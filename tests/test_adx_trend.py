import numpy as np
import pandas as pd

from indicators.library import adx
from strategies.adx_trend import ADXTrendStrategy


def _ohlcv(closes, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    closes = pd.Series(closes, index=dates, dtype="float64")
    return pd.DataFrame(
        {"OPEN": closes, "HIGH": closes, "LOW": closes, "CLOSE": closes, "VOLUME": 1000}
    )


def test_signals_match_adx_di_thresholds():
    rng = np.random.default_rng(21)
    # a sustained uptrend long enough to push ADX above the entry threshold,
    # followed by a sustained downtrend to flip -DI back above +DI
    up = 100 + np.abs(rng.normal(0.6, 0.3, 60)).cumsum()
    down = up[-1] - np.abs(rng.normal(0.6, 0.3, 60)).cumsum()
    closes = np.concatenate([up, down])
    df = _ohlcv(closes)

    strategy = ADXTrendStrategy({"lookback": 14, "entry_adx": 25.0})
    signals = strategy.generate_signals(df)

    d = adx(df["HIGH"], df["LOW"], df["CLOSE"], 14)
    expected_entry = ((d["adx"] >= 25.0) & (d["plus_di"] > d["minus_di"])).fillna(False)
    expected_exit = (d["minus_di"] > d["plus_di"]).fillna(False)

    pd.testing.assert_series_equal(signals["entry_long"], expected_entry, check_names=False)
    pd.testing.assert_series_equal(signals["exit_long"], expected_exit, check_names=False)
    assert signals["entry_long"].any()
    assert signals["exit_long"].any()


def test_no_signal_columns_for_short():
    df = _ohlcv([100, 101, 99, 98, 102])
    strategy = ADXTrendStrategy()
    signals = strategy.generate_signals(df)
    assert not signals["entry_short"].any()
    assert not signals["exit_short"].any()


def test_default_params():
    strategy = ADXTrendStrategy()
    assert strategy.params == {"lookback": 14, "entry_adx": 25.0}
