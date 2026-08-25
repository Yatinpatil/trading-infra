import numpy as np
import pandas as pd

from indicators.library import bollinger_bands
from strategies.bollinger_breakout import BollingerBreakoutStrategy


def _ohlcv(closes, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    closes = pd.Series(closes, index=dates, dtype="float64")
    return pd.DataFrame(
        {"OPEN": closes, "HIGH": closes, "LOW": closes, "CLOSE": closes, "VOLUME": 1000}
    )


def test_signals_match_band_thresholds():
    rng = np.random.default_rng(11)
    closes = 100 + rng.normal(0, 1, 80).cumsum()
    # force a sharp spike (above the upper band) and a sharp drop (below the mid band)
    closes[40] += 25
    closes[45] -= 40
    df = _ohlcv(closes)

    strategy = BollingerBreakoutStrategy({"lookback": 20, "num_std": 2.0})
    signals = strategy.generate_signals(df)

    bands = bollinger_bands(df["CLOSE"], 20, 2.0)
    expected_entry = (df["CLOSE"] > bands["upper"]).fillna(False)
    expected_exit = (df["CLOSE"] < bands["mid"]).fillna(False)

    pd.testing.assert_series_equal(signals["entry_long"], expected_entry, check_names=False)
    pd.testing.assert_series_equal(signals["exit_long"], expected_exit, check_names=False)
    assert signals["entry_long"].any()
    assert signals["exit_long"].any()


def test_no_signal_columns_for_short():
    df = _ohlcv([100, 101, 99, 98, 102])
    strategy = BollingerBreakoutStrategy()
    signals = strategy.generate_signals(df)
    assert not signals["entry_short"].any()
    assert not signals["exit_short"].any()


def test_default_params():
    strategy = BollingerBreakoutStrategy()
    assert strategy.params == {"lookback": 20, "num_std": 2.0}
