import numpy as np
import pandas as pd

from indicators.library import rsi
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy


def _ohlcv(closes, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    closes = pd.Series(closes, index=dates, dtype="float64")
    return pd.DataFrame(
        {"OPEN": closes, "HIGH": closes, "LOW": closes, "CLOSE": closes, "VOLUME": 1000}
    )


def test_signals_match_rsi_thresholds():
    rng = np.random.default_rng(3)
    closes = 100 + rng.normal(0, 1, 80).cumsum()
    # force a sharp dip and a sharp recovery so both thresholds get crossed
    closes[40] -= 25
    closes[45] += 25
    df = _ohlcv(closes)

    strategy = RSIMeanReversionStrategy({"lookback": 14, "entry_rsi": 30.0, "exit_rsi": 50.0})
    signals = strategy.generate_signals(df)

    r = rsi(df["CLOSE"], 14)
    expected_entry = (r <= 30.0).fillna(False)
    expected_exit = (r >= 50.0).fillna(False)

    pd.testing.assert_series_equal(signals["entry_long"], expected_entry, check_names=False)
    pd.testing.assert_series_equal(signals["exit_long"], expected_exit, check_names=False)
    assert signals["entry_long"].any()
    assert signals["exit_long"].any()


def test_no_signal_columns_for_short():
    df = _ohlcv([100, 101, 99, 98, 102])
    strategy = RSIMeanReversionStrategy()
    signals = strategy.generate_signals(df)
    assert not signals["entry_short"].any()
    assert not signals["exit_short"].any()


def test_default_params():
    strategy = RSIMeanReversionStrategy()
    assert strategy.params == {"lookback": 14, "entry_rsi": 30.0, "exit_rsi": 50.0}
