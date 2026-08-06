import numpy as np
import pandas as pd

from indicators.library import rate_of_change, zscore
from strategies.breakout import BreakoutStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy


def _ohlcv(closes, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    closes = pd.Series(closes, index=dates, dtype="float64")
    return pd.DataFrame(
        {"OPEN": closes, "HIGH": closes, "LOW": closes, "CLOSE": closes, "VOLUME": 1000}
    )


def test_mean_reversion_signals_match_zscore_thresholds():
    rng = np.random.default_rng(42)
    closes = 100 + rng.normal(0, 1, 80).cumsum()
    # force a sharp dip and a sharp recovery so both thresholds get crossed
    closes[40] -= 30
    closes[45] += 30
    df = _ohlcv(closes)

    strategy = MeanReversionStrategy({"lookback": 10, "entry_zscore": -2.0, "exit_zscore": 0.0})
    signals = strategy.generate_signals(df)

    z = zscore(df["CLOSE"], 10)
    expected_entry = (z <= -2.0).fillna(False)
    expected_exit = (z >= 0.0).fillna(False)

    pd.testing.assert_series_equal(signals["entry_long"], expected_entry, check_names=False)
    pd.testing.assert_series_equal(signals["exit_long"], expected_exit, check_names=False)
    assert signals["entry_long"].any()
    assert signals["exit_long"].any()


def test_mean_reversion_no_signal_columns_for_short():
    df = _ohlcv([100, 101, 99, 98, 102])
    strategy = MeanReversionStrategy()
    signals = strategy.generate_signals(df)
    assert not signals["entry_short"].any()
    assert not signals["exit_short"].any()


def test_momentum_signals_match_roc_thresholds():
    rng = np.random.default_rng(7)
    closes = 100 + rng.normal(0.3, 1, 100).cumsum()
    df = _ohlcv(closes)

    strategy = MomentumStrategy({"lookback": 20, "entry_threshold": 0.05, "exit_threshold": 0.0})
    signals = strategy.generate_signals(df)

    roc = rate_of_change(df["CLOSE"], 20)
    expected_entry = (roc >= 0.05).fillna(False)
    expected_exit = (roc <= 0.0).fillna(False)

    pd.testing.assert_series_equal(signals["entry_long"], expected_entry, check_names=False)
    pd.testing.assert_series_equal(signals["exit_long"], expected_exit, check_names=False)


def test_breakout_entry_fires_exactly_on_new_high():
    # lookback=3: entry_level at index i is max(close[i-3:i]), excluding today
    closes = [10, 11, 9, 8, 20, 12, 11]
    df = _ohlcv(closes)
    strategy = BreakoutStrategy({"entry_lookback": 3, "exit_lookback": 3})

    signals = strategy.generate_signals(df)

    # index 4 (value 20): prior 3 closes = [11,9,8], max=11 -> 20 > 11 -> breakout
    assert signals["entry_long"].iloc[4]
    # index 1 (value 11): prior 3 closes only [10] (insufficient window) -> NaN comparison -> False
    assert not signals["entry_long"].iloc[1]
    # index 5 (value 12): prior 3 closes = [9,8,20], max=20 -> 12 > 20 is False
    assert not signals["entry_long"].iloc[5]


def test_breakout_exit_fires_exactly_on_new_low():
    closes = [10, 11, 9, 8, 20, 12, 5]
    df = _ohlcv(closes)
    strategy = BreakoutStrategy({"entry_lookback": 3, "exit_lookback": 3})

    signals = strategy.generate_signals(df)

    # index 6 (value 5): prior 3 closes = [8,20,12], min=8 -> 5 < 8 -> exit signal
    assert signals["exit_long"].iloc[6]
    # index 4 (value 20): prior 3 closes = [11,9,8], min=8 -> 20 < 8 is False
    assert not signals["exit_long"].iloc[4]


def test_strategy_with_params_overrides_only_given_keys():
    strategy = MeanReversionStrategy({"lookback": 10, "entry_zscore": -2.0, "exit_zscore": 0.0})
    variant = strategy.with_params(entry_zscore=-3.0)

    assert variant.params["entry_zscore"] == -3.0
    assert variant.params["lookback"] == 10  # untouched
    assert strategy.params["entry_zscore"] == -2.0  # original unmodified
