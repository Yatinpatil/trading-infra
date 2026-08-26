import numpy as np
import pandas as pd

from indicators.library import macd
from strategies.macd_crossover import MACDCrossoverStrategy


def _ohlcv(closes, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    closes = pd.Series(closes, index=dates, dtype="float64")
    return pd.DataFrame(
        {"OPEN": closes, "HIGH": closes, "LOW": closes, "CLOSE": closes, "VOLUME": 1000}
    )


def test_signals_match_macd_signal_line_crossovers():
    rng = np.random.default_rng(5)
    up = 100 + np.abs(rng.normal(0.5, 0.5, 40)).cumsum()
    down = up[-1] - np.abs(rng.normal(0.5, 0.5, 40)).cumsum()
    closes = np.concatenate([up, down])
    df = _ohlcv(closes)

    strategy = MACDCrossoverStrategy({"fast": 12, "slow": 26, "signal": 9})
    signals = strategy.generate_signals(df)

    m = macd(df["CLOSE"], 12, 26, 9)
    expected_entry = (m["macd"] > m["signal"]).fillna(False)
    expected_exit = (m["macd"] < m["signal"]).fillna(False)

    pd.testing.assert_series_equal(signals["entry_long"], expected_entry, check_names=False)
    pd.testing.assert_series_equal(signals["exit_long"], expected_exit, check_names=False)
    assert signals["entry_long"].any()
    assert signals["exit_long"].any()


def test_no_signal_columns_for_short():
    df = _ohlcv([100, 101, 99, 98, 102])
    strategy = MACDCrossoverStrategy()
    signals = strategy.generate_signals(df)
    assert not signals["entry_short"].any()
    assert not signals["exit_short"].any()


def test_default_params():
    strategy = MACDCrossoverStrategy()
    assert strategy.params == {"fast": 12, "slow": 26, "signal": 9}
