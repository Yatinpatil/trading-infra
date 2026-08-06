import numpy as np
import pandas as pd
import pytest

from indicators.library import (
    adx,
    atr,
    bollinger_bands,
    macd,
    rate_of_change,
    rolling_correlation,
    rolling_high,
    rolling_low,
    rsi,
    true_range,
    zscore,
)


def _series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


def test_zscore_matches_manual_calculation():
    s = _series([1, 2, 3, 4, 5, 100])
    z = zscore(s, lookback=5)

    window = s.iloc[1:6]
    expected = (s.iloc[5] - window.mean()) / window.std()
    assert z.iloc[5] == pytest.approx(expected)


def test_bollinger_bands_bracket_mid_by_num_std():
    s = _series([10, 12, 11, 13, 12, 14, 13])
    bands = bollinger_bands(s, lookback=5, num_std=2.0)

    row = bands.iloc[-1]
    window = s.iloc[-5:]
    expected_mid = window.mean()
    expected_std = window.std()

    assert row["mid"] == pytest.approx(expected_mid)
    assert row["upper"] == pytest.approx(expected_mid + 2 * expected_std)
    assert row["lower"] == pytest.approx(expected_mid - 2 * expected_std)


def test_rsi_is_100_for_all_gains():
    s = _series(list(range(1, 20)))  # strictly increasing, no losses
    r = rsi(s, lookback=14)
    assert r.iloc[-1] == pytest.approx(100.0)


def test_rsi_is_0_for_all_losses():
    s = _series(list(range(20, 1, -1)))  # strictly decreasing, no gains
    r = rsi(s, lookback=14)
    assert r.iloc[-1] == pytest.approx(0.0)


def test_rsi_is_bounded():
    rng = np.random.default_rng(0)
    s = _series(100 + rng.normal(0, 1, 100).cumsum())
    r = rsi(s, lookback=14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_true_range_uses_prev_close_when_gap():
    high = _series([10, 20])
    low = _series([9, 18])
    close = _series([9.5, 19])
    tr = true_range(high, low, close)
    # day 2: high-low=2, |high-prevclose|=10.5, |low-prevclose|=8.5 -> max is 10.5
    assert tr.iloc[1] == pytest.approx(10.5)


def test_atr_is_positive_and_bounded_by_max_true_range():
    rng = np.random.default_rng(1)
    close = _series(100 + rng.normal(0, 1, 50).cumsum())
    high = close + rng.uniform(0.5, 1.5, 50)
    low = close - rng.uniform(0.5, 1.5, 50)

    a = atr(high, low, close, lookback=14).dropna()
    tr = true_range(high, low, close).dropna()
    assert (a > 0).all()
    assert a.max() <= tr.max() + 1e-9


def test_macd_hist_equals_macd_minus_signal():
    rng = np.random.default_rng(2)
    s = _series(100 + rng.normal(0, 1, 60).cumsum())
    result = macd(s)
    pd.testing.assert_series_equal(
        result["hist"], result["macd"] - result["signal"], check_names=False
    )


def test_adx_and_di_bounded_0_100():
    rng = np.random.default_rng(3)
    close = _series(100 + rng.normal(0, 1, 60).cumsum())
    high = close + rng.uniform(0.5, 1.5, 60)
    low = close - rng.uniform(0.5, 1.5, 60)

    result = adx(high, low, close, lookback=14).dropna()
    for col in ["adx", "plus_di", "minus_di"]:
        assert (result[col] >= 0).all()
        assert (result[col] <= 100).all()


def test_rolling_correlation_perfectly_correlated_series():
    a = _series([1, 2, 3, 4, 5, 6, 7])
    b = a * 2 + 1  # perfectly linearly correlated
    corr = rolling_correlation(a, b, lookback=5)
    assert corr.iloc[-1] == pytest.approx(1.0)


def test_rolling_correlation_perfectly_anticorrelated_series():
    a = _series([1, 2, 3, 4, 5, 6, 7])
    b = -a
    corr = rolling_correlation(a, b, lookback=5)
    assert corr.iloc[-1] == pytest.approx(-1.0)


def test_rate_of_change():
    s = _series([100, 105, 110, 121])
    roc = rate_of_change(s, lookback=3)
    assert roc.iloc[3] == pytest.approx((121 - 100) / 100)


def test_rolling_high_excludes_current_bar():
    s = _series([10, 50, 20, 15])
    rh = rolling_high(s, lookback=3)
    # at index 3 (value 15), window is prior 3 bars [10,50,20] -> max 50, current bar excluded
    assert rh.iloc[3] == 50


def test_rolling_low_excludes_current_bar():
    s = _series([10, 5, 20, 15])
    rl = rolling_low(s, lookback=3)
    assert rl.iloc[3] == 5
