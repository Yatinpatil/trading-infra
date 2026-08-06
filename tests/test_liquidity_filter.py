import pandas as pd

from universe.liquidity_filter import average_daily_value, filter_by_liquidity


def _df(close, volume, n=20):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"CLOSE": [close] * n, "VOLUME": [volume] * n}, index=dates)


def test_average_daily_value():
    df = _df(close=100.0, volume=1000)
    assert average_daily_value(df) == 100_000.0


def test_average_daily_value_empty_df():
    assert average_daily_value(pd.DataFrame()) == 0.0


def test_filter_by_liquidity_keeps_only_liquid_symbols():
    data = {
        "LIQUID": _df(close=1000.0, volume=10_000),   # 10,000,000 avg value
        "ILLIQUID": _df(close=10.0, volume=100),        # 1,000 avg value
    }

    result = filter_by_liquidity(data, min_avg_daily_value=1_000_000)

    assert result == ["LIQUID"]
