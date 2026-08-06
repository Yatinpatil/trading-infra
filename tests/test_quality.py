import pandas as pd

from data.quality import check_gaps, check_outliers, check_stale_prices


def _df(dates, closes, volumes=None):
    n = len(dates)
    volumes = volumes or [1000] * n
    return pd.DataFrame(
        {
            "OPEN": closes,
            "HIGH": closes,
            "LOW": closes,
            "CLOSE": closes,
            "VOLUME": volumes,
        },
        index=pd.DatetimeIndex(dates),
    )


def test_check_gaps_flags_wide_gap():
    dates = ["2024-01-01", "2024-01-02", "2024-01-20"]
    df = _df(dates, [10, 11, 12])

    gaps = check_gaps(df, max_gap_days=5)

    assert len(gaps) == 1
    assert gaps.iloc[0]["gap_days"] == 18


def test_check_gaps_no_gaps():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = _df(dates, [10, 11, 12, 13, 14])

    gaps = check_gaps(df, max_gap_days=5)

    assert gaps.empty


def test_check_stale_prices_flags_repeated_run():
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    closes = [10, 11, 12, 12, 12, 12, 12, 13]

    stale = check_stale_prices(_df(dates, closes), min_repeat_days=5)

    assert len(stale) == 1
    assert stale.iloc[0]["run_length"] == 5
    assert stale.iloc[0]["price"] == 12


def test_check_outliers_flags_large_move():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    closes = [100, 101, 50, 51]  # ~50% drop on day 3

    outliers = check_outliers(_df(dates, closes), max_daily_move_pct=0.20)

    assert len(outliers) == 1
    assert outliers.iloc[0]["close"] == 50


def test_check_outliers_no_flags_for_normal_moves():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    closes = [100, 101, 99, 102]

    outliers = check_outliers(_df(dates, closes), max_daily_move_pct=0.20)

    assert outliers.empty
