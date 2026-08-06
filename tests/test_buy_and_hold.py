import pandas as pd

from strategies.buy_and_hold import BuyAndHoldStrategy


def test_entry_only_on_first_bar_no_exits():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {"OPEN": 100, "HIGH": 101, "LOW": 99, "CLOSE": 100, "VOLUME": 1000}, index=dates
    )

    signals = BuyAndHoldStrategy().generate_signals(df)

    assert signals["entry_long"].iloc[0]
    assert not signals["entry_long"].iloc[1:].any()
    assert not signals["exit_long"].any()
