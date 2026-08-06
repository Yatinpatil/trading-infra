import pandas as pd
import pytest

from data.corporate_actions import _parse_action, adjust_for_corporate_actions


def test_parse_bonus():
    action_type, ratio = _parse_action("Bonus 1:1")
    assert action_type == "bonus"
    assert ratio == 2.0


def test_parse_bonus_uneven_ratio():
    action_type, ratio = _parse_action("Bonus 2:3")
    assert action_type == "bonus"
    assert ratio == pytest.approx(1 + 2 / 3)


def test_parse_split():
    action_type, ratio = _parse_action(
        "Face Value Split (Sub-Divide) - From Rs 10/- Per Share To Rs 2/-"
    )
    assert action_type == "split"
    assert ratio == 5.0


def test_parse_dividend_returns_none():
    assert _parse_action("Dividend - Rs 6 Per Share") is None


def test_adjust_for_bonus_removes_price_jump():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    ohlcv = pd.DataFrame(
        {
            "OPEN": [200.0, 202.0, 100.0, 101.0],
            "HIGH": [205.0, 206.0, 103.0, 104.0],
            "LOW": [198.0, 199.0, 99.0, 100.0],
            "CLOSE": [202.0, 200.0, 101.0, 102.0],
            "VOLUME": [1000, 1000, 2000, 2000],
        },
        index=dates,
    )
    # 1:1 bonus effective on the 3rd day: pre-event prices should halve.
    actions = pd.DataFrame(
        {"ex_date": [dates[2]], "action_type": ["bonus"], "ratio": [2.0]}
    )

    adjusted = adjust_for_corporate_actions(ohlcv, actions)

    assert adjusted["CLOSE"].iloc[0] == 101.0
    assert adjusted["CLOSE"].iloc[1] == 100.0
    assert adjusted["CLOSE"].iloc[2] == 101.0  # post-event, unchanged
    assert adjusted["CLOSE"].iloc[3] == 102.0
    assert adjusted["VOLUME"].iloc[0] == 2000
    assert adjusted["VOLUME"].iloc[2] == 2000  # post-event, unchanged


def test_adjust_with_no_actions_is_noop():
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    ohlcv = pd.DataFrame(
        {"OPEN": [10.0, 11.0], "HIGH": [10.0, 11.0], "LOW": [10.0, 11.0], "CLOSE": [10.0, 11.0], "VOLUME": [1, 1]},
        index=dates,
    )
    actions = pd.DataFrame(columns=["ex_date", "action_type", "ratio"])

    adjusted = adjust_for_corporate_actions(ohlcv, actions)

    pd.testing.assert_frame_equal(adjusted, ohlcv)
