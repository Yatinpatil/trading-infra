from datetime import datetime, timedelta

import pandas as pd
import pytest

import data.corporate_actions as corporate_actions_module
import db.connection as db_connection
from data.corporate_actions import _parse_action, adjust_for_corporate_actions, get_corporate_actions
from db.connection import connect


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


def test_parse_split_to_face_value_re_one():
    # NSE writes "Re 1/-" (singular rupee), not "Rs 1/-", when the new face
    # value is exactly 1 -- the overwhelmingly common case for a real split.
    # Confirmed against real TATASTEEL/NESTLEIND/KOTAKBANK/DRREDDY text.
    action_type, ratio = _parse_action(
        "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share"
    )
    assert action_type == "split"
    assert ratio == 10.0


def test_parse_split_without_per_share_before_to():
    # older announcements sometimes omit "Per Share" before "To"
    action_type, ratio = _parse_action("Face Value Split From Rs.10/- To Rs.5/-")
    assert action_type == "split"
    assert ratio == 2.0


def test_parse_dividend_returns_none():
    assert _parse_action("Dividend - Rs 6 Per Share") is None


def test_parse_bonus_debentures_is_not_mistaken_for_a_share_bonus():
    # bonus DEBENTURES are a debt distribution, not a share-count change --
    # must not match the bonus regex just because "Bonus ... N:1" appears.
    assert _parse_action("Scheme Of Arrangement - Bonus Debentures 6:1") is None


def test_parse_combined_bonus_and_split_in_one_subject_multiplies_ratios():
    # a single NSE announcement can bundle both (e.g. real BAJFINANCE 2016
    # text) -- the split component must not be dropped just because the
    # bonus regex matched first.
    action_type, ratio = _parse_action(
        "Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
    )
    assert action_type == "bonus+split"
    assert ratio == pytest.approx(2.0 * 5.0)


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


def test_fetch_requests_a_wide_explicit_date_range(monkeypatch):
    """Without from_date/to_date, NSE returns some recent-biased subset of a
    symbol's corporate action history rather than the full record --
    confirmed on TATASTEEL, where the undated call omitted a real 2022
    stock split entirely. Every fetch must pass an explicit wide range.
    """
    import jugaad_data.nse as nse_module

    captured = {}

    class FakeSession:
        def get(self, url, params, timeout):
            captured.update(params)

            class Resp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return []

            return Resp()

    class FakeNSELive:
        def __init__(self):
            self.s = FakeSession()

    monkeypatch.setattr(nse_module, "NSELive", FakeNSELive)

    corporate_actions_module._fetch_raw_corporate_actions("TATASTEEL")

    assert "from_date" in captured and "to_date" in captured
    assert captured["from_date"] == "01-01-2000"


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_connection, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("data.retry.time.sleep", lambda seconds: None)


def _backdate_fetch_log(symbol: str, days_ago: float) -> None:
    stale_time = (datetime.now() - timedelta(days=days_ago)).isoformat()
    with connect() as conn:
        conn.execute(
            "UPDATE corporate_actions_fetch_log SET fetched_at = ? WHERE symbol = ?", (stale_time, symbol)
        )


def test_a_failed_fetch_is_not_cached(monkeypatch):
    """A transient NSE failure must not be written to the cache -- otherwise
    every later call reads back the empty cache and never retries the
    network, permanently treating "couldn't reach NSE this once" as "this
    symbol has never had a bonus or split".
    """

    def always_fails(symbol):
        raise ConnectionError("NSE is down")

    monkeypatch.setattr(corporate_actions_module, "_fetch_raw_corporate_actions", always_fails)

    result = get_corporate_actions("RELIANCE", use_cache=True)

    assert result.empty
    with connect() as conn:
        row = conn.execute("SELECT 1 FROM corporate_actions_fetch_log WHERE symbol = 'RELIANCE'").fetchone()
    assert row is None


def test_a_successful_fetch_is_cached(monkeypatch):
    monkeypatch.setattr(
        corporate_actions_module,
        "_fetch_raw_corporate_actions",
        lambda symbol: [{"subject": "Bonus 1:1", "exDate": "01-Jan-2024"}],
    )

    result = get_corporate_actions("RELIANCE", use_cache=True)

    assert len(result) == 1
    with connect() as conn:
        row = conn.execute("SELECT 1 FROM corporate_actions_fetch_log WHERE symbol = 'RELIANCE'").fetchone()
    assert row is not None


def test_a_fresh_cache_is_served_without_hitting_the_network(monkeypatch):
    calls = {"n": 0}

    def counting_fetch(symbol):
        calls["n"] += 1
        return [{"subject": "Bonus 1:1", "exDate": "01-Jan-2024"}]

    monkeypatch.setattr(corporate_actions_module, "_fetch_raw_corporate_actions", counting_fetch)

    get_corporate_actions("RELIANCE", use_cache=True)
    get_corporate_actions("RELIANCE", use_cache=True)

    assert calls["n"] == 1  # second call served entirely from cache


def test_a_stale_cache_past_the_ttl_is_refreshed(monkeypatch):
    calls = {"n": 0}

    def counting_fetch(symbol):
        calls["n"] += 1
        return [{"subject": "Bonus 1:1", "exDate": "01-Jan-2024"}]

    monkeypatch.setattr(corporate_actions_module, "_fetch_raw_corporate_actions", counting_fetch)

    get_corporate_actions("RELIANCE", use_cache=True, max_cache_age_days=7)
    _backdate_fetch_log("RELIANCE", days_ago=8)

    get_corporate_actions("RELIANCE", use_cache=True, max_cache_age_days=7)

    assert calls["n"] == 2  # the stale cache triggered a refetch


def test_a_refetch_failure_falls_back_to_the_stale_cache(monkeypatch):
    monkeypatch.setattr(
        corporate_actions_module,
        "_fetch_raw_corporate_actions",
        lambda symbol: [{"subject": "Bonus 1:1", "exDate": "01-Jan-2024"}],
    )
    get_corporate_actions("RELIANCE", use_cache=True, max_cache_age_days=7)
    _backdate_fetch_log("RELIANCE", days_ago=8)

    def always_fails(symbol):
        raise ConnectionError("NSE is down")

    monkeypatch.setattr(corporate_actions_module, "_fetch_raw_corporate_actions", always_fails)

    result = get_corporate_actions("RELIANCE", use_cache=True, max_cache_age_days=7)

    assert len(result) == 1  # stale-but-known beats empty
