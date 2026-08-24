"""Offline tests for data/loaders.py — network calls are mocked out so these
run in the default (non-network) suite.
"""
import pandas as pd
import pytest

import data.loaders as loaders_module
import db.connection as db_connection


def _mock_stock_df(rows):
    """rows: list of (date_str, open, high, low, close, volume), mimicking
    jugaad_data.nse.stock_df's raw shape (DATE at 18:30:00 = IST midnight
    of the next day, per the real API's quirk).
    """
    return pd.DataFrame(
        {
            "DATE": [pd.Timestamp(d) + pd.Timedelta(hours=18, minutes=30) - pd.Timedelta(days=1) for d, *_ in rows],
            "SERIES": "EQ",
            "OPEN": [r[1] for r in rows],
            "HIGH": [r[2] for r in rows],
            "LOW": [r[3] for r in rows],
            "PREV. CLOSE": [r[1] for r in rows],
            "CLOSE": [r[4] for r in rows],
            "VOLUME": [r[5] for r in rows],
        }
    )


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(db_connection, "DB_PATH", tmp_path / "test.db")
    return tmp_path


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr("data.retry.time.sleep", lambda seconds: None)


def test_non_equity_series_rows_are_filtered_out(monkeypatch, isolated_cache):
    """stock_df's `series="EQ"` kwarg is NOT honored server-side -- NSE
    returns every series traded under the symbol, including unrelated
    NCD/bond series (e.g. NTPC's N1..ND) that share the same symbol but
    trade at a wildly different price level. Confirmed against real NTPC
    data: without this filter, dedup-by-date splices bond prices into the
    equity series, alternating between ~100 and ~1400 day to day.
    """
    import jugaad_data.nse as nse_module

    mixed = pd.DataFrame(
        {
            "DATE": [
                pd.Timestamp("2021-01-05") + pd.Timedelta(hours=18, minutes=30) - pd.Timedelta(days=1),
                pd.Timestamp("2021-01-05") + pd.Timedelta(hours=18, minutes=30) - pd.Timedelta(days=1),
            ],
            "SERIES": ["N6", "EQ"],  # bond series row happens to sort after the real equity row
            "OPEN": [1440.0, 98.0],
            "HIGH": [1445.0, 98.5],
            "LOW": [1430.0, 97.0],
            "PREV. CLOSE": [1435.0, 97.5],
            "CLOSE": [1439.0, 97.85],
            "VOLUME": [500, 24_000_000],
        }
    )
    monkeypatch.setattr(nse_module, "stock_df", lambda **kwargs: mixed)

    result = loaders_module.get_raw_ohlcv("NTPC", "2021-01-01", "2021-01-10", use_cache=False)

    assert len(result) == 1
    assert result["CLOSE"].iloc[0] == 97.85
    assert result["VOLUME"].iloc[0] == 24_000_000


def test_fetch_corrects_date_offset(monkeypatch, isolated_cache):
    import jugaad_data.nse as nse_module

    mock_df = _mock_stock_df([("2024-01-25", 10, 11, 9, 10.5, 100)])
    monkeypatch.setattr(nse_module, "stock_df", lambda **kwargs: mock_df)

    result = loaders_module.get_raw_ohlcv("TEST", "2024-01-20", "2024-01-31", use_cache=False)

    assert result.index[0] == pd.Timestamp("2024-01-25")


def test_fetch_dedupes_repeated_dates(monkeypatch, isolated_cache):
    import jugaad_data.nse as nse_module

    rows = [
        ("2024-01-25", 10, 11, 9, 10.5, 100),
        ("2024-01-25", 10, 11, 9, 10.5, 100),  # duplicate row, as if the API returned it twice
        ("2024-01-26", 11, 12, 10, 11.5, 100),
    ]
    monkeypatch.setattr(nse_module, "stock_df", lambda **kwargs: _mock_stock_df(rows))

    result = loaders_module.get_raw_ohlcv("TEST", "2024-01-20", "2024-01-31", use_cache=False)

    assert len(result) == 2
    assert not result.index.duplicated().any()


def test_fetch_retries_a_transient_failure_before_succeeding(monkeypatch, isolated_cache):
    import jugaad_data.nse as nse_module

    calls = {"n": 0}
    mock_df = _mock_stock_df([("2024-01-25", 10, 11, 9, 10.5, 100)])

    def flaky_stock_df(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("NSE hiccup")
        return mock_df

    monkeypatch.setattr(nse_module, "stock_df", flaky_stock_df)

    result = loaders_module.get_raw_ohlcv("TEST", "2024-01-20", "2024-01-31", use_cache=False)

    assert calls["n"] == 3
    assert result.index[0] == pd.Timestamp("2024-01-25")


def test_fetch_raises_after_exhausting_retries(monkeypatch, isolated_cache):
    import jugaad_data.nse as nse_module

    def always_fails(**kwargs):
        raise ConnectionError("NSE is down")

    monkeypatch.setattr(nse_module, "stock_df", always_fails)

    with pytest.raises(ConnectionError, match="NSE is down"):
        loaders_module.get_raw_ohlcv("TEST", "2024-01-20", "2024-01-31", use_cache=False)


def test_get_raw_ohlcv_serves_subset_range_from_cache_without_refetching(monkeypatch, isolated_cache):
    import jugaad_data.nse as nse_module

    call_count = {"n": 0}

    def fake_stock_df(**kwargs):
        call_count["n"] += 1
        return _mock_stock_df(
            [
                ("2024-01-20", 10, 11, 9, 10.5, 100),
                ("2024-01-25", 11, 12, 10, 11.5, 100),
                ("2024-01-30", 12, 13, 11, 12.5, 100),
            ]
        )

    monkeypatch.setattr(nse_module, "stock_df", fake_stock_df)

    loaders_module.get_raw_ohlcv("TEST", "2024-01-20", "2024-01-30", use_cache=True)
    # second call asks for a range fully inside what's already cached
    result = loaders_module.get_raw_ohlcv("TEST", "2024-01-22", "2024-01-28", use_cache=True)

    assert call_count["n"] == 1  # second call served entirely from cache
    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2024-01-25")
