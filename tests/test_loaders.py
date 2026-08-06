"""Offline tests for data/loaders.py — network calls are mocked out so these
run in the default (non-network) suite.
"""
import pandas as pd
import pytest

import data.loaders as loaders_module


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
    monkeypatch.setattr(loaders_module, "CACHE_DIR", tmp_path)
    return tmp_path


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
