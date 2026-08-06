"""Integration tests against live NSE endpoints. Excluded from default runs
(see pytest.ini addopts) — run explicitly with `pytest -m network`.
"""
import shutil

import pytest

from data.loaders import get_ohlcv, get_raw_ohlcv

pytestmark = pytest.mark.network

SAMPLE_SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]


@pytest.fixture
def no_cache(tmp_path, monkeypatch):
    import data.corporate_actions as ca_module
    import data.loaders as loaders_module

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(loaders_module, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(ca_module, "CACHE_DIR", cache_dir)
    yield
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.mark.parametrize("symbol", SAMPLE_SYMBOLS)
def test_get_ohlcv_returns_data_for_sample_stocks(no_cache, symbol):
    df = get_ohlcv(symbol, "2024-01-01", "2024-01-31")

    assert not df.empty
    assert list(df.columns) == ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    assert df["CLOSE"].gt(0).all()


def test_get_raw_ohlcv_caches_and_extends_range(no_cache):
    first = get_raw_ohlcv("RELIANCE", "2024-01-10", "2024-01-20")
    assert not first.empty

    extended = get_raw_ohlcv("RELIANCE", "2024-01-01", "2024-01-31")
    assert extended.index.min() <= first.index.min()
    assert extended.index.max() >= first.index.max()
