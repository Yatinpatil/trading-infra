import datetime

import pandas as pd

from data.yahoo_fallback import fetch_yahoo_ohlcv

SYMBOL = "RELIANCE"
TARGET = datetime.date(2026, 8, 26)


def _mock_yf_download(open_, high, low, close, volume):
    columns = pd.MultiIndex.from_tuples(
        [("Adj Close", f"{SYMBOL}.NS"), ("Close", f"{SYMBOL}.NS"), ("High", f"{SYMBOL}.NS"),
         ("Low", f"{SYMBOL}.NS"), ("Open", f"{SYMBOL}.NS"), ("Volume", f"{SYMBOL}.NS")]
    )
    return pd.DataFrame(
        [[close, close, high, low, open_, volume]],
        columns=columns,
        index=pd.DatetimeIndex([pd.Timestamp(TARGET)]),
    )


def test_accepts_a_plausible_bar(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(yf, "download", lambda *a, **k: _mock_yf_download(1310.0, 1315.6, 1298.0, 1298.0, 5_735_384))

    bar = fetch_yahoo_ohlcv(SYMBOL, TARGET, prev_close=1317.0)

    assert bar == {"OPEN": 1310.0, "HIGH": 1315.6, "LOW": 1298.0, "CLOSE": 1298.0, "VOLUME": 5_735_384}


def test_rejects_a_bar_implying_an_implausible_move(monkeypatch):
    """A close far from the last known real NSE close is presumed to be a
    split-adjustment mismatch or other Yahoo data quirk, not a real price
    -- e.g. Yahoo's OHLC comes back split-adjusted unless auto_adjust=False
    is passed, which would show up as exactly this kind of jump.
    """
    import yfinance as yf

    monkeypatch.setattr(yf, "download", lambda *a, **k: _mock_yf_download(660.0, 665.0, 655.0, 658.0, 5_000_000))

    bar = fetch_yahoo_ohlcv(SYMBOL, TARGET, prev_close=1317.0)  # implies a ~50% drop -- e.g. an unadjusted 2:1 split

    assert bar is None


def test_accepts_without_a_sanity_check_when_no_prev_close_is_known(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(yf, "download", lambda *a, **k: _mock_yf_download(100.0, 101.0, 99.0, 100.5, 1000))

    bar = fetch_yahoo_ohlcv(SYMBOL, TARGET, prev_close=None)

    assert bar is not None


def test_returns_none_on_empty_response(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(yf, "download", lambda *a, **k: pd.DataFrame())

    assert fetch_yahoo_ohlcv(SYMBOL, TARGET, prev_close=1317.0) is None


def test_returns_none_rather_than_raising_on_a_fetch_error(monkeypatch):
    import yfinance as yf

    def raises(*args, **kwargs):
        raise RuntimeError("network hiccup")

    monkeypatch.setattr(yf, "download", raises)

    assert fetch_yahoo_ohlcv(SYMBOL, TARGET, prev_close=1317.0) is None
