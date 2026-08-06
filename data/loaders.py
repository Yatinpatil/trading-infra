"""OHLCV and corporate-action loaders for NSE equities, backed by jugaad-data.

All fetched data is cached locally as Parquet so repeated backtest runs don't
re-hit NSE. Cache files are keyed by symbol; callers ask for a date range and
the cache is extended (not re-fetched) when the requested range grows.
"""
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from data.corporate_actions import adjust_for_corporate_actions, get_corporate_actions

CACHE_DIR = Path(__file__).parent / "cache"

OHLCV_COLUMNS = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]


def _to_date(d) -> date:
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    if isinstance(d, datetime):
        return d.date()
    return d


def _ohlcv_cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_ohlcv.parquet"


def _fetch_raw_ohlcv(symbol: str, start: date, end: date) -> pd.DataFrame:
    from jugaad_data.nse import stock_df

    raw = stock_df(symbol=symbol, from_date=start, to_date=end, series="EQ")
    if raw.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS).set_index(
            pd.DatetimeIndex([], name="DATE")
        )

    raw = raw.rename(columns={"PREV. CLOSE": "PREV_CLOSE"})
    # jugaad-data's DATE is IST midnight expressed in naive UTC (18:30:00 the
    # calendar day before) — normalizing without the +5:30 correction silently
    # shifts every trading date back by one day.
    raw["DATE"] = (pd.to_datetime(raw["DATE"]) + pd.Timedelta(hours=5, minutes=30)).dt.normalize()
    raw = raw.set_index("DATE").sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]  # defend against dupe rows in a flaky NSE response
    return raw[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]].astype(
        {"OPEN": "float64", "HIGH": "float64", "LOW": "float64", "CLOSE": "float64", "VOLUME": "int64"}
    )


def _load_cache(symbol: str) -> pd.DataFrame | None:
    path = _ohlcv_cache_path(symbol)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _save_cache(symbol: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_ohlcv_cache_path(symbol))


def get_raw_ohlcv(symbol: str, start, end, use_cache: bool = True) -> pd.DataFrame:
    """Unadjusted OHLCV as reported by NSE, extended/cached across calls."""
    start, end = _to_date(start), _to_date(end)
    cached = _load_cache(symbol) if use_cache else None

    if cached is not None and not cached.empty:
        cached_start, cached_end = cached.index.min().date(), cached.index.max().date()
        missing_ranges = []
        if start < cached_start:
            missing_ranges.append((start, cached_start))
        if end > cached_end:
            missing_ranges.append((cached_end, end))

        fetched = [cached]
        for m_start, m_end in missing_ranges:
            fetched.append(_fetch_raw_ohlcv(symbol, m_start, m_end))
        combined = pd.concat(fetched)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = _fetch_raw_ohlcv(symbol, start, end)

    if use_cache and not combined.empty:
        _save_cache(symbol, combined)

    mask = (combined.index.date >= start) & (combined.index.date <= end)
    return combined.loc[mask].copy()


def get_ohlcv(symbol: str, start, end, adjust: bool = True, use_cache: bool = True) -> pd.DataFrame:
    """OHLCV for `symbol` between `start` and `end` (inclusive), split/bonus-adjusted by default.

    Adjustment is backward (today's prices are truth; history is scaled down),
    the standard convention so indicators computed over the series don't see
    fake jumps at ex-dates.
    """
    raw = get_raw_ohlcv(symbol, start, end, use_cache=use_cache)
    if raw.empty or not adjust:
        return raw

    actions = get_corporate_actions(symbol, use_cache=use_cache)
    return adjust_for_corporate_actions(raw, actions)
