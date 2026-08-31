"""OHLCV and corporate-action loaders for NSE equities, backed by jugaad-data.

All fetched data is cached in the project's SQLite store (db/) so repeated
backtest runs don't re-hit NSE. Rows are keyed by (symbol, date); callers
ask for a date range and the cache is extended (not re-fetched) when the
requested range grows.
"""
import logging
from datetime import date, datetime

import pandas as pd

from data.corporate_actions import adjust_for_corporate_actions, get_corporate_actions
from data.retry import with_retries
from data.yahoo_fallback import fetch_yahoo_ohlcv
from db.connection import connect

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]


def _to_date(d) -> date:
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    if isinstance(d, datetime):
        return d.date()
    return d


def _fetch_raw_ohlcv(symbol: str, start: date, end: date) -> pd.DataFrame:
    from jugaad_data.nse import stock_df

    raw = with_retries(lambda: stock_df(symbol=symbol, from_date=start, to_date=end, series="EQ"))
    if raw.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS).set_index(
            pd.DatetimeIndex([], name="DATE")
        )

    # stock_df's `series` kwarg is NOT honored server-side: NSE returns every
    # series traded under this symbol (equity plus unrelated NCD/bond series
    # like N1..ND, which share the same NSE symbol but trade at wildly
    # different price levels). Filtering here is required, not defensive —
    # without it, the later dedup-by-date keeps whichever series happens to
    # sort last for each date, silently splicing bond prices into the equity
    # series (confirmed against NTPC: unfiltered data alternates between
    # ~100 and ~1400 day to day).
    raw = raw[raw["SERIES"] == "EQ"]
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
    with connect() as conn:
        rows = conn.execute(
            "SELECT date, open, high, low, close, volume FROM ohlcv WHERE symbol = ? ORDER BY date",
            (symbol.upper(),),
        ).fetchall()
    if not rows:
        return None
    df = pd.DataFrame(
        [(r["date"], r["open"], r["high"], r["low"], r["close"], r["volume"]) for r in rows],
        columns=["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"],
    )
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df.set_index("DATE").astype(
        {"OPEN": "float64", "HIGH": "float64", "LOW": "float64", "CLOSE": "float64", "VOLUME": "int64"}
    )


def _save_cache(symbol: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    rows = [
        (symbol.upper(), idx.strftime("%Y-%m-%d"), float(row.OPEN), float(row.HIGH), float(row.LOW), float(row.CLOSE), int(row.VOLUME))
        for idx, row in zip(df.index, df.itertuples(index=False))
    ]
    with connect() as conn:
        conn.executemany(
            "INSERT INTO ohlcv (symbol, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(symbol, date) DO UPDATE SET "
            "open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume",
            rows,
        )


# NSE's history endpoint has been observed to intermittently drop the most
# recent day for some specific requested date ranges while returning it
# fine for others -- e.g. (2026-08-21, 2026-08-24) reproducibly missed the
# 24th while (2026-08-20, 2026-08-24) didn't, and on a later day the exact
# opposite held for a 5-day- vs. 3-day-wide window. Which range is affected
# shifts, so no single fixed padding is reliable -- retrying with a few
# genuinely different range shapes is what actually recovers the missing
# day in practice.
_TRAILING_FETCH_OFFSETS_DAYS = (5, 3, 7, 2, 14)


def _fetch_trailing_range(
    symbol: str, cached_end: date, end: date, prev_close: float | None, allow_yahoo_fallback: bool
) -> pd.DataFrame:
    result = pd.DataFrame(columns=OHLCV_COLUMNS)
    for attempt, offset_days in enumerate(_TRAILING_FETCH_OFFSETS_DAYS):
        candidate_start = min(cached_end, end - pd.Timedelta(days=offset_days))
        result = _fetch_raw_ohlcv(symbol, candidate_start, end)
        if not result.empty and end in result.index.date:
            if attempt > 0:
                logger.warning(
                    "%s: NSE dropped %s from %d earlier fetch attempt(s), recovered on retry",
                    symbol, end, attempt,
                )
            return result
    logger.warning("%s: NSE never returned %s across %d attempts", symbol, end, len(_TRAILING_FETCH_OFFSETS_DAYS))

    if not allow_yahoo_fallback:
        return result

    bar = fetch_yahoo_ohlcv(symbol, end, prev_close)
    if bar is not None:
        fallback_row = pd.DataFrame([bar], index=pd.DatetimeIndex([pd.Timestamp(end)]))
        result = pd.concat([result, fallback_row])
        result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def get_raw_ohlcv(symbol: str, start, end, use_cache: bool = True, allow_yahoo_fallback: bool = True) -> pd.DataFrame:
    """Unadjusted OHLCV as reported by NSE, extended/cached across calls.

    `allow_yahoo_fallback` gates data/yahoo_fallback.py's last-resort use --
    it must stay off for any "has NSE published yet" check (see
    scripts/poll_and_run_paper_trading.py), or that check would itself
    silently accept a Yahoo bar as "NSE is out," triggering the real step
    hours before NSE would ever actually publish and permanently locking in
    the less-authoritative source for the whole universe every day instead
    of only the rare day NSE genuinely never delivers.
    """
    start, end = _to_date(start), _to_date(end)
    cached = _load_cache(symbol) if use_cache else None

    if cached is not None and not cached.empty:
        cached_start, cached_end = cached.index.min().date(), cached.index.max().date()
        fetched = [cached]
        if start < cached_start:
            fetched.append(_fetch_raw_ohlcv(symbol, start, cached_start))
        if end > cached_end:
            prev_close = float(cached.loc[pd.Timestamp(cached_end), "CLOSE"])
            fetched.append(_fetch_trailing_range(symbol, cached_end, end, prev_close, allow_yahoo_fallback))

        combined = pd.concat(fetched)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = _fetch_raw_ohlcv(symbol, start, end)

    if use_cache and not combined.empty:
        _save_cache(symbol, combined)

    mask = (combined.index.date >= start) & (combined.index.date <= end)
    return combined.loc[mask].copy()


def get_ohlcv(
    symbol: str, start, end, adjust: bool = True, use_cache: bool = True, allow_yahoo_fallback: bool = True
) -> pd.DataFrame:
    """OHLCV for `symbol` between `start` and `end` (inclusive), split/bonus-adjusted by default.

    Adjustment is backward (today's prices are truth; history is scaled down),
    the standard convention so indicators computed over the series don't see
    fake jumps at ex-dates.
    """
    raw = get_raw_ohlcv(symbol, start, end, use_cache=use_cache, allow_yahoo_fallback=allow_yahoo_fallback)
    if raw.empty or not adjust:
        return raw

    actions = get_corporate_actions(symbol, use_cache=use_cache)
    return adjust_for_corporate_actions(raw, actions)
