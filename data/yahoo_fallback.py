"""Last-resort Yahoo Finance fallback for a single missing EOD bar.

Used only from data/loaders.py's _fetch_trailing_range, only after every
NSE retry shape has already come back without the target date -- NSE
stays the authoritative source for everything else (historical fetches,
corporate actions, the whole rest of the pipeline). This exists so a day
NSE is unusually late (or simply never publishes) doesn't leave paper
trading silently stalled.

Yahoo's OHLC has NOT been vetted the way NSE's has -- see data/loaders.py's
own SERIES-filter and corporate-action bugs found earlier in the NSE
pipeline -- so a fetched bar is sanity-checked against the last known real
NSE close before being trusted. A same-day move past MAX_PLAUSIBLE_MOVE is
presumed to be a Yahoo data quirk (its OHLC is split-adjusted unless
auto_adjust=False is passed explicitly, which is easy to get wrong and
would show up as exactly this kind of implausible jump) rather than a real
price and is rejected -- returning None here just means the caller falls
through to its existing "no data today" behavior, not a crash.

A bar accepted here gets cached like any other (see get_raw_ohlcv's
upsert): if NSE later publishes the real value for that date, the next
fetch that covers it overwrites this one. What does NOT self-correct is
a paper-trading fill already made using this bar -- same as a real broker
fill, it's final once made.
"""
import logging
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

MAX_PLAUSIBLE_MOVE = 0.20  # vs. the last known real NSE close


def fetch_yahoo_ohlcv(symbol: str, target_date: date, prev_close: float) -> dict | None:
    try:
        import yfinance as yf

        df = yf.download(
            f"{symbol}.NS",
            start=target_date,
            end=target_date + pd.Timedelta(days=1),
            auto_adjust=False,  # raw OHLC, matching NSE's unadjusted convention -- not split/dividend adjusted
            progress=False,
        )
    except Exception:
        logger.warning("Yahoo fallback fetch failed for %s on %s", symbol, target_date, exc_info=True)
        return None

    if df.empty:
        return None

    row = df.iloc[-1]
    try:
        close = float(row[("Close", f"{symbol}.NS")])
        bar = {
            "OPEN": float(row[("Open", f"{symbol}.NS")]),
            "HIGH": float(row[("High", f"{symbol}.NS")]),
            "LOW": float(row[("Low", f"{symbol}.NS")]),
            "CLOSE": close,
            "VOLUME": int(row[("Volume", f"{symbol}.NS")]),
        }
    except (KeyError, ValueError, TypeError):
        logger.warning("Yahoo fallback returned an unexpected shape for %s on %s", symbol, target_date)
        return None

    if prev_close and abs(close / prev_close - 1.0) > MAX_PLAUSIBLE_MOVE:
        logger.warning(
            "Yahoo fallback for %s on %s rejected: close %.2f vs. last real NSE close %.2f "
            "implies a %.0f%% move, past the %.0f%% plausibility bound (likely a split-adjustment "
            "mismatch or other Yahoo data quirk, not a real price)",
            symbol, target_date, close, prev_close, (close / prev_close - 1.0) * 100, MAX_PLAUSIBLE_MOVE * 100,
        )
        return None

    logger.warning("Used Yahoo Finance fallback for %s on %s: NSE never returned this day", symbol, target_date)
    return bar
