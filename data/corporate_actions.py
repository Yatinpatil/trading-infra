"""Corporate action fetch/parse and price adjustment.

NSE's raw daily OHLCV is unadjusted, so a 1:1 bonus or a face-value split
shows up as a ~50% overnight "crash" that will wreck any indicator computed
across the event. This module fetches the corporate action calendar for a
symbol and derives a cumulative backward-adjustment factor: prices before an
ex-date are scaled down so the whole series is expressed in today's terms.

Only bonus issues and face-value splits are adjusted (they mechanically
change share count/face value and must be adjusted for price continuity).
Dividends are intentionally left unadjusted — that's a price-return vs.
total-return choice, not a bug; see get_corporate_actions docstring.
"""
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd

from data.retry import with_retries

CACHE_DIR = Path(__file__).parent / "cache"

BONUS_RE = re.compile(r"Bonus\s+(\d+)\s*:\s*(\d+)", re.IGNORECASE)
# NSE writes "Re 1/-" (singular rupee), not "Rs 1/-", whenever the new face
# value is exactly 1 -- which is the overwhelmingly common case for a split
# ("From Rs 10/- ... To Re 1/-"). Matching only "Rs" silently missed almost
# every real split (confirmed on TATASTEEL/NESTLEIND/KOTAKBANK/DRREDDY).
# "Per Share" before "To" is also optional -- some older announcements omit it.
SPLIT_RE = re.compile(
    r"Face\s*Value\s*Split.*?R[se]\.?\s*([\d.]+)\s*/?-?\s*(?:Per\s*Share\s*)?To\s*R[se]\.?\s*([\d.]+)",
    re.IGNORECASE,
)


def _actions_cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_corp_actions.parquet"


def _parse_action(subject: str) -> tuple[str, float] | None:
    """Return (action_type, adjustment_ratio) or None if `subject` isn't a bonus/split.

    adjustment_ratio is the multiple by which share count increases (bonus)
    or price mechanically drops (split): pre-event price / ratio == post-event price.

    A single NSE announcement can bundle both in one subject string (e.g.
    "Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To
    Rs 2/- Per Share") — both regexes are checked independently and their
    ratios multiplied, rather than returning on the first match, or the
    split component would be silently dropped.
    """
    ratio = 1.0
    matched_types = []

    m = BONUS_RE.search(subject)
    if m:
        num, denom = float(m.group(1)), float(m.group(2))
        if denom > 0:
            ratio *= 1.0 + num / denom
            matched_types.append("bonus")

    m = SPLIT_RE.search(subject)
    if m:
        old_fv, new_fv = float(m.group(1)), float(m.group(2))
        if new_fv > 0 and old_fv > new_fv:
            ratio *= old_fv / new_fv
            matched_types.append("split")

    if not matched_types:
        return None
    return "+".join(matched_types), ratio


def _fetch_raw_corporate_actions(symbol: str) -> list[dict]:
    from jugaad_data.nse import NSELive

    def _do_fetch():
        n = NSELive()
        resp = n.s.get(
            "https://www.nseindia.com/api/corporates-corporateActions",
            params={
                "index": "equities",
                "symbol": symbol,
                # Without an explicit date range, NSE returns some
                # recent-biased subset of a symbol's corporate action
                # history instead of the full record — confirmed on
                # TATASTEEL, where the undated call omitted a real 2022
                # stock split entirely. A wide, fixed range gets everything.
                "from_date": "01-01-2000",
                "to_date": date.today().strftime("%d-%m-%Y"),
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    return with_retries(_do_fetch)


def get_corporate_actions(symbol: str, use_cache: bool = True, max_cache_age_days: float = 7.0) -> pd.DataFrame:
    """DataFrame[ex_date, action_type, ratio] of bonus/split events for `symbol`, oldest first.

    Unlike the OHLCV cache (which extends its date range over time), a
    corporate-action calendar has no natural "range" to extend — so the
    cache carries a TTL (`max_cache_age_days`) instead: once it's older than
    that, this refetches, since NSE keeps announcing new actions for a
    symbol for as long as it keeps trading and a paper/live account can stay
    open for months. A fetch failure (even after retries) falls back to the
    existing cache if there is one — stale data beats none — and only
    returns empty if there's truly nothing cached yet. Either way, a failed
    fetch is never itself written to the cache: caching an empty result on
    failure would permanently poison it, since every later call would read
    that empty parquet back and skip the network entirely, silently treating
    "we couldn't reach NSE this one time" as "this symbol has never had a
    bonus or split".
    """
    path = _actions_cache_path(symbol)
    if use_cache and path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days < max_cache_age_days:
            return pd.read_parquet(path)

    try:
        raw = _fetch_raw_corporate_actions(symbol)
        fetch_succeeded = True
    except Exception:
        if use_cache and path.exists():
            return pd.read_parquet(path)  # stale cache beats nothing
        raw = []
        fetch_succeeded = False

    rows = []
    for entry in raw:
        parsed = _parse_action(entry.get("subject", ""))
        if parsed is None:
            continue
        ex_date_str = entry.get("exDate", "-")
        if ex_date_str == "-" or not ex_date_str:
            continue
        try:
            ex_date = pd.to_datetime(ex_date_str, format="%d-%b-%Y")
        except ValueError:
            continue
        action_type, ratio = parsed
        rows.append({"ex_date": ex_date, "action_type": action_type, "ratio": ratio})

    df = pd.DataFrame(rows, columns=["ex_date", "action_type", "ratio"])
    df = df.sort_values("ex_date").reset_index(drop=True)

    if use_cache and fetch_succeeded:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    return df


def adjust_for_corporate_actions(ohlcv: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Apply cumulative backward split/bonus adjustment to OPEN/HIGH/LOW/CLOSE.

    VOLUME is scaled up by the same factor (more shares trade post-bonus/split
    for the same rupee turnover), so traded-value-based liquidity filters stay
    consistent across the adjustment.
    """
    if actions.empty:
        return ohlcv.copy()

    adjusted = ohlcv.copy()
    factor = pd.Series(1.0, index=adjusted.index)

    for _, row in actions.iterrows():
        pre_event = adjusted.index < row["ex_date"]
        factor.loc[pre_event] *= row["ratio"]

    for col in ["OPEN", "HIGH", "LOW", "CLOSE"]:
        adjusted[col] = adjusted[col] / factor
    adjusted["VOLUME"] = (adjusted["VOLUME"] * factor).round().astype("int64")

    return adjusted
