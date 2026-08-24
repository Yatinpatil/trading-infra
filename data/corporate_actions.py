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
from datetime import date, datetime

import pandas as pd

from data.retry import with_retries
from db.connection import connect

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


def _load_cached_actions(symbol: str) -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute(
            "SELECT ex_date, action_type, ratio FROM corporate_actions WHERE symbol = ? ORDER BY ex_date",
            (symbol,),
        ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows], columns=["ex_date", "action_type", "ratio"])
    if not df.empty:
        df["ex_date"] = pd.to_datetime(df["ex_date"])
    return df


def _save_actions(symbol: str, df: pd.DataFrame) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM corporate_actions WHERE symbol = ?", (symbol,))
        if not df.empty:
            rows = [
                (symbol, row.ex_date.strftime("%Y-%m-%d"), row.action_type, float(row.ratio))
                for row in df.itertuples(index=False)
            ]
            conn.executemany(
                "INSERT INTO corporate_actions (symbol, ex_date, action_type, ratio) VALUES (?,?,?,?)", rows
            )
        conn.execute(
            "INSERT INTO corporate_actions_fetch_log (symbol, fetched_at) VALUES (?, ?) "
            "ON CONFLICT(symbol) DO UPDATE SET fetched_at = excluded.fetched_at",
            (symbol, datetime.now().isoformat()),
        )


def _cache_age_days(symbol: str) -> float | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT fetched_at FROM corporate_actions_fetch_log WHERE symbol = ?", (symbol,)
        ).fetchone()
    if row is None:
        return None
    return (datetime.now() - datetime.fromisoformat(row["fetched_at"])).total_seconds() / 86400


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
    that empty cache back and skip the network entirely, silently treating
    "we couldn't reach NSE this one time" as "this symbol has never had a
    bonus or split".
    """
    symbol = symbol.upper()
    age_days = _cache_age_days(symbol) if use_cache else None
    if age_days is not None and age_days < max_cache_age_days:
        return _load_cached_actions(symbol)

    try:
        raw = _fetch_raw_corporate_actions(symbol)
        fetch_succeeded = True
    except Exception:
        if use_cache and age_days is not None:
            return _load_cached_actions(symbol)  # stale cache beats nothing
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
        _save_actions(symbol, df)

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
