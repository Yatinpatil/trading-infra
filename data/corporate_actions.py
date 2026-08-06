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
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"

BONUS_RE = re.compile(r"Bonus\s+(\d+)\s*:\s*(\d+)", re.IGNORECASE)
SPLIT_RE = re.compile(
    r"Face\s*Value\s*Split.*?Rs\.?\s*([\d.]+)\s*/?-?\s*Per\s*Share\s*To\s*Rs\.?\s*([\d.]+)",
    re.IGNORECASE,
)


def _actions_cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_corp_actions.parquet"


def _parse_action(subject: str) -> tuple[str, float] | None:
    """Return (action_type, adjustment_ratio) or None if `subject` isn't a bonus/split.

    adjustment_ratio is the multiple by which share count increases (bonus)
    or price mechanically drops (split): pre-event price / ratio == post-event price.
    """
    m = BONUS_RE.search(subject)
    if m:
        num, denom = float(m.group(1)), float(m.group(2))
        if denom > 0:
            return "bonus", 1.0 + num / denom

    m = SPLIT_RE.search(subject)
    if m:
        old_fv, new_fv = float(m.group(1)), float(m.group(2))
        if new_fv > 0 and old_fv > new_fv:
            return "split", old_fv / new_fv

    return None


def _fetch_raw_corporate_actions(symbol: str) -> list[dict]:
    from jugaad_data.nse import NSELive

    n = NSELive()
    resp = n.s.get(
        "https://www.nseindia.com/api/corporates-corporateActions",
        params={"index": "equities", "symbol": symbol},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def get_corporate_actions(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    """DataFrame[ex_date, action_type, ratio] of bonus/split events for `symbol`, oldest first."""
    path = _actions_cache_path(symbol)
    if use_cache and path.exists():
        return pd.read_parquet(path)

    try:
        raw = _fetch_raw_corporate_actions(symbol)
    except Exception:
        raw = []

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

    if use_cache:
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
